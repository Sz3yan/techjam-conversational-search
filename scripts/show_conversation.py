"""Replay public sessions and print the conversation as it actually happens.

Read-only debugging aid: it drives `starter.agent.Agent` through the exact loop
`evaluator.local_evaluator.evaluate` uses -- same simulated customer, same
override timing, same stop condition -- but narrates every turn instead of
collapsing the session into a metric. Scores printed here therefore match
`results.json` for the same sessions.

    python3 -m scripts.show_conversation --limit 3
    python3 -m scripts.show_conversation --sample-id public_0007 --verbose
    python3 -m scripts.show_conversation --scenario intent_override --only misses
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent


class Palette:
    """ANSI codes, or empty strings when colour is off."""

    NAMES = {
        "user": "\033[36m",      # cyan
        "agent": "\033[35m",     # magenta
        "hit": "\033[32m",       # green
        "miss": "\033[31m",      # red
        "dim": "\033[2m",
        "bold": "\033[1m",
        "target": "\033[33m",    # yellow
        "reset": "\033[0m",
    }

    def __init__(self, enabled: bool) -> None:
        for name, code in self.NAMES.items():
            setattr(self, name, code if enabled else "")


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def wrap(text: str, indent: str, width: int) -> str:
    """Wrap `text` to `width`, indenting every line after the first."""
    words, lines, current = str(text).split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return f"\n{indent}".join(lines) if lines else ""


def replay(
    agent: Agent,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    """Run one session, capturing every turn. Mirrors `evaluate`'s inner loop."""
    session_id = f"public_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

    turns: list[dict] = []
    hit_turn: int | None = None
    best_rank: int | None = None
    prompt_tokens = completion_tokens = 0

    for turn in range(1, MAX_TURNS + 1):
        error = None
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception as exc:  # the evaluator swallows this as a dead turn
            error = f"{type(exc).__name__}: {exc}"
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            error = error or "malformed response (evaluator would discard this turn)"
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        usage = response.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                prompt_tokens += usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
                completion_tokens += usage["completion_tokens"]

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        record = {
            "turn": turn,
            "user_message": user_message,
            "agent_message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": ranked,
            "dropped": max(0, len(response.get("recommendations") or []) - len(ranked)),
            "error": error,
            "note": None,
        }
        turns.append(record)

        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            record["note"] = f"target at rank {best_rank} -- session ends"
            break
        if target in ranked and not override_applied:
            record["note"] = "target shown, but the override has not landed yet: no credit"
        if turn == MAX_TURNS:
            record["note"] = "turn limit reached"
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "difficulty_bucket": sample.get("difficulty_bucket"),
        "user_profile": sample["user_profile"],
        "target": target,
        "target_title": str(products.get(target, {}).get("title", "<not in catalog>")),
        "intent_card": card,
        "behavior": behavior,
        "turns": turns,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def render(session: dict, products: dict[str, dict], colour: Palette, args: argparse.Namespace) -> None:
    width = args.width
    body = width - 4
    print(f"\n{colour.bold}{'=' * width}{colour.reset}")
    header = f"{session['sample_id']}  [{session['scenario_type']}"
    if session["difficulty_bucket"]:
        header += f" / {session['difficulty_bucket']}"
    print(f"{colour.bold}{header}]{colour.reset}")
    if not args.hide_target:
        print(f"{colour.target}target {session['target']}{colour.reset}  {truncate(session['target_title'], body - 20)}")
    if args.verbose:
        profile = session["user_profile"].get("summary") or json.dumps(session["user_profile"])
        print(f"{colour.dim}profile: {wrap(profile, '         ', body)}{colour.reset}")
        card = session["intent_card"]
        print(f"{colour.dim}hidden constraints: {truncate('; '.join(card.get('hard_constraints', [])), body)}{colour.reset}")
        print(f"{colour.dim}hidden preferences: {truncate('; '.join(card.get('soft_preferences', [])), body)}{colour.reset}")
        override = session["behavior"].get("override")
        if override:
            print(f"{colour.dim}override lands on turn {override.get('turn')}{colour.reset}")
    print(f"{colour.bold}{'=' * width}{colour.reset}")

    for record in session["turns"]:
        print(f"\n{colour.dim}--- turn {record['turn']} ---{colour.reset}")
        print(f"{colour.user}customer{colour.reset}  {wrap(record['user_message'], '          ', body)}")
        message = record["agent_message"] or f"{colour.dim}(no message){colour.reset}"
        print(f"{colour.agent}agent   {colour.reset}  {wrap(message, '          ', body)}")
        ask = record["ask_attribute"]
        print(f"{colour.dim}          asks about: {ask if ask else 'nothing'}{colour.reset}")
        if record["error"]:
            print(f"{colour.miss}          error: {record['error']}{colour.reset}")
        if not record["recommendations"]:
            print(f"{colour.dim}          shows: nothing{colour.reset}")
        for rank, parent_asin in enumerate(record["recommendations"], start=1):
            is_target = parent_asin == session["target"]
            mark = f"{colour.hit}>>{colour.reset}" if is_target else "  "
            title = truncate(products.get(parent_asin, {}).get("title", "?"), body - 24)
            price = products.get(parent_asin, {}).get("price")
            tail = f" {colour.dim}${price}{colour.reset}" if price not in (None, "") else ""
            print(f"        {mark} {rank:>2}. {parent_asin}  {title}{tail}")
        if record["dropped"]:
            print(f"{colour.dim}          ({record['dropped']} recommendation(s) dropped: unknown asin or duplicate){colour.reset}")
        if record["note"]:
            print(f"{colour.dim}          note: {record['note']}{colour.reset}")

    if session["hit"]:
        outcome = (
            f"{colour.hit}HIT{colour.reset} on turn {session['first_hit_turn']} "
            f"at rank {session['best_rank']} (RR {session['reciprocal_rank']:.3f})"
        )
    else:
        outcome = f"{colour.miss}MISS{colour.reset} after {len(session['turns'])} turns"
    print(f"\n{colour.bold}result:{colour.reset} {outcome}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay public sessions and print the conversation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=3, help="sessions to replay (0 = all)")
    parser.add_argument("--sample-id", action="append", default=[], help="replay these ids only (repeatable)")
    parser.add_argument("--scenario", choices=("buying", "browsing", "intent_override", "boundary"))
    parser.add_argument("--only", choices=("all", "hits", "misses"), default="all", help="print only these outcomes")
    parser.add_argument("--verbose", action="store_true", help="also show profile, hidden constraints, override turn")
    parser.add_argument("--hide-target", action="store_true", help="watch it blind, without the answer")
    parser.add_argument("--json", action="store_true", help="emit transcripts as JSON instead of text")
    parser.add_argument("--width", type=int, default=88)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    colour = Palette(sys.stdout.isatty() and not args.no_color and not args.json)
    samples = load_jsonl(args.dataset)
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [sample for sample in samples if sample["sample_id"] in wanted]
    if args.scenario:
        samples = [sample for sample in samples if sample["scenario_type"] == args.scenario]
    if not samples:
        parser.error("no sessions matched the filters")

    # `--only` needs every session run before it knows which to print, so the
    # limit applies to what is printed, not to what is replayed.
    budget = args.limit if args.limit > 0 else len(samples)
    if args.only == "all":
        samples = samples[:budget]

    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    replayed: list[dict] = []
    shown = 0
    for sample in samples:
        session = replay(agent, sample, catalog_ids, categories, products)
        replayed.append(session)
        if args.only == "hits" and not session["hit"]:
            continue
        if args.only == "misses" and session["hit"]:
            continue
        if shown >= budget:
            continue
        shown += 1
        if not args.json:
            render(session, products, colour, args)

    if args.json:
        print(json.dumps(replayed, indent=2))
        return

    summary = metric_summary([
        {
            "hit": session["hit"],
            "first_hit_turn": session["first_hit_turn"],
            "reciprocal_rank": session["reciprocal_rank"],
        }
        for session in replayed
    ])
    print(f"\n{colour.bold}{'=' * args.width}{colour.reset}")
    print(
        f"{len(replayed)} session(s) replayed, {shown} printed  |  "
        f"hit rate {summary['hit_rate_at_10']:.3f}  MRR {summary['mrr']:.3f}  MTTC {summary['mttc']:.2f}"
    )


if __name__ == "__main__":
    main()
