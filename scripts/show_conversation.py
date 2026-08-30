"""Replay a session turn by turn, with the belief state exposed.

This drives the real agent through the evaluator's own simulator functions, so
what it prints is exactly what gets scored -- no reimplementation of the
protocol, no chance of the demo and the score disagreeing.

Alongside the dialogue it prints what the agent was thinking: how much of the
catalogue is still in play, how confident the leader is, which question won on
expected information gain and by how much, and why the recommendation list is
the width it is. That is the part worth watching -- the visible behaviour
(stay quiet, ask, then commit) is an output of those numbers, not a script.

    python3 -m scripts.show_conversation --sample-id public_0001
    python3 -m scripts.show_conversation --scenario boundary --limit 2
    python3 -m scripts.show_conversation --scenario intent_override --natural
"""

from __future__ import annotations

import argparse
import math
import uuid

import evaluator.local_evaluator as ev
from socratic.config import AgentConfig
from starter.agent import Agent

RULE = "-" * 76


def wrap(text: str, indent: int = 10, width: int = 76) -> str:
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width - indent:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    pad = " " * indent
    return f"\n{pad}".join(lines) if lines else ""


def replay(agent: Agent, sample: dict, catalog_ids, categories, products, verbose: bool) -> dict:
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = ev.materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = ev.initial_message(effective, ev.coarse_category(categories.get(target, [])), disclosed)

    title = str(products[target].get("title") or "")[:70]
    print(f"\n{sample['sample_id']}  [{sample['scenario_type']} / {sample.get('difficulty_bucket','?')}]")
    print(f"target {target}  {title}")
    print("hidden card: " + "; ".join(card["hard_constraints"] + card["soft_preferences"])[:150])

    hit_turn = best_rank = None
    for turn in range(1, ev.MAX_TURNS + 1):
        response = agent.respond(session_id, message, turn, ev.TOP_K)
        session = agent.sessions[session_id]
        head, probabilities, entropy = session.belief.posterior(top=8)

        print(f"\n--- turn {turn} ---")
        print(f"customer  {wrap(message)}")
        print(f"agent     {wrap(response.get('message', ''))}")
        print(f"          asks about: {response.get('ask_attribute')}")
        if verbose:
            effective_size = 2.0**entropy
            leader = probabilities[0] if probabilities else 0.0
            # 2**entropy is the perplexity of the posterior -- the size of the
            # uniform distribution carrying the same uncertainty. It is not a
            # candidate count; it is how many products the agent is *as unsure
            # between* as it currently is.
            print(
                f"          belief: {entropy:.2f} bits of uncertainty "
                f"(~{effective_size:,.0f} products' worth), leader p={leader:.3f}"
            )
            if session.question_scores:
                ranked_questions = ", ".join(
                    f"{name} {gain:+.2f}" for name, gain in session.question_scores
                )
                print(f"          question value (bits): {ranked_questions}")
            print(f"          shows {len(response.get('recommendations') or [])} of a possible 10")
        ranked = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
        for position, asin in enumerate(ranked[:5], start=1):
            marker = ">>" if asin == target else "  "
            product_title = str(products[asin].get("title") or "")[:58]
            print(f"        {marker} {position}. {asin}  {product_title}")

        if override_applied and target in ranked:
            best_rank, hit_turn = ranked.index(target) + 1, turn
            print(f"          -> target at rank {best_rank}; session ends")
            break
        if turn == ev.MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            message, boundary_used = ev.customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    outcome = (
        f"HIT on turn {hit_turn} at rank {best_rank} (RR {1.0 / best_rank:.3f})"
        if hit_turn else "MISS"
    )
    print(f"\nresult: {outcome}\n{RULE}")
    return {"hit": hit_turn is not None, "turn": hit_turn, "rank": best_rank}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay sessions through the real agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--quiet", action="store_true", help="hide belief-state annotations")
    parser.add_argument(
        "--natural", action="store_true",
        help="let the local model write the customer-facing prose",
    )
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
    if args.scenario:
        samples = [s for s in samples if s["scenario_type"] == args.scenario]
    samples = samples[: args.limit]

    config = AgentConfig()
    if args.natural:
        config = config.replace(llm_uses=frozenset(config.llm_uses | {"message"}))
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agent = Agent(args.catalog, config=config)
    for sample in samples:
        replay(agent, sample, catalog_ids, categories, products, verbose=not args.quiet)


if __name__ == "__main__":
    main()
