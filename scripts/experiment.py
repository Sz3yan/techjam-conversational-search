"""Re-runnable experiment harness.

Every table in SOLUTION.md is produced by this file. It drives the official
evaluator -- importing its scoring loop rather than reimplementing it -- with
the agent under a named configuration and an optional customer-side stressor,
so measured numbers and scored numbers cannot drift apart.

    python3 -m scripts.experiment --suite ablations
    python3 -m scripts.experiment --suite robustness
    python3 -m scripts.experiment --configs shipped --stressors clean hostile
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import evaluator.local_evaluator as ev
from scripts.stressors import Stressor
from socratic.config import AgentConfig
from starter.agent import Agent

BASE = AgentConfig()
# The ablation table isolates the deterministic core, so every variant in it
# has the model switched off. The model's own contribution gets its own suite.
DET = BASE.replace(llm_mode="off")

CONFIGS: dict[str, AgentConfig] = {
    "shipped": BASE,
    "deterministic": DET,
    # -- what the local model adds -----------------------------------------
    "llm_segment_only": BASE.replace(llm_uses=frozenset({"segment"})),
    "llm_rerank_only": BASE.replace(llm_uses=frozenset({"rerank"})),
    "llm_dense": BASE.replace(channels=frozenset({"clause", "token", "category", "price", "dense"})),
    # -- question policy ----------------------------------------------------
    "ask_ladder": DET.replace(question_policy="ladder"),
    "ask_none": DET.replace(question_policy="none"),
    # -- recommendation width -----------------------------------------------
    "width_always_10": DET.replace(width_policy="fixed", fixed_width=10),
    "width_always_3": DET.replace(width_policy="fixed", fixed_width=3),
    "width_always_1": DET.replace(width_policy="fixed", fixed_width=1),
    "width_schedule": DET.replace(width_policy="schedule"),
    # -- intent override -----------------------------------------------------
    "override_retract": DET.replace(override_mode="retract"),
    "override_erase": DET.replace(override_mode="erase"),
    # -- evidence ------------------------------------------------------------
    "no_elimination": DET.replace(eliminate_shown=False),
    "no_clause_channel": DET.replace(channels=frozenset({"token", "category", "price"})),
    "no_token_channel": DET.replace(channels=frozenset({"clause", "category", "price"})),
    "no_category_channel": DET.replace(channels=frozenset({"clause", "token", "price"})),
    "no_price_channel": DET.replace(channels=frozenset({"clause", "token", "category"})),
    "clause_only": DET.replace(channels=frozenset({"clause"})),
    # -- lookahead -------------------------------------------------------------
    "depth_1": DET.replace(depth=1),
    "depth_3": DET.replace(depth=3),
}

SUITES = {
    "ablations": (
        ["deterministic", "ask_ladder", "ask_none", "width_always_10", "width_always_3",
         "width_always_1", "width_schedule", "override_retract", "override_erase",
         "no_elimination", "no_clause_channel", "no_token_channel",
         "no_category_channel", "no_price_channel", "clause_only", "depth_1", "depth_3"],
        ["clean"],
    ),
    "robustness": (
        ["deterministic", "shipped", "llm_segment_only", "llm_dense", "ask_ladder"],
        ["clean", "paraphrase", "hostile", "starved"],
    ),
    "quick": (["shipped"], ["clean"]),
    "llm": (
        ["deterministic", "llm_segment_only", "llm_rerank_only", "llm_dense", "shipped"],
        ["clean"],
    ),
}


def run(config: AgentConfig, samples, catalog_ids, categories, products, catalog_path, stressor):
    agent = Agent(catalog_path, config=config)
    started = time.perf_counter()
    with Stressor(stressor):
        result = ev.evaluate(agent, samples, catalog_ids, categories, products)
    elapsed = time.perf_counter() - started
    turns = sum(
        (s["first_hit_turn"] or ev.MAX_TURNS) for s in result["sessions"]
    )
    result["wall_seconds"] = round(elapsed, 2)
    result["ms_per_turn"] = round(1000.0 * elapsed / max(1, turns), 2)
    return result


def table(rows: list[dict]) -> str:
    header = "| Configuration | Stressor | Hit@10 | MRR | MTTC | Score | ms/turn |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['stressor']} | {row['hit_rate_at_10']:.3f} | "
            f"{row['mrr']:.4f} | {row['mttc']:.3f} | **{row['score']:.4f}** | {row['ms_per_turn']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Socratic experiment harness")
    parser.add_argument("--suite", choices=sorted(SUITES), default=None)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--stressors", nargs="*", default=None)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sweep", nargs=2, metavar=("FIELD", "VALUES"), default=None,
        help="sweep one config field, e.g. --sweep alpha 3,6,9 (from the deterministic base)",
    )
    parser.add_argument("--output", default="results/experiments.json")
    args = parser.parse_args()

    names, stressors = SUITES[args.suite] if args.suite else (["shipped"], ["clean"])
    if args.configs:
        names = args.configs
    if args.sweep:
        field, raw = args.sweep
        names = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            current = getattr(DET, field)
            value = type(current)(token) if not isinstance(current, bool) else token == "true"
            label = f"{field}={token}"
            CONFIGS[label] = DET.replace(**{field: value})
            names.append(label)
    if args.stressors:
        stressors = args.stressors

    samples = ev.load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)

    rows = []
    detail = {}
    for name in names:
        config = CONFIGS[name]
        for stressor in stressors:
            result = run(config, samples, catalog_ids, categories, products, args.catalog, stressor)
            row = {
                "config": name,
                "stressor": stressor,
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": result["mrr"],
                "mttc": result["mttc"],
                "score": result["recommended_technical_score"],
                "ms_per_turn": result["ms_per_turn"],
                "tokens": result["reported_token_usage"]["total_tokens"],
            }
            rows.append(row)
            detail[f"{name}::{stressor}"] = {
                k: v for k, v in result.items() if k != "sessions"
            }
            print(
                f"{name:22} {stressor:11} score={row['score']:.4f} "
                f"hit={row['hit_rate_at_10']:.3f} mrr={row['mrr']:.4f} "
                f"mttc={row['mttc']:.3f} ({result['wall_seconds']}s)",
                flush=True,
            )

    print("\n" + table(rows))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "detail": detail}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
