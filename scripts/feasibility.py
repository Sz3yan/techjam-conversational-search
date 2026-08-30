"""Measure what the agent costs to run.

Produces the numbers in SOLUTION.md's feasibility disclosure: index build time,
per-turn latency percentiles, peak memory and real token usage, for both the
offline configuration and the one with the local model enabled.

    python3 -m scripts.feasibility
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import evaluator.local_evaluator as ev
from socratic.config import AgentConfig
from socratic.index import CatalogIndex
from starter.agent import Agent


def peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    divisor = 2**30 if sys.platform == "darwin" else 2**20
    return usage / divisor


class Timed:
    """Wraps an agent to record per-turn wall time."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.latencies: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.latencies.append((time.perf_counter() - started) * 1000.0)
        return response


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/feasibility.json")
    args = parser.parse_args()

    report: dict = {}

    started = time.perf_counter()
    CatalogIndex.build(args.catalog)
    report["index_build_seconds"] = round(time.perf_counter() - started, 2)

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)

    for label, config in (
        ("offline (no model)", AgentConfig(llm_mode="off")),
        ("local model enabled", AgentConfig()),
    ):
        agent = Agent(args.catalog, config=config)
        timed = Timed(agent)
        started = time.perf_counter()
        result = ev.evaluate(timed, samples, catalog_ids, categories, products)
        wall = time.perf_counter() - started
        usage = result["reported_token_usage"]
        report[label] = {
            "technical_score": result["recommended_technical_score"],
            "turns": len(timed.latencies),
            "latency_ms_p50": round(percentile(timed.latencies, 0.50), 1),
            "latency_ms_p95": round(percentile(timed.latencies, 0.95), 1),
            "latency_ms_mean": round(statistics.fmean(timed.latencies), 1),
            "wall_seconds_200_sessions": round(wall, 1),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "model_calls": getattr(agent.llm, "calls", 0) if agent.llm else 0,
            "peak_rss_gb": round(peak_rss_gb(), 2),
        }
        print(f"{label}: {json.dumps(report[label], indent=2)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
