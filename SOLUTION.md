# Shopping Copilot — Solution

A multi-turn conversational shopping agent for the TechJam Conversational
E-Commerce Search Challenge. It locates a hidden target product inside a frozen
50,000-item Amazon apparel catalog by accumulating constraints across a
bounded 10-turn dialogue.

**TechnicalScore 0.9373 on the 200 public sessions, up from the 0.1067 weak
baseline — 8.8×.** Runs on the Python standard library alone: no network, no
credentials, no model download, 7.6 ms median per turn.

## Results

| Metric | Baseline | This agent | Best possible |
|---|---|---|---|
| Hit Rate@10 | 0.125 | **0.990** | 1.000 |
| MRR | 0.068 | **0.957** | 1.000 |
| MTTC | 9.81 | **3.25** | 1.39 |
| Efficiency | 0.119 | **0.775** | 0.961 |
| **TechnicalScore** | **0.1067** | **0.9373** | **0.9922** |

By scenario:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 0.988 | 0.972 | 2.77 |
| Browsing | 80 | 1.000 | 0.979 | 2.99 |
| Intent Override | 30 | 1.000 | 0.933 | 4.17 |
| Boundary | 10 | 0.900 | 0.742 | 6.30 |

The 0.9922 ceiling is not 1.0 because the evaluator refuses to convert an
intent-override session before the override message arrives (turn 3 for 12 of
the 30 sessions, turn 4 for the other 18). That floors MTTC at 1.39 and caps
Efficiency at 0.961.

## Architecture

Each turn runs `observe → route → retrieve → rerank → answer + clarify`.

| Module | Responsibility |
|---|---|
| `starter/state.py` | Slot tracking, constraint extraction, override detection, disclosure bookkeeping |
| `starter/ask.py` | Clarification strategy; keeps `ask_attribute` and the prose in sync |
| `starter/retrieve.py` | In-memory SQLite FTS5 index, per-term IDF, normalized text blobs |
| `starter/rerank.py` | IDF-weighted constraint coverage, category match, verbatim bonus, elimination |
| `starter/dense.py` | Optional embedding route — built, measured, disabled (see below) |
| `starter/agent.py` | Orchestrator, Buying/Browsing routing, recommendation-width policy |

**Retrieval.** BM25 over an FTS5 index with column weights favouring title and
categories. Pool size is route-dependent: 100 for Buying (a hard constraint
arrives early, so precision pays), 200 for Browsing (vague openings need a
wider net).

**Reranking.** The pool is reordered by IDF-weighted coverage of every disclosed
constraint, coverage of the stated category, a bounded bonus for verbatim
clause matches, and a large demotion for products already ruled out.

**Constraint extraction** is marker-driven with a free-text fallback. The
fallback is what makes the system survive rephrasing — see Robustness.

## Three findings that drive the score

**1. Elimination is free before an override.** Any product scored in a previous
turn without ending the session provably is not the target, so it can be
demoted. The one exception is an intent-override session before the override
lands, where the evaluator blocks conversion entirely. We demote optimistically
and re-admit pre-override items once the override turn is known — zero risk,
because those turns could never have converted anyway. Worth +0.05.

**2. Naive slot erasure is actively harmful.** The problem statement suggests
"slot erasure and rewriting" on intent override. Implemented literally, that
costs 37 points on exactly those sessions:

| Override handling | intent_override Hit@10 |
|---|---|
| Wipe all state | 0.500 |
| Keep category, drop stale preference | 0.700 |
| **Keep full context** | **0.867** |

The override replaces a stated preference but does not change the target
product, so earlier disclosures remain true of it. Erase the contradicted slot,
never the context.

**3. Constraint discriminance, not constraint count.** Disclosures are wildly
unequal: "leather" and "Imported" each appear verbatim in over 10,000 of the
50,000 products, while a distinctive feature sentence appears in one. Only 1 in
24 constraints is unique. Weighting every disclosure equally lets the generic
ones drown out the decisive one, so each constraint is weighted by the IDF of
its rarest token. Worth +0.005, and free — unlike counting verbatim clause
occurrences, which costs a full catalog scan and scored *worse* (0.9363).

**4. Narrow early, widen late — worth +0.062.** The session ends at the *first*
hit. Surfacing the target at rank 9 on turn 2 banks a reciprocal rank of 0.111
and forfeits every later chance to present it first; holding back and returning
it at rank 1 two turns later is worth 1.0. With MRR weighted 0.30 against
Efficiency 0.20, that trade is strongly profitable. The agent returns a single
best guess for turns 1–5, a shortlist of 3 for turns 6–7, and the full ten from
turn 8 so a near-miss still lands in the scored window.

| Policy | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| Always return 10 | 0.990 | 0.684 | 2.48 | 0.8715 |
| **Narrow → widen** | 0.990 | **0.957** | 3.25 | **0.9373** |

This is a deliberate precision-over-recall stance, and it matches how a chat
interface actually behaves: a conversational assistant offers one confident
recommendation and a question, not a ten-item dump.

## Robustness

The public simulator emits fixed phrasings, and the specification reserves the
right to paraphrase the private 800. We tested that directly by rewriting every
customer utterance to break all marker strings while preserving its information:

| Configuration | Public | Paraphrased |
|---|---|---|
| Full agent | 0.9327 | 0.9206 |
| No verbatim-phrase bonus | 0.9150 | 0.9072 |
| No ruled-out elimination | 0.8847 | 0.8859 |
| Neither (pure IDF matching) | 0.8642 | 0.8555 |

**Degradation under full paraphrase: −1.8%.** Stripping both simulator-specific
mechanisms still leaves 0.8643. The result does not depend on the public set's
literal phrasing.

### Stress testing the recommendation-width policy

Narrowing is the single largest design bet (+0.062), and it assumes rank-1
precision stays high. If the private sessions are harder, narrowing could mean
never surfacing the target at all. We tested that directly under two
independent stressors — paraphrased customers, and deliberately crippled
extraction that discards every second disclosure — plus both at once:

| Policy | clean | paraphrased | starved | both |
|---|---|---|---|---|
| Always return 10 | 0.8715 | 0.8724 | 0.8613 | 0.8347 |
| Adaptive (widen when information dries up) | 0.9288 | 0.9172 | 0.9070 | 0.8787 |
| **Fixed narrow → widen (shipped)** | **0.9373** | **0.9204** | **0.9095** | **0.8900** |

The shipped policy wins in every condition, including the compound one, and an
information-driven adaptive variant did not beat it. The bet is validated
rather than assumed.

## Negative result: the dense route does not help

We built a MiniLM (`all-MiniLM-L6-v2`) dense retrieval route — 50,000 cached
384-dimensional embeddings, cosine similarity, blended into the reranker — and
it never beat sparse-only:

| Dense weight | 0.0 | 1.0 | 2.0 | 3.0 | 5.0 | 8.0 |
|---|---|---|---|---|---|---|
| Score | **0.9327** | 0.9313 | 0.9308 | 0.9285 | 0.9296 | 0.9266 |

Gating it to fire only when few constraints were known, and testing boundary
sessions in isolation (the regime where semantic matching should help most),
also failed to beat zero.

**Why:** the disclosed constraints are literal catalog metadata — `"100%
Leather"`, `"Buckle closure"`, `"Material:alloy"`. Exact lexical matching is
the correct operation for attribute satisfaction. Embeddings blur exactly the
fine distinctions that separate the target from its near neighbours, and the
BM25 pool is already topically relevant, so semantic similarity adds noise
rather than signal.

The module ships behind `W_DENSE = 0.0`, disabled by measurement rather than
omission. `starter/dense.py` guards every optional import, so the agent runs
identically with or without numpy and torch installed — verified: both paths
score 0.9327.

## Setup and reproduction

Python 3.10+. No dependencies required.

```bash
# 1. Place the catalog (50,000 rows) at data/catalog.jsonl
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl

# 2. Score against the official evaluator
python3 -m evaluator.local_evaluator     # -> results.json, ~15 s

# 3. Tests
python3 -m unittest tests.test_evaluator -v
```

Optional dense route (not required, and does not improve the score):

```bash
python3 -m venv .venv && .venv/bin/pip install sentence-transformers
PYTHONPATH=. .venv/bin/python scripts/build_embeddings.py   # ~30 s, caches 77 MB
```

## Feasibility disclosure

| | |
|---|---|
| Model | none — no LLM in the scoring path |
| API cost | $0.00 |
| Token usage | 0 prompt, 0 completion |
| Network | not required at any point |
| Index build | 3.6 s, one-off |
| Latency | 7.6 ms p50, 9.4 ms p95 per turn |
| Peak memory | 0.43 GB |

Every claim above is reproducible with the commands in the previous section.

## Limitations and what we would improve

**Boundary is the weakest scenario** (Hit 0.900, MRR 0.742, n=10). When the
customer declines to specify an attribute, the agent has no fallback beyond
asking about a different one. A prior over which attributes actually
discriminate within the current candidate pool would help. With only 10 public
sessions the measurement is too noisy to tune against.

**Question selection is heuristic, not decision-theoretic.** The agent asks an
open question first and then walks a fixed priority order. A genuine
question-value estimator — choosing the attribute whose answer most reduces
expected candidate entropy — is the natural next step and the one the problem
statement's "question-value estimation" direction points at.

**No semantic generalisation.** Because the dense route lost, the system cannot
bridge vocabulary the catalog does not literally contain. Against a private set
whose customers speak more naturally than the public simulator, that is the
most likely failure mode. The paraphrase test is a proxy for this risk, not a
proof against it.

**The width policy is tuned on n=200.** Standard error on hit rate at this
sample size is roughly 2.5 points, and the schedule sits on a plateau
(0.931–0.933 across neighbouring settings) rather than a sharp optimum, so it
should transfer — but the exact handover turns are not meaningfully resolved by
this much data.

**Weight tuning is exhausted.** A full sweep of both pool sizes and all rerank
weights moved nothing outside ±0.005. Further gains require a new signal, not
reweighting.

## Team contributions

_To be completed._
