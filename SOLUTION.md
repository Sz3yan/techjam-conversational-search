# Shopping Copilot — Solution

A multi-turn conversational shopping agent for the TechJam Conversational
E-Commerce Search Challenge. It locates a hidden target product inside a frozen
50,000-item Amazon apparel catalog by accumulating constraints across a
bounded 10-turn dialogue.

**TechnicalScore 0.9373 on the 200 public sessions, up from the 0.1067 weak
baseline — 8.8×.** Runs on the Python standard library alone: no network, no
credentials, no model download, no third-party package.

The headline metrics are reproduced by the official evaluator
(`python3 -m evaluator.local_evaluator`). Ablation figures were measured during
development with a throwaway harness whose baseline configuration reproduced
`results.json` to six decimals; that harness is not part of the submission, so
those rows are reported as measured rather than as re-runnable.

## Results

| Metric | Baseline | This agent | Best possible |
|---|---|---|---|
| Hit Rate@10 | 0.125 | **0.990** | 1.000 |
| MRR | 0.068 | **0.9574** | 1.000 |
| MTTC | 9.81 | **3.245** | 1.39 |
| Efficiency | 0.119 | **0.7755** | 0.961 |
| **TechnicalScore** | **0.1067** | **0.9373** | **0.9922** |

By scenario:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 0.988 | 0.972 | 2.78 |
| Browsing | 80 | 1.000 | 0.979 | 2.99 |
| Intent Override | 30 | 1.000 | 0.933 | 4.17 |
| Boundary | 10 | 0.900 | 0.742 | 6.30 |

The 0.9922 ceiling is not 1.0 because the evaluator refuses to convert an
intent-override session before the override message arrives — turn 3 for 12 of
the 30 sessions, turn 4 for the other 18. That floors MTTC at
`(170·1 + 12·3 + 18·4) / 200 = 1.39` and caps Efficiency at 0.961.

## Architecture

Each turn runs `observe → route → retrieve → rerank → answer + clarify`.

| Module | Responsibility |
|---|---|
| `starter/state.py` | Slot tracking, constraint extraction, override detection, disclosure bookkeeping |
| `starter/ask.py` | Clarification strategy; keeps `ask_attribute` and the prose in sync |
| `starter/retrieve.py` | In-memory SQLite FTS5 index, per-term IDF, normalized text blobs |
| `starter/rerank.py` | IDF-weighted constraint coverage, category match, verbatim bonus, elimination |
| `starter/agent.py` | Orchestrator, Buying/Browsing routing, recommendation-width policy |

Tooling: `scripts/show_conversation.py` replays any session turn by turn.

**Retrieval.** BM25 over an FTS5 index with column weights favouring title and
categories. Pool size is route-dependent: 100 for Buying (a hard constraint
arrives early, so precision pays), 200 for Browsing (vague openings need a
wider net).

**Reranking.** The pool is reordered by IDF-weighted coverage of every disclosed
constraint, coverage of the stated category, a bounded bonus for verbatim
clause matches, and a large demotion for products already ruled out.

**Constraint extraction** is marker-driven with a free-text fallback.

## Why FTS5 and IDF weighting

Both choices follow from one property of the data, established below: the
customer quotes catalog metadata verbatim.

**FTS5** gives BM25 ranking, phrase queries and a C-speed inverted index from
the standard library — no dependency, no build step beyond 3.6 s of ingest, and
no model download. Since the customer's words are literally substrings of the
target's catalog row, a lexical index is not a fallback for something better;
it is the operation the task actually calls for.

**IDF weighting** is needed because disclosures are wildly unequal in value.
Measured document frequencies across the 50,000-product catalog:

| Token | Products containing it | IDF |
|---|---|---|
| `imported` | 15,300 | 2.18 |
| `polyester` | 10,884 | 2.53 |
| `cotton` | 9,775 | 2.63 |
| `leather` | 7,503 | 2.90 |
| `alloy` | 803 | 5.13 |
| `pentagram` | 16 | 8.99 |
| `kandinsky` | 1 | 11.13 |

A single card can pair `Imported` with a phrase that occurs in one product.
Only **25 of 798** public constraints contain a token appearing in one product
or fewer — roughly 1 in 32. Weighting every disclosure equally lets the 15,300-
product token drown out the decisive one, so each constraint is weighted by the
IDF of its rarest token.

## How a session actually runs

`scripts/show_conversation.py` replays any public session through the real agent
and prints the dialogue turn by turn. It imports the evaluator's own simulator
functions rather than reimplementing them, so what it prints is exactly what
gets scored.

```bash
python3 -m scripts.show_conversation --sample-id public_0001 --verbose
```

```text
public_0001  [buying / easy]
target B09PYB7B6Z  QIAN0813 Celttic Knot Triple Moon Pentagram Pentacle Star…
hidden constraints: Material:alloy; Triple Moon Pentagram Symbol

--- turn 1 ---
customer  I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
agent     Going on Material:alloy, here are the closest matches. Anything else
          that matters for this one?
          asks about: other
            1. B09LV2N6CM  YYDSBBA Sunflower Alloy Necklace for Women…

--- turn 2 ---
customer  For that, what matters is: Triple Moon Pentagram Symbol; The Triple
          Moon represents the Phases of the Moon…
agent     Going on Triple Moon Pentagram Symbol, …, here are the closest matches.
        >>  1. B09PYB7B6Z  QIAN0813 Celttic Knot Triple Moon Pentagram… $9.99
          note: target at rank 1 -- session ends

result: HIT on turn 2 at rank 1 (RR 1.000)
```

### The customer holds the answer sheet, and never asks anything

Before turn 1 the evaluator reads the hidden target product and compresses it
into an **intent card** (`intent_card()` in `evaluator/local_evaluator.py`): the
product title becomes `target_category`, and its `features` and `details`
fields — plus any regex-detected material, colour, and price — become two
`hard_constraints` and two `soft_preferences`. Every public session's card holds
exactly four constraints.

So the simulated customer is not reasoning or improvising. It is reciting
catalog metadata about one specific product, a slice at a time, and the session
ends when the agent hands that product back. Above, `Material:alloy` and
`Triple Moon Pentagram Symbol` are not things a shopper thought up — they are
`features[0]` and `features[1]` lifted verbatim off `B09PYB7B6Z`. This is why
exact lexical matching is the correct operation here, and it is the root cause
of the dense negative result further down.

### The customer releases only what it is asked for

`customer_reply()` reads the agent's `ask_attribute`, sorts every
still-undisclosed constraint into a bucket with `classify_constraint()`, and
returns up to two whose bucket matches. Four branches:

| What the agent asks | What the customer says back |
|---|---|
| `null` (no attribute) | "Those options are not quite right yet." — a wasted turn |
| An attribute with undisclosed constraints in its bucket | "For that, what matters is: …" — up to **two** constraints released |
| An attribute whose bucket is empty | "I don't have an additional preference for *X*." |
| Any attribute, first time only, in a `boundary` session | "I don't have a preference for *X*; please use your judgment." |

Independently of the ask, an `intent_override` session pre-empts the reply on
its scheduled turn with *"Actually, ignore my earlier preference…"*. That turn
is `rng.choice([3, 4])` seeded from `sample_id + scenario_type`
(`materialize_hidden_fields()`), so it is deterministic: 12 sessions get turn 3,
18 get turn 4.

The customer never asks a question and never volunteers a second constraint
unprompted. `ask_attribute` is the key, and each turn unlocks at most two
slices of a four-slice card.

### The agent chooses the key

`next_attribute()` in `starter/ask.py` picks that key. It asks the open-ended
`other` first, and only walks a priority ladder — material → style → use_case →
colour → size → budget → feature → brand → category — once `other` is spent,
skipping any attribute already asked, already covered, or refused.

That ordering matters because in `customer_reply()` the matching test is
`attribute == "other" or classify_constraint(value) == attribute`. **`other` is
a wildcard**: it matches any undisclosed constraint, and takes the first two in
card order — hard constraints before soft preferences, the most decisive
information first. A specific ask can only ever unlock its own bucket, and
misses whenever that bucket is empty.

Measured across all 647 turns of the public set:

| Ask | Times used | Turns that returned nothing |
|---|---|---|
| `other` (wildcard) | 509 | 34 (6.7%) |
| A specific attribute | 138 | 81 (58.7%) |

A specific question comes back empty more often than not; the wildcard almost
never does. A further 10 turns (all `boundary`) lose the wildcard to a refusal.

### How the ladder order was chosen

The shipped order is a hand-authored guess at how much a typical answer narrows
an apparel catalog. That guess does **not** match how constraints actually
distribute across buckets:

| Bucket | Share of the 800 public constraints | Ladder position |
|---|---|---|
| `feature` | 50.5% | 7th |
| `material` | 37.8% | 1st |
| `color` | 7.5% | 4th |
| `style` | 2.4% | 2nd |
| `size` | 1.4% | 5th |
| `use_case` | 0.5% | 3rd |
| `budget`, `brand`, `category` | 0% — never occur | 6th, 8th, 9th |

Reordering the ladder to match observed frequency is the obvious fix, and it
was tested. It loses:

| Ladder order | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| **Shipped (material-first)** | **0.990** | **0.9574** | 3.245 | **0.9373** |
| By observed bucket frequency | 0.985 | 0.9471 | 3.190 | 0.9328 |
| Frequency order, dead buckets dropped | 0.980 | 0.9453 | 3.195 | 0.9297 |
| Reversed | 0.980 | 0.9464 | 3.295 | 0.9280 |

The reason is that the ladder only fires *after* the `other` wildcard is spent,
and the wildcard drains the common buckets first. By the time the ladder runs,
the frequent buckets are usually already covered and get skipped — so ordering
by global frequency optimises for a situation that has already passed.

The spread is ~0.01, close to what n=200 can resolve, so this is evidence the
shipped order is not *worse* rather than proof it is optimal. It is kept
because nothing tested beat it.

### Why it converges in three turns

| First hit on turn | 1 | 2 | 3 | 4 | 5 | 6–10 | never |
|---|---|---|---|---|---|---|---|
| Sessions | 9 | 74 | 62 | 26 | 9 | 18 | 2 |

145 of 200 sessions land by turn 3, and 188 of the 198 hits land at rank 1.
The card holds four constraints; 256 informative replies handed over two at
once and only 36 handed over one, so two wildcard exchanges typically transfer
the entire card. At the moment of conversion the agent holds an average of
**3.20** of the target's 4 constraints.

The failure modes are visible in the same tool. `--scenario boundary` shows the
weakest case: the customer burns the wildcard on a refusal, pushing MTTC to
6.30. `--only misses` shows both public failures, and they share one cause:
every constraint on the card is catalog boilerplate. `public_0020` discloses
`cotton`, `color: grey` and a heather-blend note; `public_0180` discloses
`100% Mesh`, `Imported`, `Rubber sole`. Nothing on either card carries enough
IDF to separate one product from the thousands sharing those phrases.

## Which signals earn their place

| Configuration | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| **Full agent (shipped)** | **0.990** | **0.9574** | **3.245** | **0.9373** |
| No verbatim-phrase bonus | 0.975 | 0.9241 | 3.475 | 0.9152 |
| No ruled-out elimination | 0.955 | 0.8635 | 3.595 | 0.8846 |
| Neither (pure IDF matching) | 0.945 | 0.8328 | 3.900 | 0.8643 |

**Elimination is the larger of the two, and it is free.** Any product scored in
a previous turn without ending the session provably is not the target, so it
can be demoted. The one exception is an intent-override session before the
override lands, where the evaluator blocks conversion entirely; we demote
optimistically and re-admit pre-override items once the override turn is known
— zero risk, because those turns could never have converted anyway. Worth
**+0.053**.

**The verbatim-phrase bonus is worth +0.022**, and is deliberately bounded at
`W_PHRASE = 1.5` because it is the signal most specific to the public
simulator's literal phrasing.

## Narrow early, widen late — worth +0.066

| Policy | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| Always return 10 | 0.990 | 0.6868 | **2.480** | 0.8715 |
| Always return 3 | 0.985 | 0.8267 | 2.840 | 0.9037 |
| **Narrow → widen (shipped)** | 0.990 | **0.9574** | 3.245 | **0.9373** |

The session ends at the *first* hit. Surfacing the target at rank 9 on turn 2
banks a reciprocal rank of 0.111 and forfeits every later chance to present it
first; holding back and returning it at rank 1 two turns later is worth 1.0.
With MRR weighted 0.30 against Efficiency 0.20, that trade is strongly
profitable: widening buys 0.014 of Efficiency and pays 0.082 of MRR.

The agent returns a single best guess for turns 1–5, a shortlist of 3 for turns
6–7, and the full ten from turn 8 so a near-miss still lands in the scored
window. This is a deliberate precision-over-recall stance, and it matches how a
chat interface actually behaves: one confident recommendation and a question,
not a ten-item dump.

## Intent override: keep the context, drop only the contradicted slot

The problem statement suggests "slot erasure and rewriting" on intent override.
The override replaces a stated preference but does not change the target
product, so earlier disclosures remain true of it.

| Override handling | Overall score | intent_override Hit@10 | intent_override MRR |
|---|---|---|---|
| **Keep full context (shipped)** | **0.9373** | 1.000 | **0.933** |
| Drop constraints disclosed on turn 1 | 0.9361 | 1.000 | 0.909 |
| Wipe all state on override | 0.9350 | 1.000 | 0.916 |

Keeping context wins, but by **0.0023** — a small margin, and hit rate is
unaffected in all three. An earlier iteration of this agent showed a much
larger gap; on the current tree, elimination and IDF weighting recover enough
signal that override handling is no longer decisive. Reported as measured.

## Robustness under paraphrase

The public simulator emits fixed phrasings and the specification reserves the
right to paraphrase the private 800. We tested that by rewriting every
customer utterance so that **all** marker strings the agent keys on are broken
— `CATEGORY_RE`, all three `CONSTRAINT_MARKERS`, `NO_ADDITIONAL_RE`,
`NO_PREFERENCE_RE`, `OVERRIDE_RE`, `STALLED_RE` and every `BOILERPLATE` phrase
in `starter/state.py` — while preserving the information content. A second
stressor cripples extraction by discarding every second disclosure.

| Configuration | clean | paraphrased | starved | both |
|---|---|---|---|---|
| Full agent | **0.9373** | 0.7581 | 0.9197 | 0.6986 |
| No verbatim-phrase bonus | 0.9152 | 0.7581 | 0.8978 | 0.6986 |
| No ruled-out elimination | 0.8846 | 0.7427 | 0.8489 | 0.6036 |
| Neither (pure IDF matching) | 0.8643 | 0.7427 | 0.8291 | 0.6036 |

**This is the weakest part of the system and we are reporting it plainly.**
Under a total marker break the score falls from 0.9373 to 0.7581 — a 19%
degradation, not a marginal one. Starved extraction alone is survivable
(−0.018); paraphrase is not.

Isolating the cause:

| Paraphrase scope | Score |
|---|---|
| None | 0.9373 |
| Only the "no preference" markers | 0.9310 |
| Everything **except** the "no preference" markers | 0.7634 |
| Everything | 0.7581 |

The damage is not in exhaustion tracking; it is in constraint extraction. When
the `For that, what matters is: A; B.` marker survives, `_extract()` splits on
`;` and yields two clean, separately IDF-weighted constraints. When it does
not, the free-text fallback treats the whole utterance as one blob — merging
two constraints and dragging in filler words, which dilutes exactly the IDF
weighting the reranker depends on.

Note also that under paraphrase the verbatim-phrase bonus becomes inert (the
first two rows converge to the same score), which is the expected behaviour for
a signal keyed to literal phrasing.

**The fix is known and not yet implemented:** the fallback should segment an
utterance into clauses on conjunctions and punctuation, rather than emitting
one blob. That is the highest-value robustness work remaining.

## Negative result: the dense route does not help

We built a MiniLM (`all-MiniLM-L6-v2`) dense retrieval route — 50,000 cached
384-dimensional embeddings, cosine similarity, blended into the reranker — and
it never beat sparse-only:

| Dense weight | 0.0 | 1.0 | 2.0 | 3.0 | 5.0 | 8.0 |
|---|---|---|---|---|---|---|
| Score | **0.9327** | 0.9313 | 0.9308 | 0.9285 | 0.9296 | 0.9266 |

*(Swept against the then-current 0.9327 baseline, before constraint
discriminance was added. Relative ordering is the result; the absolute
baseline has since moved to 0.9373.)*

Gating it to fire only when few constraints were known, and testing boundary
sessions in isolation (the regime where semantic matching should help most),
also failed to beat zero.

**Why:** the disclosed constraints are literal catalog metadata — `"100%
Leather"`, `"Buckle closure"`, `"Material:alloy"`. Exact lexical matching is
the correct operation for attribute satisfaction. Embeddings blur exactly the
fine distinctions that separate the target from its near neighbours, and the
BM25 pool is already topically relevant, so semantic similarity adds noise
rather than signal.

The route has since been **removed from the tree**. The shipped agent has no
optional imports and no numpy, torch or sentence-transformers dependency — the
sparse path is the only path. Removal was verified score-neutral: 0.937314
before and after, with all 200 per-session records identical.

## Setup and reproduction

Python 3.10+. No dependencies.

```bash
# 1. Place the catalog (50,000 rows) at data/catalog.jsonl
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl

# 2. Score against the official evaluator
python3 -m evaluator.local_evaluator          # -> results.json, ~20 s

# 3. Tests
python3 -m unittest discover tests -v

# 4. Watch sessions play out turn by turn
python3 -m scripts.show_conversation --limit 3 --verbose
```

Steps 1-4 reproduce every headline and per-scenario number in this document.
The ablation and robustness tables are development measurements and are not
re-derivable from the shipped tree; each states the configuration it varied so
the experiment can be reconstructed.

## Feasibility disclosure

| | |
|---|---|
| Model | none — no LLM in the scoring path |
| API cost | $0.00 |
| Token usage | 0 prompt, 0 completion |
| Network | not required at any point |
| Dependencies | Python standard library only |
| Index build | 3.6 s, one-off |
| Latency | 14.0 ms p50, 34.6 ms p95 per turn (647 turns, 4 passes) |
| Peak memory | 0.40 GB |

Measured on the development machine (Apple silicon, Python 3.14). Latency is
dominated by the FTS5 query and the rerank scan over the candidate pool, so the
Browsing route (pool 200) sits at the upper end.

## Limitations and what we would improve

**Paraphrase robustness is the real weakness.** A 19% drop under full marker
break is the largest single risk to private-set performance, and the cause is
localised to one function (`_extract()`'s fallback branch). Clause segmentation
is the fix.

**Boundary is the weakest scenario** (Hit 0.900, MRR 0.742, n=10). When the
customer declines to specify an attribute, the agent has no fallback beyond
asking about a different one. With only 10 public sessions the measurement is
too noisy to tune against — its entire remaining shortfall is 0.0117 of final
score.

**Question selection is heuristic, not decision-theoretic.** The agent asks an
open question first and then walks a fixed priority order that, as shown above,
is not justified by bucket frequency — merely not beaten by it. A genuine
question-value estimator — choosing the attribute whose answer most reduces
expected candidate entropy — is the natural next step. 125 of 647 turns
currently return no new information; eliminating them is worth about +0.0125.

**No semantic generalisation.** Because the dense route lost, the system cannot
bridge vocabulary the catalog does not literally contain. Against a private set
whose customers speak more naturally than the public simulator, that is the
most likely failure mode, and the paraphrase result above is the evidence for
it rather than against it.

**Everything is tuned on n=200.** Standard error on hit rate at this sample
size is roughly 2.5 points — larger than the entire remaining gap to a perfect
score. Treat any difference below ~0.005 as unresolved by this data. Of the
0.0627 still on the table, 8% sits in Hit Rate, 20% in MRR and 72% in
Efficiency — and 0.0078 of that Efficiency share is locked by the
override-conversion gate, so roughly 0.037 is genuinely addressable.

**Weight tuning is exhausted.** A full sweep of both pool sizes and all rerank
weights moved nothing outside ±0.005. Further gains require a new signal.
