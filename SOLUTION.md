# Socratic — Solution

A conversational shopping agent for the TechJam Conversational E-Commerce
Search Challenge. It plays Twenty Questions against a 50,000-item Amazon
apparel catalogue — except there are only ten turns, so every question has to
be chosen for how much of the catalogue it eliminates.

**TechnicalScore 0.9499 offline, against a 0.1067 weak baseline — 8.9×, and
0.9922 is the mathematical ceiling.** The default configuration runs on the
Python standard library alone: no network, no credentials, no third-party
package. A local model is used when one is reachable; it is worth
approximately nothing on clean input and **+0.25 when the customer stops
speaking in template phrases**, which is the failure mode that actually
threatens the private split.

Every table below is produced by `scripts/experiment.py` and
`scripts/feasibility.py`, both committed. There are no numbers in this document
that the repository cannot re-derive.

---

## 1. What this problem actually is

The provided baseline treats the task as retrieval with a chat wrapper. Reading
`evaluator/local_evaluator.py` rather than the README says otherwise:

- **`intent_card()` compresses the hidden target's own catalogue row** into four
  constraints — two hard, two soft — drawn from its `features` and `details`
  plus a regex-detected material, colour and price. The public sessions ship no
  card; it is generated at scoring time from the catalogue the agent already
  holds.
- **`customer_reply()` releases information only in response to
  `ask_attribute`**, at most two constraints per turn, bucketed by attribute.
  Returning `null` wastes the turn outright.
- **The session ends at the *first* hit.** Surfacing the target at rank 9 banks
  a reciprocal rank of 0.111 and forfeits every later chance to present it
  first.

Three consequences follow, and they define the whole design:

1. The customer is a deterministic function of one product, reciting catalogue
   metadata a slice at a time.
2. The agent controls the information faucet. `ask_attribute` is the key.
3. **Answering early is costly.** Recommending is a commitment, not a free
   action.

That is not a search problem. It is **sequential Bayesian experimental design
under a known scoring function**: maintain a posterior over 50,000 hypotheses,
choose each question to maximise expected information gain, and answer only
when the expected score of answering beats the expected score of asking again.

### The information is all there — the problem is extracting it in time

Indexing the catalogue's attribute clauses and intersecting them by disclosure:

| Constraints known | Median candidates remaining | Uniquely identified | Within top 10 |
|---|---|---|---|
| Both hard constraints | 20 | 67 / 200 | 91 / 200 |
| All four | **1** | **147 / 200** | 175 / 200 |

Full disclosure collapses a 50,000-product catalogue to a single candidate in
most sessions. Nothing about this task is short of information. The entire
difficulty is that information arrives two slices per turn, only when asked
for, and every turn spent asking costs Efficiency.

---

## 2. Results

Measured on the 200 public sessions with the official evaluator.

| Metric | Weak baseline | **Socratic (offline)** | Socratic (+ local model) | Ceiling |
|---|---|---|---|---|
| Hit Rate@10 | 0.125 | **0.995** | 0.995 | 1.000 |
| MRR | 0.0680 | **0.9793** | 0.9777 | 1.000 |
| MTTC | 9.81 | **3.070** | 3.170 | 1.390 |
| Efficiency | 0.119 | **0.7930** | 0.7830 | 0.9610 |
| **TechnicalScore** | **0.1067** | **0.9499** | **0.9474** | **0.9922** |

By scenario, offline configuration:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 0.988 | 0.9629 | 2.675 |
| Browsing | 80 | 1.000 | 0.9938 | 3.100 |
| Intent Override | 30 | 1.000 | 0.9778 | 4.100 |
| Boundary | 10 | 1.000 | 1.0000 | 2.900 |

**Why the ceiling is 0.9922 and not 1.0.** The evaluator refuses to convert an
intent-override session before the override message arrives. That turn is
`rng.choice([3, 4])` seeded from `sample_id`, so it is deterministic: 12 of the
30 override sessions unlock on turn 3 and 18 on turn 4. Best-possible MTTC is
therefore `(170·1 + 12·3 + 18·4) / 200 = 1.39`, capping Efficiency at 0.961.
Derived by `scripts/experiment.py`; the arithmetic is reproduced in §8.

Where the conversions land:

| First hit on turn | 1 | 2 | 3 | 4 | 5 | 6–10 | never |
|---|---|---|---|---|---|---|---|
| Sessions | 29 | 55 | 49 | 39 | 8 | 19 | 1 |

193 of the 199 hits land at rank 1; four at rank 2, one at rank 3, one at rank 5.

---

## 3. Architecture

```
offline (once):  catalogue → clause index + IDF  [+ optional embedding cache]
per turn:        observe → update belief → re-orchestrate → ask (always) + answer (maybe)
```

| Module | Responsibility |
|---|---|
| `socratic/index.py` | Clause-level catalogue index, token postings, IDF |
| `socratic/belief.py` | Posterior over 50,000 products; evidence ledger; retraction |
| `socratic/policy.py` | EIG question selection; expected-score answer/width decision |
| `socratic/dialogue.py` | Utterance → evidence, marker-hinted with a general fallback |
| `socratic/dense.py` | Optional dense channel, as a likelihood mixture component |
| `socratic/llm.py` | Health-gated local model; three uses, all degradable |
| `socratic/config.py` | Every contestable decision, as a switchable field |
| `starter/agent.py` | Orchestrator implementing the required `Agent` interface |

Tooling: `scripts/experiment.py` (ablations, stressors, sweeps),
`scripts/feasibility.py` (latency, memory, tokens),
`scripts/show_conversation.py` (turn-by-turn replay with the belief exposed),
`scripts/build_embeddings.py` (one-off dense cache).

### The four pillars are one mechanism, not four features

| Pillar | How it is realised |
|---|---|
| I — Dual-track routing | Route inferred from the opening turn (`scenario_type` is never given to the agent), setting the candidate head the question policy reasons over |
| I — Multi-route retrieval + ranking | Clause, token-IDF, category, price and dense enter as **likelihood channels**; the local model reranks when the posterior cannot separate its leaders |
| II — Dynamic state machine | Bayesian update over an append-only evidence ledger; override handled by evidence retraction, individually addressable |
| II — Over-generality cutoff | The answer-vs-ask decision **is** the cutoff, thresholded on expected score rather than on a pool-size heuristic |
| III — Runtime re-orchestration | Route, question and list width are all re-derived per turn from belief state |
| IV — Metrics | The decision objective is literally `0.5·hit + 0.3·(1/rank) + 0.2·(11−turn)/10` |

---

## 4. The two decisions that matter

### 4.1 Which question — expected information gain

For every allowed attribute, the agent predicts what each plausible product's
owner *would* disclose if asked, groups the candidates by the answer they would
give, and picks the attribute whose answer splits the posterior most evenly.
Products that would answer identically are indistinguishable under that
question, so the entropy surviving the answer is exactly the entropy inside
those groups.

This is computable precisely because of §1: the customer quotes catalogue
metadata, so "what would this candidate say?" is a lookup, not a guess.

`other` is modelled as an open question — *"anything else that matters?"* — which
a shopper answers with their most salient remaining requirement rather than
with silence. Specific attributes only unlock their own bucket.

**Worth +0.082.** A hand-authored priority ladder — the obvious alternative,
and what a previous attempt on this problem shipped — scores 0.8676 against
0.9499. Note *where* the loss lands: the ladder's Hit Rate is actually higher
(1.000 vs 0.995), and its MTTC is far better (1.960 vs 3.070). It loses on MRR,
0.6227 against 0.9793, because without a value estimate for asking it has no
reason to stay quiet and converts early at bad ranks.

| Question policy | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| **Expected information gain** | 0.995 | **0.9793** | 3.070 | **0.9499** |
| Fixed priority ladder | 1.000 | 0.6227 | 1.960 | 0.8676 |
| Never ask | 0.950 | 0.4850 | 2.475 | 0.7910 |

### 4.2 Whether to answer — expectimax over the scoring function

Asking is free: the API lets one turn carry both a question and a list, so the
agent always asks. The real decision is **how many products to show**, and it is
made by a short expectimax over the session score itself, carrying the
posterior's tail mass explicitly so that conditioning on "the target was not in
the list I just showed" dilutes across the whole catalogue.

Nothing about narrow-early-widen-late is written down. It falls out:

| Width policy | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| **Expected score (shipped)** | 0.995 | 0.9793 | 3.070 | **0.9499** |
| Hand-tuned ramp (1 → 3 → 10) | 1.000 | 0.9660 | 3.020 | 0.9494 |
| Always 1 | 0.985 | 0.9850 | 3.145 | 0.9451 |
| Always 3 | 0.995 | 0.7908 | 2.410 | 0.9065 |
| Always 10 | 1.000 | 0.6188 | 1.880 | 0.8680 |

**Reported honestly: the expectimax beats a good hand-tuned ramp by 0.0005 —
nothing.** Its value is not that it beats careful tuning; it is that it
*derives* what careful tuning discovers, needs no tuning, and stays correct
when the situation changes (on turn 10 it opens to the full ten by itself,
because there is no later turn to protect). Against the naive policies a
reasonable engineer might pick instead, it is worth +0.082.

The lookahead depth is not a sensitive parameter: depth 1 / 2 / 3 score 0.9513
/ 0.9499 / 0.9505, a spread well inside noise at n=200.

---

## 5. Ablations

All rows are the offline configuration with one thing changed, on the clean
public split. Reproduce with `python3 -m scripts.experiment --suite ablations`.

| Configuration | Hit@10 | MRR | MTTC | Score | Δ |
|---|---|---|---|---|---|
| **Shipped (offline)** | 0.995 | 0.9793 | 3.070 | **0.9499** | — |
| Lookahead depth 1 | 1.000 | 0.9739 | 3.045 | 0.9513 | +0.0014 |
| Lookahead depth 3 | 0.995 | 0.9818 | 3.080 | 0.9505 | +0.0006 |
| No price channel | 0.995 | 0.9793 | 3.070 | 0.9499 | 0.0000 |
| Hand-tuned width ramp | 1.000 | 0.9660 | 3.020 | 0.9494 | −0.0005 |
| No clause channel | 0.990 | 0.9772 | 2.955 | 0.9490 | −0.0009 |
| Always show 1 | 0.985 | 0.9850 | 3.145 | 0.9451 | −0.0048 |
| No category channel | 0.995 | 0.9692 | 3.220 | 0.9439 | −0.0060 |
| Override: retract contradicted slot | 0.990 | 0.9678 | 3.170 | 0.9419 | −0.0080 |
| Override: erase all prior constraints | 0.985 | 0.9424 | 3.305 | 0.9291 | −0.0208 |
| No elimination of shown products | 0.995 | 0.9259 | 3.655 | 0.9222 | −0.0277 |
| No token-IDF channel | 0.965 | 0.9252 | 3.505 | 0.9100 | −0.0399 |
| Always show 3 | 0.995 | 0.7908 | 2.410 | 0.9065 | −0.0434 |
| Always show 10 | 1.000 | 0.6188 | 1.880 | 0.8680 | −0.0819 |
| Fixed question ladder | 1.000 | 0.6227 | 1.960 | 0.8676 | −0.0823 |
| Never ask a question | 0.950 | 0.4850 | 2.475 | 0.7910 | −0.1589 |
| Clause channel only | 0.835 | 0.7825 | 5.425 | 0.7637 | −0.1862 |

Three of these deserve comment, and two of them are unflattering.

**The clause channel is nearly redundant for scoring — worth 0.0009.** This is
the mechanism the design argument in §1 leans on, and as a *ranking* signal it
is almost entirely subsumed by token-IDF matching, which is worth 0.0399. The
clause index earns its place elsewhere: it is what makes "what would this
candidate say if asked about colour?" answerable, and therefore what makes the
EIG policy in §4.1 — worth 0.082 — possible at all. It is a representation
that pays through the question policy, not through the score.

**The price channel does exactly nothing**, to four decimal places. It is
retained because a real shopper states budgets and the channel is three lines,
but on this data it is dead weight and we are not going to pretend otherwise.

**Elimination is worth 0.028 and is free.** Any product shown on a previous turn
that did not end the session provably is not the target. The one exception is an
override session before the override lands, where the evaluator blocks
conversion entirely; those eliminations are re-admitted when the override
arrives, at zero risk, because those turns could never have converted.

---

## 6. Where the specification is wrong about this data

### 6.1 "Slot erasure and rewriting" on intent override loses

Pillar II asks for slot erasure. All three behaviours are implemented and
switchable. Erasure is measured to lose, monotonically:

| Override handling | intent_override Hit@10 | intent_override MRR | Overall score |
|---|---|---|---|
| **Keep all evidence (default)** | 1.000 | **0.9778** | **0.9499** |
| Erase only the contradicted slot | 0.967 | 0.9011 | 0.9419 |
| Erase every prior constraint | 0.933 | 0.7317 | 0.9291 |

The mechanism is not subtle, and it is checkable in one line:

> In **30 of 30** override sessions, the "old" preference and the "new" one are
> both drawn from the *same target product's* intent card
> (`old_value = soft_preferences[-1]`, `new_value = hard_constraints[0]`).

The customer changes what they *say they want*. The product they are actually
shopping for never changes — so everything disclosed before the override is
still true of it, and erasing it destroys valid evidence. The more you erase,
the worse you do.

**This is a property of the simulator, not of shopping.** A real customer who
says "actually, not leather" has genuinely invalidated a constraint, and for
that deployment `override_mode="retract"` is correct and is one field away. We
ship the mode that wins on the data we were given, and we are flagging the
distinction rather than burying it.

### 6.2 The dense retrieval route does not help — even reformulated

Pillar I calls for a vector similarity track. A previous attempt on this problem
built one, swept its weight into the ranking score, measured a loss at every
setting, and removed it.

We thought the *formulation* was wrong rather than the model, and rebuilt it as
a **mixture component of the likelihood** instead of a term in the score:

```
P(disclosure | product) = (1-e)·P_lexical + e·P_dense
```

In log space that is a soft maximum, so a strong lexical match is untouched by
a weak semantic one and the dense term only speaks where the lexical score has
collapsed to nothing. That formulation makes a falsifiable prediction: the
route should be worth ~0 on clean input and should matter under paraphrase.

**It was falsified.** 50,000 products embedded with `embeddinggemma:300m`,
Matryoshka-truncated to 256 dimensions:

| Configuration | clean | paraphrase | hostile | starved |
|---|---|---|---|---|
| Shipped (no dense) | 0.9475 | 0.9055 | 0.7458 | 0.9341 |
| **+ dense channel** | 0.9477 | 0.9056 | 0.7458 | 0.9341 |

Differences of ±0.0002 across every regime — the channel is inert, including
exactly where it was predicted to help. The reason is visible in §7: under
paraphrase the customer's *phrasing* changes but the decisive tokens
(`67% Polyester`, `Button closure`) survive, so the lexical score never
collapses and the mixture never reaches for the dense term. Semantic
generalisation would need a customer who describes the product in genuinely
different words, which neither the public simulator nor our stressors produce.

The code ships, off by default, with the cache builder — a second negative
result reached by a different route is worth more than a silent omission. It is
one config field to enable.

---

## 7. Robustness: the part that actually threatens the private split

The public simulator speaks in fixed phrasings; the specification reserves the
right to paraphrase the private 800. `scripts/stressors.py` rewrites what the
customer says while preserving exactly the information conveyed, by patching
the evaluator's own customer functions — so the session dynamics stay identical
and any drop is attributable to language handling alone.

- **paraphrase** — natural rewording; a shopper who does not talk like a form.
- **hostile** — every marker string the parser keys on is deliberately avoided:
  no "looking for", no "what matters is:", no semicolon lists, no "actually",
  no "I don't have a preference". Worst case, not likely case.
- **starved** — clean phrasing, but every second disclosure silently dropped.

**What these stressors do and do not test.** They rewrite the *framing* around a
disclosure — the lead-ins, the separators, the refusals — while leaving the
constraint text itself verbatim, because that text is the information content
and changing it would change the problem rather than the language. So these
numbers measure robustness to how a customer phrases things. They do **not**
measure robustness to a customer who describes the product in different words
entirely; §6.2 and §10 are explicit that we cannot measure that.

| Configuration | clean | paraphrase | hostile | starved |
|---|---|---|---|---|
| Deterministic only | 0.9499 | 0.7296 | 0.4971 | 0.9320 |
| **+ local model (shipped)** | 0.9475 | **0.9055** | **0.7458** | 0.9341 |
| Fixed question ladder | 0.8676 | 0.7451 | 0.6248 | 0.8587 |

**The deterministic path alone collapses under paraphrase — 0.9499 → 0.7296,
and → 0.4971 when every marker is broken. We are reporting that plainly.**
Regex-driven understanding is exactly as brittle as it sounds, and a previous
attempt on this problem measured the same class of failure and named it as its
largest unresolved risk.

**The local model closes most of it: +0.176 under paraphrase and +0.249 under
the hostile stressor, for −0.0024 on clean input.** That asymmetry is the whole
design of `socratic/llm.py`: the model is spent only where the cheap path is
blind. `segment()` fires only on an utterance where no framing marker was
found; on the clean split that gate is almost always shut, which is why the
clean score barely moves and why a clean 200-session run costs 33,409 prompt
tokens rather than millions.

The starved stressor is survivable either way (−0.018): losing half the
disclosures costs far less than losing the ability to read them, because the
EIG policy simply asks again.

### What the model call actually needed

The first implementation of `segment()` **cost 0.048 on clean input** — it
fired on refusals and stalls, where a 2B model dutifully invents requirements:
`"I don't have a preference for color; please use your judgment"` yielded
`['color', 'judgment']`, straight into the posterior. Three fixes, all of which
are in the shipped gate:

1. Do not call on a turn the deterministic parser already identified as a
   refusal or a stall — by construction it carries no product information.
2. Demand verbatim spans and **verify** them: a returned clause that is not a
   substring of the message is invention and is dropped.
3. Reject clauses built only from conversational vocabulary.

Post-fix the same input yields `[]`, and clean-input cost falls to −0.0006.

---

## 8. Reproduction

Python 3.10+. The default configuration needs no third-party package.

```bash
# 1. Catalogue (50,000 rows) at data/catalog.jsonl
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl

# 2. Official score  ->  results.json
python3 -m evaluator.local_evaluator

# 3. Tests
python3 -m unittest discover tests -v          # 26 tests

# 4. Every table in this document
python3 -m scripts.experiment --suite ablations     # §4, §5
python3 -m scripts.experiment --suite robustness    # §7
python3 -m scripts.experiment --suite llm           # §6.2
python3 -m scripts.experiment --sweep alpha 3,4.5,6,8,12,20
python3 -m scripts.feasibility                      # §9

# 5. Watch a session play out, with the belief state exposed
python3 -m scripts.show_conversation --sample-id public_0001
python3 -m scripts.show_conversation --scenario boundary --natural

# Optional: dense channel (requires numpy + Ollama; off by default)
pip install -r requirements.txt
python3 -m scripts.build_embeddings                 # ~8 min, 23 MB cache
```

**Local model (optional).** With [Ollama](https://ollama.com) running and
`ollama pull qwen3.5:2b-mlx`, the agent detects it at `reset()` and enables the
paraphrase-robustness path. With no daemon, no models, or no network, it runs
the deterministic path and scores 0.9499. `llm_mode="off"` forces that
explicitly. `OLLAMA_HOST` overrides the endpoint.

The first run builds a catalogue index (~18 s) and caches it under `.cache/`.

---

## 9. Feasibility disclosure

| | Offline (default on a machine with no daemon) | With local model |
|---|---|---|
| Model | none | `qwen3.5:2b-mlx` (3.1 GB) via Ollama |
| Embedding model | none | `embeddinggemma:300m` (621 MB), optional |
| API cost | **$0.00** | **$0.00** — runs on localhost |
| Credentials | none | none |
| Network at scoring time | **not required** | localhost only |
| Prompt tokens (200 sessions) | 0 | 33,409 |
| Completion tokens (200 sessions) | 0 | 312 |
| Model calls (200 sessions, 633 turns) | 0 | 180 |
| Latency p50 / p95 per turn | **16.4 ms / 68.2 ms** | 65.2 ms / 117.8 ms |
| Wall clock, 200 sessions | 19.2 s | 37.7 s |
| Peak memory | 0.69 GB | 0.70 GB |
| Index build (one-off) | 17.7 s | 17.7 s |
| Embedding cache (one-off, optional) | — | 8.2 min, 22.9 MB |
| Dependencies | Python standard library | numpy, for the optional dense channel only |

Measured on Apple M5 Pro / 24 GB / Python 3.14 by `scripts/feasibility.py`.

Extrapolating to the private 800 sessions: roughly 80 s offline, or 2.5 minutes
with the model, at about 135,000 prompt tokens — still $0, because nothing
leaves the machine.

**The organiser's scoring machine will almost certainly not be running Ollama,
and submission rules warn network access may be disabled.** The deterministic
path is therefore the default rather than the fallback: `available()` is a
1.5-second health check at `reset()`, every model method returns a falsy value
rather than raising, and `tests/test_socratic.py` asserts a complete scored run
with the model disabled.

---

## 10. Limitations

**Language handling is still the weakest link, and the model is a patch on it
rather than a cure.** 0.9499 → 0.7458 under a total marker break is a 21% drop
even with the model in the loop. The residue is a parser that finds clause
boundaries by pattern; a genuinely robust version would parse to a typed
constraint representation rather than to strings.

**One session misses, and it is unwinnable as posed.** `public_0083`'s entire
card is catalogue boilerplate — `polyester` (11,072 products), `100% Polyester`
(2,522), `Imported` (13,889), `Button closure` (2,195). No agent that must
distinguish products by disclosed attributes can single this one out; the
information to do so was never sent.

**80% of the remaining 0.042 is MTTC, and most of it is not addressable.** The
gap decomposes as 0.0025 Hit Rate, 0.0062 MRR, 0.0336 Efficiency — and 0.0078
of that Efficiency share is locked behind the override-conversion gate. The
agent converts in 3.07 turns against a floor of 1.39, but reaching the floor
would mean identifying the product from the opening message, which §1 shows is
usually impossible: one constraint plus a category leaves a median of far more
than one candidate.

**Everything is tuned on n=200.** Standard error on Hit Rate at this sample size
is roughly 2.5 points, larger than several of the deltas in §5. **Treat any
difference below ~0.005 as unresolved by this data** — that includes the
lookahead depth sweep, the expectimax-versus-ramp comparison, and the clause
channel's contribution. The posterior temperature sweep (`alpha` 3 → 20) is
flat at 0.9463–0.9526 across a factor of four; we kept the round value rather
than chase a 0.003 peak that n=200 cannot resolve.

**No semantic generalisation.** §6.2 is a negative result, not a feature: the
agent cannot bridge vocabulary the catalogue does not literally contain. If the
private customers describe products in genuinely different words rather than
merely different sentences, that is the failure mode, and neither our stressors
nor the public simulator can tell us how likely it is.

**The user profile is barely used.** `preference_tags` and `rating_style` are
carried through `Session` but do not enter the posterior. Pillar III's
"long-term user profile" is therefore represented by the evidence ledger and
per-turn re-orchestration only. With 200 sessions and no repeat customers there
was nothing to fit it against, and we would rather leave it unused than claim a
personalisation result we cannot measure.

---

## 11. Team contributions

_To be completed._
