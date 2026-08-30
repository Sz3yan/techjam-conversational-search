"""Local model access via Ollama, health-gated and used sparingly.

Three design rules, in order of importance:

1. **The deterministic path is the default, not the fallback.** The organiser's
   scoring machine will almost certainly not be running Ollama and submission
   rules warn network access may be disabled, so every method here returns a
   falsy value rather than raising. The agent scores a complete run with the
   daemon down.

2. **Spend the model only where the cheap path is blind.** `segment()` fires
   only on an utterance where no known framing marker was found -- that is,
   exactly the paraphrased input the regex parser cannot see. `rerank()` fires
   only when the posterior's top candidates are too close to separate. On the
   clean public split both gates are almost always shut, so the LLM costs
   nearly nothing there and earns its keep under paraphrase.

3. **Report what it costs.** Ollama returns real token counts; they are passed
   straight through to the `usage` field rather than estimated.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
HOST = _HOST if _HOST.startswith("http") else f"http://{_HOST}"
HEALTH_TIMEOUT = 1.5
CALL_TIMEOUT = 20.0
MAX_CLAUSE = 180

SEGMENT_PROMPT = """Extract the shopper's PRODUCT requirements from the message.

Return ONLY a JSON object, no prose:
{{"requirements": [...], "refuses": true/false, "attribute": "<attribute they declined to specify, or null>", "changed_mind": true/false}}

Rules:
- Each requirement must be copied VERBATIM from the message.
- A requirement describes the product (a fabric, colour, size, feature, use).
- If the message states no product attribute, requirements MUST be [].
- Never turn a refusal or a complaint into a requirement.
- "refuses" is true only if they decline to state a preference.
- "changed_mind" is true only if they are replacing something said earlier.

Example. Message: "I don't have a preference for colour, your call"
{{"requirements": [], "refuses": true, "attribute": "color", "changed_mind": false}}

Example. Message: "it is 100% cotton and machine washable"
{{"requirements": ["100% cotton", "machine washable"], "refuses": false, "attribute": null, "changed_mind": false}}

Message: "{message}"
JSON:"""

RERANK_PROMPT = """A shopper wants a product meeting ALL of these:
{constraints}

Candidates:
{candidates}

Which candidate best satisfies every requirement? Reply with ONLY its number.
Answer:"""

MESSAGE_PROMPT = """You are a shopping assistant. Write ONE short, natural sentence \
presenting this recommendation, then ONE short question asking about {attribute}.

Known requirements: {constraints}
Recommending: {title}

Reply with the two sentences only, no preamble."""

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
# Words that describe the conversation rather than the product. A phrase built
# only from these is the model paraphrasing the question back at us.
META_TOKENS = frozenset("""
preference preferences attribute attributes judgment judgement option options
specific additional more else anything nothing something thing things matter
matters requirement requirements need needs want wants opinion choice call
yours mine your my the a an no not don t have has for about on of and or is
are be it that this these those please sorry ok okay yes yeah right well
""".split())


class LocalModel:
    def __init__(self, config) -> None:
        self.config = config
        self.host = HOST
        self._available: bool | None = None
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.calls = 0

    # ---------------------------------------------------------------- infra

    def available(self) -> bool:
        if self._available is None:
            self._available = self._probe()
        return self._available

    def _probe(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=HEALTH_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False
        names = {model.get("name", "") for model in payload.get("models", [])}
        return self.config.llm_model in names

    def _generate(self, prompt: str, *, num_predict: int = 128) -> str:
        if not self.available():
            return ""
        body = json.dumps({
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": num_predict},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=CALL_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return ""
        self.calls += 1
        self._usage["prompt_tokens"] += int(payload.get("prompt_eval_count") or 0)
        self._usage["completion_tokens"] += int(payload.get("eval_count") or 0)
        return str(payload.get("response") or "").strip()

    def take_usage(self) -> dict:
        usage, self._usage = self._usage, {"prompt_tokens": 0, "completion_tokens": 0}
        return usage

    # ----------------------------------------------------------------- uses

    def segment(self, message: str, parsed) -> list[str]:
        """Recover constraints from an utterance with no recognised markers.

        Returns [] when the deterministic parser already found its footing, so
        on clean input this costs nothing at all.
        """
        if "segment" not in self.config.llm_uses or not message.strip():
            return []
        if parsed.marked:
            # A known framing marker was found, so the deterministic parser had
            # its footing. Spending a model call here would buy nothing.
            return []
        if parsed.is_stall or (parsed.refused_attribute and not parsed.constraints):
            # The customer told us nothing about the product. Asking a model to
            # find requirements in a refusal only invents them -- measured at
            # -0.048 before this gate existed.
            return []
        if not self.available():
            return []
        raw = self._generate(SEGMENT_PROMPT.format(message=message.replace('"', "'")), num_predict=160)
        if not raw:
            return []
        match = JSON_OBJECT.search(raw)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return []
        if data.get("refuses") and not parsed.refused_attribute:
            attribute = data.get("attribute")
            if isinstance(attribute, str) and attribute:
                parsed.refused_attribute = attribute.strip().lower().replace(" ", "_")
        if data.get("changed_mind"):
            parsed.is_override = True
        haystack = message.lower()
        out: list[str] = []
        for item in data.get("requirements") or []:
            if not isinstance(item, str):
                continue
            clause = item.strip()[:MAX_CLAUSE].lower()
            if not clause or clause in out:
                continue
            # The prompt demands verbatim spans; anything else is invention.
            if clause not in haystack:
                continue
            if all(token in META_TOKENS for token in re.findall(r"[a-z]+", clause)):
                continue
            out.append(clause)
        return out[:4]

    def rerank(self, constraints: list[str], candidates: list[tuple[int, str]]) -> int | None:
        """Pick the best candidate when the posterior cannot separate them."""
        if "rerank" not in self.config.llm_uses or len(candidates) < 2:
            return None
        if not self.available():
            return None
        listing = "\n".join(f"{i + 1}. {title[:110]}" for i, (_pid, title) in enumerate(candidates))
        wanted = "\n".join(f"- {c[:110]}" for c in constraints[:6]) or "- (not yet stated)"
        raw = self._generate(
            RERANK_PROMPT.format(constraints=wanted, candidates=listing), num_predict=8
        )
        match = re.search(r"\d+", raw)
        if not match:
            return None
        choice = int(match.group(0))
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1][0]
        return None

    def message(self, session, attribute, chosen, index) -> str:
        if "message" not in self.config.llm_uses or not chosen:
            return ""
        if not self.available():
            return ""
        constraints = ", ".join(session.belief.disclosed[-3:]) or "nothing specific yet"
        return self._generate(
            MESSAGE_PROMPT.format(
                attribute=attribute or "anything else that matters",
                constraints=constraints[:300],
                title=index.titles[chosen[0]][:120],
            ),
            num_predict=80,
        )
