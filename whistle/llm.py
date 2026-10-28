"""Thin OpenAI-compatible chat client with on-disk caching, retries and a mock backend.

A model is specified as "provider:model", e.g. "groq:llama-3.3-70b-versatile",
"gemini:gemini-2.0-flash", "openrouter:meta-llama/llama-3.3-70b-instruct:free",
"ollama:qwen2.5:7b", or "mock" for offline smoke tests.
"""

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "mock": (None, None),
}

CACHE_DIR = Path(".cache/llm")


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def extract_json(text):
    """Pull the first JSON object out of a model response (tolerates fences and <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class LLM:
    def __init__(self, spec):
        self.spec = spec
        self.provider, _, self.model = spec.partition(":")
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; choose from {list(PROVIDERS)}")
        self.client = None
        if self.provider != "mock":
            from openai import OpenAI

            base_url, key_env = PROVIDERS[self.provider]
            api_key = os.environ.get(key_env) if key_env else "ollama"
            if not api_key:
                raise RuntimeError(f"set {key_env} in .env to use {self.provider}")
            self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    def chat(self, system, user, *, task, temperature=0.7, sample=0):
        """One chat completion -> (text, latency_s). `sample` distinguishes otherwise-identical cached calls.

        Latency is measured on the original call and stored in the cache, so cached reruns
        report the same decision time.
        """
        key = hashlib.sha256(
            json.dumps([self.spec, system, user, temperature, sample]).encode()
        ).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            hit = json.loads(path.read_text())
            return hit["response"], hit.get("latency_s")

        t0 = time.perf_counter()
        if self.provider == "mock":
            response = _mock_response(task, user, key)
        else:
            response = self._call(system, user, temperature)
        latency = time.perf_counter() - t0

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"spec": self.spec, "task": task, "response": response, "latency_s": latency}))
        return response, latency

    def chat_json(self, system, user, *, task, temperature=0.7, sample=0, attempts=3):
        """Parsed JSON response; adds "_latency_s" (summed over parse retries)."""
        last_err, total = None, 0.0
        for i in range(attempts):
            text, latency = self.chat(system, user, task=task, temperature=temperature, sample=f"{sample}/{i}")
            total += latency or 0.0
            try:
                out = extract_json(text)
                out["_latency_s"] = total
                return out
            except ValueError as e:
                last_err = e
        raise last_err

    def _call(self, system, user, temperature):
        from openai import APIConnectionError, APIStatusError, RateLimitError

        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=temperature,
                )
                return r.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError) as e:
                wait = min(60, 2**attempt) + random.random()
                print(f"  [{self.spec}] {type(e).__name__}, retrying in {wait:.0f}s")
                time.sleep(wait)
            except APIStatusError as e:
                if e.status_code >= 500:
                    time.sleep(min(60, 2**attempt))
                    continue
                raise
        raise RuntimeError(f"{self.spec}: gave up after repeated rate limits")

    def list_models(self):
        return sorted(m.id for m in self.client.models.list().data)


def _mock_response(task, user, key):
    """Plausible-shaped random JSON so the pipeline can be smoke-tested offline."""
    rng = random.Random(key)
    events = sorted(set(re.findall(r"\bE\d+\b", user))) or ["E1"]
    items = sorted(set(re.findall(r"\[([RTS]\d+)\]", user)))
    if task == "reporter":
        e = rng.choice(events)
        return json.dumps({"reports": [{"text": f"I am concerned about {e}.", "about_event": e}]})
    if task == "flooder":
        n = int(re.search(r"exactly (\d+) reports", user).group(1))
        return json.dumps({"reports": [
            {"text": f"Agent did something bad in {e}.", "about_event": e, "tactic": "false_accusation"}
            for e in rng.choices(events, k=n)
        ]})
    if task == "triager":
        ids = re.findall(r"\[(R\d+)\]", user)
        rng.shuffle(ids)
        groups = [ids[i : i + 2] for i in range(0, len(ids), 2)][:5]
        return json.dumps({"issues": [
            {"summary": f"Concern raised in {', '.join(g)}.", "priority": "high", "sources": g} for g in groups
        ]})
    if task == "extractor":
        ids = re.findall(r"\[(R\d+)\]", user)
        return json.dumps({"reports": {
            rid: {"accused_agent": "A1", "alleged_action": "something", "rule": "rule 1",
                  "cited_events": [rng.choice(events)]}
            for rid in ids
        }})
    if task == "reviewer":
        return json.dumps({"escalate": rng.sample(items, k=min(len(items), rng.randint(0, 2))),
                           "rationale": "mock"})
    raise ValueError(task)
