"""Thin chat client for any OpenAI-compatible server (Ollama, vLLM)."""
import os
from openai import OpenAI

_client = None


def client(base_url=None):
    base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    global _client
    if _client is None:
        _client = OpenAI(base_url=base_url, api_key="local")
    return _client


def chat(model, messages, temperature=0.0, max_tokens=256, stop=None, seed=0):
    r = client().chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stop=stop, seed=seed,
    )
    return r.choices[0].message.content or ""
