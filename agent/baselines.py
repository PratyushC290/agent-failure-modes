"""Closed-book baselines (no tool), as in the ReAct paper: direct answer and CoT."""
import re
from .llm import chat

DIRECT = "Answer the question with a short answer (a few words). Output only: Answer: <answer>"
COT = ("Think step by step, then give a short final answer (a few words).\n"
       "Format:\nThought: <reasoning>\nAnswer: <answer>")
ANS_RE = re.compile(r"Answer:\s*(.+)", re.S)


def _run(system, model, question, max_tokens, seed):
    out = chat(model, [{"role": "system", "content": system},
                       {"role": "user", "content": f"Question: {question}"}],
               max_tokens=max_tokens, seed=seed)
    m = ANS_RE.search(out)
    ans = (m.group(1).strip().splitlines()[0] if m else "").strip()
    return {"answer": ans, "traj": [{"step": 1, "raw": out, "action": ("Finish", ans) if m else None, "obs": None}],
            "halt": "finish" if m else "no_answer"}


def run_direct(model, question, seed=0):
    return _run(DIRECT, model, question, 40, seed)


def run_cot(model, question, seed=0):
    return _run(COT, model, question, 300, seed)
