"""ReAct loop: Thought / Action / Observation, actions are Search[q], Lookup[kw] and Finish[a]."""
import re
from .llm import chat

SYSTEM = (
    "Solve a question answering task with interleaving Thought, Action, Observation steps. "
    "Thought can reason about the current situation, and Action can be three types:\n"
    "(1) Search[entity], which returns the best matching paragraph.\n"
    "(2) Lookup[keyword], which returns the next sentence containing keyword in the last paragraph Searched.\n"
    "(3) Finish[answer], which returns the answer and finishes the task.\n"
    "Output exactly one Thought and one Action per turn, then stop. Here are some examples.\n"
)
# Lenient: tolerates "Action\nSearch[..]" (no colon) and numbered "Action 3:". STRICT_RE marks
# exact-format outputs so format deviations stay countable as a failure mode.
ACTION_RE = re.compile(r"Action\s*\d*\s*:?\s*(Search|Lookup|Finish)\[(.*?)\]", re.S)
STRICT_RE = re.compile(r"^Action( \d+)?: (Search|Lookup|Finish)\[", re.M)


def run_react(model, question, tool, max_steps=7, fewshot="", seed=0):
    """Returns dict with answer, trajectory (list of step dicts) and halt reason."""
    messages = [{"role": "system", "content": SYSTEM + fewshot},
                {"role": "user", "content": f"Question: {question}"}]
    traj = []
    for step in range(1, max_steps + 1):
        out = chat(model, messages, stop=["Observation"], seed=seed)
        m = ACTION_RE.search(out)
        if not m:
            traj.append({"step": step, "raw": out, "action": None, "obs": "Invalid action format.", "fmt_ok": False})
            messages += [{"role": "assistant", "content": out},
                         {"role": "user", "content": f"Observation {step}: Invalid action format. Use Search[..], Lookup[..] or Finish[..]."}]
            continue
        kind, arg = m.group(1), m.group(2).strip()
        fmt_ok = bool(STRICT_RE.search(out))
        if kind == "Finish":
            traj.append({"step": step, "raw": out, "action": ("Finish", arg), "obs": None, "fmt_ok": fmt_ok})
            return {"answer": arg, "traj": traj, "halt": "finish"}
        obs = tool.search(arg) if kind == "Search" else tool.lookup(arg)
        traj.append({"step": step, "raw": out, "action": (kind, arg), "obs": obs, "fmt_ok": fmt_ok})
        messages += [{"role": "assistant", "content": out},
                     {"role": "user", "content": f"Observation {step}: {obs}"}]
    return {"answer": "", "traj": traj, "halt": "max_steps"}
