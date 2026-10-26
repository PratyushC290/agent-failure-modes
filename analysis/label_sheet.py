"""Build a review sheet of failed runs: compact trajectory + tool-capability tag + empty label column."""
import json, sys, csv
from datasets import load_dataset
from agent.tools import ParagraphSearch
from analysis.signals import signals
from analysis.summarize import contains


def tool_capable(ex):
    """True if querying the tool with each gold title returns that page at rank 1."""
    t = ParagraphSearch(ex["context"]["title"], ex["context"]["sentences"])
    return all(t.search(g).startswith(f"[{g}]") for g in set(ex["supporting_facts"]["title"]))


def compact(r):
    lines = []
    for s in r["traj"]:
        a = s["action"]
        lines.append(f"{s['step']}. " + (f"{a[0]}[{a[1][:60]}]" if a else "INVALID") + " -> " + (s["obs"] or "")[:70].replace("\n", " "))
    return " | ".join(lines)


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    rs = [json.loads(l) for l in open(path)]
    ds = {ex["id"]: ex for ex in load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=0).select(range(len(rs)))}
    rows, stats = [], {}
    for r in rs:
        ex = ds[r["id"]]
        cap = tool_capable(ex)
        r["_cap"] = cap
        bad = r["halt"] == "max_steps" or not contains(r["answer"], r["gold"])
        key = ("right" if not bad else r["halt"] if r["halt"] == "max_steps" else "wrong_finished", cap)
        stats[key] = stats.get(key, 0) + 1
        if bad:
            sg = signals(r, set(ex["supporting_facts"]["title"]))
            rows.append({"id": r["id"], "outcome": r["halt"], "tool_capable": int(cap), "gold_pages_hit": f"{sg['gold_page_hit']}/{sg['gold_total']}",
                         "answer_seen": int(sg["seen_answer"]), "question": r["question"], "gold": r["gold"], "pred": r["answer"],
                         "trajectory": compact(r), "label": "", "secondary": "", "notes": ""})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    for k in sorted(stats, key=str): print(k, stats[k])
