"""Cheap per-trajectory signals used to ground the failure taxonomy (no LLM calls)."""
import json, sys, collections
from datasets import load_dataset
from eval.metrics import norm
from analysis.summarize import contains


def gold_titles(n, seed=0):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=seed).select(range(n))
    return {ex["id"]: set(ex["supporting_facts"]["title"]) for ex in ds}


def signals(r, gold):
    steps = [s for s in r["traj"] if s["action"]]
    searches = [s["action"][1] for s in steps if s["action"][0] == "Search"]
    obs_titles = []
    for s in steps:
        if s["obs"] and s["obs"].startswith("["):
            obs_titles.append(s["obs"][1:s["obs"].index("]")])
    seen_answer = any(norm(r["gold"]) in norm(s["obs"] or "") for s in steps)
    return {
        "dup_query": len(searches) - len(set(q.lower() for q in searches)),
        "max_same_title": max(collections.Counter(obs_titles).values(), default=0),
        "lookup_noresult": sum(1 for s in steps if s["action"][0] == "Lookup" and s["obs"] in ("No Results", "No more results.\n")),
        "gold_page_hit": len(gold & set(obs_titles)),
        "gold_total": len(gold),
        "seen_answer": seen_answer,
    }


if __name__ == "__main__":
    rs = [json.loads(l) for l in open(sys.argv[1])]
    g = gold_titles(len(rs))
    groups = {"max_steps": [r for r in rs if r["halt"] == "max_steps"],
              "finished_wrong": [r for r in rs if r["halt"] == "finish" and not contains(r["answer"], r["gold"])],
              "finished_right": [r for r in rs if r["halt"] == "finish" and contains(r["answer"], r["gold"])]}
    for name, grp in groups.items():
        S = [signals(r, g[r["id"]]) for r in grp]
        n = len(S)
        print(f"{name} (n={n})")
        print("  mean gold supporting pages retrieved: %.2f / %.2f" % (sum(s['gold_page_hit'] for s in S)/n, sum(s['gold_total'] for s in S)/n))
        print("  ZERO gold pages ever retrieved: %d  | BOTH retrieved: %d" % (sum(s['gold_page_hit'] == 0 for s in S), sum(s['gold_page_hit'] >= s['gold_total'] for s in S)))
        print("  gold answer string appeared in an observation: %d" % sum(s['seen_answer'] for s in S))
        print("  >=2 duplicate queries: %d | same page returned >=3x: %d | >=2 empty Lookups: %d" % (
            sum(s['dup_query'] >= 2 for s in S), sum(s['max_same_title'] >= 3 for s in S), sum(s['lookup_noresult'] >= 2 for s in S)))
