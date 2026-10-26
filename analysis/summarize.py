"""Summarize result JSONL files: EM / F1 / lenient-contains with 95% bootstrap CIs, halt reasons."""
import json, sys, collections
from eval.metrics import norm, em, f1, relaxed, bootstrap_ci


def contains(pred, gold):
    p, g = norm(pred), norm(gold)
    return float(bool(p) and bool(g) and (g in p or p in g))


def summarize(path):
    rs = [json.loads(l) for l in open(path)]
    out = {"n": len(rs), "halt": dict(collections.Counter(r["halt"] for r in rs))}
    # Recompute from saved answers so metric changes apply to old runs (stored em/f1 may be stale).
    for k, vals in [("em", [em(r["answer"], r["gold"]) for r in rs]), ("f1", [f1(r["answer"], r["gold"]) for r in rs]),
                    ("contains", [contains(r["answer"], r["gold"]) for r in rs]),
                    ("relaxed", [relaxed(r["answer"], r["gold"]) for r in rs])]:
        m, lo, hi = bootstrap_ci(vals)
        out[k] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
    out["avg_steps"] = round(sum(len(r["traj"]) for r in rs) / len(rs), 2)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p.split("/")[-1], summarize(p))
