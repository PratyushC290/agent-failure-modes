"""SQuAD-style EM/F1 and bootstrap CIs."""
import re, string, collections
import numpy as np


def norm(s):
    s = s.lower().replace("\u2019", "'").replace("\u2018", "'")
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return float(norm(pred) == norm(gold))


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if not p or not g or n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def relaxed(pred, gold):
    """Lenient match: one side's normalized token set contains the other's.

    Counts name-part omissions ("Marianne Faithfull" vs full legal name) and word-order or
    date-format changes as correct. Known false-positive risk: a single shared token such as
    "Washington" vs "George Washington". Report alongside EM/F1, never instead of them.
    """
    p, g = set(norm(pred).split()), set(norm(gold).split())
    return float(bool(p) and bool(g) and (p <= g or g <= p))


def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])
