"""Run a method on a HotpotQA distractor subset; write trajectories to JSONL."""
import argparse, json, pathlib
from datasets import load_dataset
from tqdm import tqdm
from agent.react import run_react
from agent.baselines import run_direct, run_cot
from agent.tools import ParagraphSearch
from eval.metrics import em, f1

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5:3b-instruct")
ap.add_argument("--method", choices=["react", "direct", "cot"], default="react")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--fewshot", default="", help="path to few-shot text for ReAct")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/react.jsonl")
a = ap.parse_args()
fewshot = open(a.fewshot).read() if a.fewshot else ""

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=a.seed).select(range(a.n))
pathlib.Path(a.out).parent.mkdir(exist_ok=True)
with open(a.out, "w") as f:
    for ex in tqdm(ds):
        tool = ParagraphSearch(ex["context"]["title"], ex["context"]["sentences"])
        if a.method == "react":
            r = run_react(a.model, ex["question"], tool, fewshot=fewshot, seed=a.seed)
        else:
            r = (run_direct if a.method == "direct" else run_cot)(a.model, ex["question"], seed=a.seed)
        r.update(id=ex["id"], question=ex["question"], gold=ex["answer"], model=a.model, method=a.method,
                 em=em(r["answer"], ex["answer"]), f1=f1(r["answer"], ex["answer"]))
        f.write(json.dumps(r) + "\n")
