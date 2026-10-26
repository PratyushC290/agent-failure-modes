#!/bin/sh
# Sequential n=200 passes on the shared remote GPU (one request at a time). Same seed => same 200 questions.
export LLM_BASE_URL=http://localhost:11435/v1
cd "$(dirname "$0")/.."
M=gemma3:12b
.venv/bin/python -m eval.run_hotpot --method direct --model $M --n 200 --out results/gemma12b_direct_n200.jsonl
.venv/bin/python -m eval.run_hotpot --method cot    --model $M --n 200 --out results/gemma12b_cot_n200.jsonl
.venv/bin/python -m eval.run_hotpot --method react  --model $M --n 200 --fewshot configs/react_fewshot.txt --out results/gemma12b_react_n200.jsonl
