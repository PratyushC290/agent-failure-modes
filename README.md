# Agent failure modes with small open LLMs

Reproduce ReAct / Reflexion on HotpotQA (distractor) with small open models, build a failure
taxonomy, and test cheap inference-time fixes. See the plan for details.

## Quick start
```
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
python -m eval.run_hotpot --n 10 --out results/smoke.jsonl
```
