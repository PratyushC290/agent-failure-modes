# Failure taxonomy, v1 (single-rater labels, pending agreement check)

Source: `results/gemma12b_react_n200.jsonl` (gemma3:12b, ReAct + few-shot, n=200). 200 runs = 114 right, 51 finished wrong, 35 hit the 7-step limit (right/wrong judged by the lenient "contains gold" check). Drafted from reading 18 of the 35 step-limit runs and 24 of the 51 wrong-but-finished runs. **Counts below are from the automated signals (`analysis/signals.py`) over all runs; the category counts are NOT yet measured. They need the labeling pass.**

## A. Runs that hit the step limit (35)
| code | name | what it looks like | example |
|---|---|---|---|
| S1 | Retrieval-miss loop | Never reaches a needed page. Often searches a description ("American singer songwriter actress LGBT activist") instead of an entity name, and our word-overlap retriever returns something unrelated. | step-limit #0, #3, #6, #11 |
| S2 | Query churn | Reaches a page, then keeps rewording the query and gets the same paragraph back. | #5, #7, #10, #12, #14 |
| S3 | Answer seen, not recognized | The gold answer text is already in an observation but the agent keeps searching. | #7 (saw "Sir Nicholas Raymond Winterton" at step 1) |
| S4 | Lookup misuse | Repeated `Lookup[...]` for a keyword that isn't in the last paragraph, giving "No Results". | #1, #4, #9, #15, #16 |
| S5 | Ungrounded entity guessing | Searches or looks up entities no observation has mentioned (guessed from memory). | #0 ("Lady Gaga"), #9 (five actors), #15 ("Stuyvesant Town") |

Signals over all 35: same page returned 3 or more times in 32 runs; no gold supporting page ever retrieved in 12; both retrieved in only 3; gold answer text appeared in an observation in 13 (upper bound, short answers can match by chance).

## B. Finished but wrong (51)
| code | name | what it looks like | example |
|---|---|---|---|
| F1 | Evaluation artifact | Answer is right up to format or granularity. Fixable by metric, not by the agent. | "Marianne Faithfull" vs full legal name; "August 15, 1843" vs "15 August 1843" |
| F2 | Wrong-document answer | Answers confidently from one retrieved paragraph that is a distractor or wrong entity. | "BitTorrent Open Source License" (wrong game); "To Pimp a Butterfly" |
| F3 | Premature stop | Stops after one hop and does not verify the second. | Venus/Pulitzer: answered "OBIE Awards" |
| F4 | Composition error | Has the right evidence but combines it wrongly. | Perri vs That Darn Cat! (wrong date comparison); Augustus vs Caligula |
| F5 | Answer-type mismatch | Answers a different aspect than asked. | "Eugenio Derbez" for "what distinction"; "the title role in ..." instead of a name |
| F6 | Parametric override | Contradicts or ignores the observation using memorized knowledge. | The Shining author: answered "Arthur C. Clarke" |
| F8 | Gold-label problem | The dataset gold is wrong or not a valid answer (e.g. gold "by" for a yes/no question). Added after labeling. | Perri vs That Darn Cat! |
| F7 | Forced guess or abstention | After getting stuck, gives up or invents an answer. | "I am unable to determine ..."; Layne Staley "April 2002" |

Signals over all 51: both gold pages retrieved in 17 (so the error is the agent's own), none retrieved in 10 (retrieval), gold text appeared in an observation in 32.

## Measured counts (rater 1 = Claude; `analysis/labels_claude.csv`; n=86 failures of 200 ReAct runs)
| step-limit runs (35) | n | finished-wrong runs (51) | n |
|---|---|---|---|
| S2 query churn | 16 | F1 evaluation artifact | 15 |
| S1 retrieval-miss loop | 11 | F5 answer-type mismatch | 15 |
| S5 ungrounded entity guessing | 5 | F3 premature stop | 5 |
| S3 answer seen, not recognized | 3 | F4 composition error | 5 |
| S4 Lookup misuse | 0 primary (2 secondary) | F2 wrong-document | 3 |
| | | F7 forced guess / abstention | 3 |
| | | F8 gold-label problem | 3 |
| | | F6 parametric override | 2 |

Takeaways (single rater, n=86, treat as provisional):
- **15 of the 51 wrong answers are really correct** (F1: "Ronald Reagan" vs "Ronald Wilson Reagan", date format, apostrophes). The strict "contains gold" check misses name-part omissions. The token-subset `relaxed` metric (added to `eval/metrics.py`) recovers 11 of these 15 (ReAct 0.575 -> 0.615) and flagged no other labeled failure as correct. The remaining 4 need semantic judgment (e.g. "Russian" vs "The cinema of Russia"). My earlier estimate of 0.64 was too high. Report EM, F1 and relaxed together.
- **F5 is as large as any real error class**: the agent finds the right page but answers a different aspect (a county instead of a county seat, an actor instead of a TV series).
- **Half the step-limit failures are S2 (query churn)**, which loop detection could plausibly help. S4 never occurred as the main cause, so it will likely be merged into S2/S5 or dropped.
- 3 gold-label problems (F8) show the benchmark itself is noisy.

## Confound to resolve: tool failure vs agent failure
Only 12 of 35 step-limit runs and 10 of 51 wrong runs never retrieved a gold page, but many partial-retrieval cases (S1, F2) are caused by our retriever, a word-overlap search over 10 paragraphs, not by the agent. The paper's Wikipedia search is stronger. A fair taxonomy has to separate the two, e.g. with an oracle-retriever ablation or by tagging each failure with "gold page retrievable or not."

### Resolved (decision: tag it, keep the tool)
`analysis/label_sheet.py` tags each run `tool_capable` = querying the tool with each gold title returns that page at rank 1 (so the tool could serve it if the agent names the entity). Results on the n=200 ReAct run:

| outcome | tool_capable | not capable |
|---|---|---|
| right | 103 | 11 |
| hit step limit | 30 | 5 |
| finished wrong | 47 | 4 |

Failure rate is 77/180 = 43% when the tool is capable and 9/20 = 45% when it is not (small n for the second). So retriever quality does not explain the failure rate at the exact-title level. The remaining difficulty is naming the right entity, especially the second-hop (bridge) entity, which the question does not state. Caveat: "capable" assumes the agent already knows the title, which is the hard part in multi-hop questions.

## Next
1. Done: rater 1 labels (`analysis/labels_claude.csv`).
2. Agreement check: the project owner labels the blind sample `analysis/labels_blind_sample.csv` (30 of the 86, fixed seed, without seeing rater 1's labels), then we compute Cohen's kappa. Disagreements decide which categories to merge or redefine.
3. Only after that is an LLM-assisted labeler worth running, and it is compared against the human labels first.
