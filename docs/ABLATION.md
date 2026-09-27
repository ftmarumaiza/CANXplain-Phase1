\# Ablation Study — CANXplain Phase 1



Nine ablation arms (A0–A8), each isolating one component of the

proposed system. All arms were run under `configs/full.yaml`: 3 seeds

(0, 1, 2), 3 source datasets (`car\_hacking`, `can\_ids`, `road`), both

same-dataset and cross-dataset evaluation. 348 total run records in

`results/full/runs.jsonl`. This file is a static counterpart to the

version generated at runtime into `results/full/ABLATION.md`.



Values below are taken directly from `results/full/SUMMARY.txt`,

`HANDOFF.md`, and `results/full/runs.jsonl` as checked during the run.

Nothing here is estimated or extrapolated beyond what those files

report.



\---



\## 1. The arms



Each arm removes exactly one component from the full system (A0),

except A6–A8, which instead restrict the feature set. `removes`

quotes the codebase's own description of what changes relative to A0.



| Arm | Name | Removes (relative to A0) | Meta-learning search | SHAP term (β) | Complexity term (γ) | Feature groups |

|---|---|---|---|---|---|---|

| A0 | Full CANXplain | nothing — the complete proposed system | yes | yes | yes | temporal, temporal\_id, frequency, dlc (23 features) |

| A1 | Without meta-learning | the ACHILLES-inspired competition/evolution search | no (fixed default RF/DNN) | yes | yes | same as A0 |

| A2 | Accuracy-only selection | explanation-awareness and the complexity penalty from selection | yes | no | no | same as A0 |

| A3 | Without SHAP consistency | the β·SHAP\_consistency term only | yes | no | yes | same as A0 |

| A4 | Without complexity penalty | the γ·Complexity term only | yes | yes | no | same as A0 |

| A5 | No meta-learning + accuracy-only | both the search and the explanation/complexity terms | no | no | no | same as A0 |

| A6 | Reduced feature set | the proposed temporal and frequency feature engineering | yes | yes | yes | basic (3 features) |

| A7 | Temporal features only | the frequency / CAN-ID statistics groups | yes | yes | yes | temporal, temporal\_id (varies by dataset) |

| A8 | Frequency features only | the timing-based feature groups | yes | yes | yes | frequency (varies by dataset) |



\### Fitness expressions actually used



\- A0, A6, A7, A8: `0.5·balanced\_accuracy + 0.3·SHAP\_consistency − 0.2·complexity\_norm`

\- A1: same expression as A0, evaluated over a fixed candidate set (no search)

\- A2: `0.5·balanced\_accuracy` (search retained, SHAP/complexity not in the objective)

\- A3: `0.5·balanced\_accuracy − 0.2·complexity\_norm`

\- A4: `0.5·balanced\_accuracy + 0.3·SHAP\_consistency`

\- A5: `0.5·balanced\_accuracy`, evaluated over a fixed candidate set (no search)



For A2, A3, A5, SHAP consistency is still \*\*measured\*\* for the

selected model (for comparability against A0 in Table 8) even though

it does not influence selection.



\### The question each arm answers (as documented in `ablations.py`)



\- \*\*A0\*\* — reference: what does the full system achieve, same-dataset

&#x20; and cross-dataset. Every other arm is compared against it.

\- \*\*A1\*\* — how much of the result comes from searching the

&#x20; configuration space at all, versus the features and objective alone.

\- \*\*A2\*\* — does explanation-aware selection buy anything over

&#x20; conventional accuracy-driven selection. \*\*This is the single most

&#x20; important ablation in the study\*\* — the direct test of the

&#x20; CANXplain premise, and the arm closest to ACHILLES-style selection.

\- \*\*A3\*\* — isolated from the complexity penalty, what does explanation

&#x20; stability (β) contribute on its own. A2 removes two things at once;

&#x20; A3 removes only one, so A2-vs-A3 attributes the effect correctly.

\- \*\*A4\*\* — does complexity-aware selection (γ) actually restrain model

&#x20; size, and at what cost in detection performance.

\- \*\*A5\*\* — the honest floor: what a conventional practitioner pipeline

&#x20; produces on the same data and splits. If A0 cannot beat A5, the

&#x20; framework has not justified itself.

\- \*\*A6\*\* — how much of the performance comes from the proposed

&#x20; vendor-agnostic features rather than the model machinery.

\- \*\*A7\*\* — do timing features alone generalise across vehicles. If A7

&#x20; holds up cross-dataset it is the strongest argument for the

&#x20; vendor-agnostic premise.

\- \*\*A8\*\* — do CAN-ID frequency statistics alone generalise. Compared

&#x20; with A7, shows which feature family carries cross-dataset transfer.



\---



\## 2. Same-dataset results (mean over 3 seeds and 3 source datasets)



From `results/full/SUMMARY.txt`:



| Arm | F1 | Accuracy | SHAP consistency | Complexity |

|---|---|---|---|---|

| A0 | 0.8884 | 0.8575 | 0.9166 | 2704.3333 |

| A1 | 0.9109 | 0.8907 | 0.9207 | 414347.4444 |

| A2 | 0.8950 | 0.8662 | 0.8899 | 312947.0000 |

| A3 | 0.8688 | 0.8112 | 0.8280 | 473.8889 |

| A4 | 0.9043 | 0.8739 | 0.9280 | 390935.4444 |

| A5 | 0.9109 | 0.8907 | 0.9207 | 414347.4444 |

| A6 | 0.8296 | 0.7536 | 1.0000\* | 1426.1111 |

| A7 | 0.8708 | 0.8170 | 0.8241 | 2333.5556 |

| A8 | 0.8188 | 0.7354 | 0.8955 | 324.3333 |



\\\* A6's SHAP consistency is \*\*not comparable\*\* to the other arms — see

§5. A1 and A5 are numerically identical, which is expected: both skip

the search entirely and evaluate the same fixed default candidates, so

the presence or absence of the β/γ terms in the (unused) fitness

expression makes no difference to which model is selected.



\---



\## 3. Cross-dataset results — the generalisation result



Cross-dataset rows are the ones that bear on the generalisation claim;

same-dataset rows do not. Mean over 3 seeds and all dataset pairs

(including `can\_ids`, see §6 for why that dataset needs a caveat):



| Arm | F1 | Accuracy | ROC-AUC |

|---|---|---|---|

| A0 | 0.5683 | 0.5782 | 0.6233 |

| A1 | 0.6224 | 0.5912 | 0.6091 |

| A2 | 0.6324 | 0.5910 | 0.6128 |

| A3 | 0.5847 | 0.5798 | 0.6238 |

| A4 | 0.6233 | 0.5919 | 0.6054 |

| A5 | 0.6224 | 0.5912 | 0.6091 |

| A6 | 0.4825 | 0.5461 | 0.6082 |

| A7 | 0.5836 | 0.5800 | 0.6080 |

| A8 | 0.4920 | 0.5500 | 0.5937 |



\### Restricted to reliably-labelled datasets only (car\_hacking ↔ road; excludes can\_ids)



This is the more trustworthy subset, since `can\_ids`'s label quality

issue (§6) affects every arm roughly equally and can obscure real

differences:



| Arm | F1 | ROC-AUC |

|---|---|---|

| A8 | 0.4585 | 0.7146 |

| A2 | 0.5323 | 0.6239 |

| A6 | 0.4423 | 0.6193 |

| A4 | 0.5380 | 0.6084 |

| A5 | 0.5737 | 0.6063 |

| A1 | 0.5737 | 0.6063 |

| A0 | 0.5148 | 0.6039 |

| A7 | 0.5195 | 0.5684 |

| A3 | 0.5341 | 0.5141 |



\*\*A0 does not lead this table on either metric.\*\* On the car\_hacking↔road

subset, A0's F1 (0.5148) is below A5's (0.5737), and A0's ROC-AUC

(0.6039) is not distinguishable from A5's (0.6063).



\---



\## 4. Headline statistical result — A0 vs A5



Paired t-test on cross-dataset F1, matched by

(dataset\_train, dataset\_test, seed), restricted to car\_hacking↔road,

n = 6:



```

A0: \[0.2947, 0.5302, 0.4177, 0.5848, 0.6291, 0.6323]   mean 0.514

A5: \[0.5041, 0.5195, 0.5236, 0.6212, 0.6298, 0.6439]   mean 0.574

t = -1.703, p = 0.149   (NOT significant)

```



\*\*This study does not demonstrate a statistically significant

difference between A0 (full CANXplain) and A5 (the naive

accuracy-only, no-search floor) in cross-dataset generalisation, at

n = 3 seeds.\*\* The numeric trend favours A5, but n = 6 paired

observations is underpowered to confirm or rule out an effect of this

size in either direction (see §7). \*\*Do not report that CANXplain

improves cross-dataset generalisation over the naive baseline\*\* — that

claim is not supported by this data.



What the ablation study \*\*does\*\* support, independent of the A0-vs-A5

result:



\- \*\*The complexity penalty (γ) meaningfully reduces model size\*\* for

&#x20; comparable same-dataset accuracy. A0 averages complexity 2704

&#x20; same-dataset vs A1/A5's 414347 — roughly 150× smaller, for F1 0.888

&#x20; vs 0.911. This is the strongest, most defensible finding in the

&#x20; study.

\- \*\*The β and γ terms are not inert\*\* — A0, A3, and A5 select

&#x20; different model configurations (`config\_id`) in essentially every

&#x20; seed and source dataset checked, so the selection process is doing

&#x20; real work even where it does not demonstrably improve cross-dataset

&#x20; transfer at this sample size.



\---



\## 5. SHAP consistency: not comparable across feature-set arms (A6/A7/A8)



Spearman rank correlation over a very short feature vector is close to

meaningless: with 3 features (A6) there are only 6 possible rankings,

so two folds can agree perfectly by chance far more often than with

the full 23-feature vector. A6 reports `shap\_consistency: 1.0000` —

this is a scale artifact, not evidence of superior explanation

stability, and must not be read as A6 outperforming A0 on this metric.



The codebase enforces this distinction directly: every run record

carries `shap\_consistency\_n\_features` and a boolean

`shap\_consistency\_comparable` flag

(`MIN\_FEATURES\_FOR\_COMPARABILITY = 5` in

`canxplain/explain/consistency.py`). Table 8

(`table8\_shap\_consistency\_comparison`) marks reduced-feature rows with

`comparable\_to\_A0 = no`. A6/A7/A8's SHAP consistency values should be

reported, if at all, with this caveat attached — never presented

alongside A0's on the same scale without it.



\---



\## 6. `can\_ids` as source: a labelling-quality case study, not equal-weight evidence



Models trained on `can\_ids` and evaluated cross-dataset collapse to

at-or-below-chance performance, regardless of ablation arm:



\- A0/seed2, can\_ids→road: balanced accuracy 0.501, MCC 0.011,

&#x20; false-positive rate 98.7%

\- A6/seed2, can\_ids→road: false-positive rate exactly 1.0 (predicts

&#x20; "attack" on every window)

\- A5/seed2 (naive floor), can\_ids→road: balanced accuracy 0.492,

&#x20; MCC −0.024 — essentially the same collapse



This is not specific to CANXplain's selection process — the naive

floor (A5) fails identically. Root cause: three of `can\_ids`'s four

raw OTIDS `.txt` captures carry no per-row ground truth and are

labelled wholesale by filename, producing a flat 0.75 attack rate

across every split. A model trained on a 75%-attack distribution

learns a near-unconditional "predict attack" shortcut, which transfers

as an almost-universal false alarm against a dataset with a genuinely

different attack rate. See `README.md` §7 ("Label provenance") for the

mechanism.



ACHILLES (Mowla et al.) reports the same \*shape\* of failure without

diagnosing a cause: CAR-HACK→CAN-IDS accuracy 0.4467 (below chance),

ORNL→SAD accuracy 0.5048 with zero precision/recall, and CAN-IDS

flagged repeatedly as their weakest-generalising dataset. This work's

contribution is not the observation of the collapse but a mechanistic

explanation for it (coarse, file-level labelling) that the original

paper does not investigate.



\*\*Practical consequence for interpretation:\*\* treat any table or

statistic that includes `can\_ids` cross-dataset results as measuring,

in part, the effect of label quality rather than purely the effect of

model selection. The car\_hacking↔road subset (§3, §4) is the more

trustworthy basis for claims about generalisation.



\---



\## 7. Feature-family results on `road`: A7 vs A8



A8 (frequency-only, 7 features on road) collapses to a majority-class

classifier \*\*even same-dataset\*\* on `road`: `balanced\_accuracy: 0.500`,

`false\_positive\_rate: 1.0`, identically across all 3 seeds. This did

not happen on `car\_hacking` or `can\_ids`.



A7 (temporal-only) does not show the same collapse on `road`:

`balanced\_accuracy` ranges 0.510–0.725 across seeds, with 2 of 3 seeds

showing balanced\_accuracy > 0.70.



A plausible explanation: `road` contains masquerade attacks (payload

substitution on an existing periodic ID, no extra messages injected),

which do not perturb message frequency at all. A frequency-only

feature set has nothing to key on for this attack class; timing

features are affected because masquerade traffic still occupies its

usual inter-arrival slot but with altered payload semantics that can

correlate with surrounding traffic timing in ways frequency counts

cannot capture. This is a plausible mechanism, not independently

verified beyond the A7/A8 comparison itself.



\---



\## 8. Statistical power — read this before citing any p-value from this study



`n\_pairs` in `statistical\_tests.csv` is seeds × dataset pairs, not

seeds. At 3 seeds × 2 dataset pairs (car\_hacking↔road), n = 6, and

some comparisons reach p < 0.05 — but six observations from three

underlying seeds are not six independent replicates. The codebase's

own `paired\_test` function sets `sufficient\_power=False` below 5

seeds and attaches a note that such a result must not be reported as

a significance claim. Treat `sufficient\_power=True` at n=3 seeds as

"the test executed," not as "this is a publishable significance

result." Five or more seeds is the stated minimum for anything

intended as a significance claim; this study used 3.



\---



\## 9. Summary of defensible claims from this ablation study



\*\*Supported by the data:\*\*

\- The complexity penalty (γ) substantially reduces selected model size

&#x20; without a corresponding loss in same-dataset accuracy (§4).

\- The SHAP-consistency (β) and complexity (γ) terms are not inert —

&#x20; they change which model gets selected, consistently, across seeds

&#x20; and source datasets.

\- `can\_ids`'s coarse, file-level labelling produces catastrophic,

&#x20; arm-independent cross-dataset failure — a data-quality effect, not a

&#x20; selection-method effect (§6).

\- On `road`, frequency-only features carry near-zero signal while

&#x20; temporal features retain some; this traces plausibly to the

&#x20; dataset's masquerade attack composition (§7).



\*\*Not supported by the data — do not claim these:\*\*

\- That CANXplain (A0) achieves statistically significantly better

&#x20; cross-dataset generalisation than the naive accuracy-only floor

&#x20; (A5). The paired test does not support this at n = 3 seeds (§4).

\- That any arm's SHAP consistency value is comparable across different

&#x20; feature-set sizes (§5).

\- That the default fitness weights (α=0.5, β=0.3, γ=0.2) are derived

&#x20; or optimal — the sensitivity sweep shows them sitting unremarkably

&#x20; within the grid, not at an identified optimum.

