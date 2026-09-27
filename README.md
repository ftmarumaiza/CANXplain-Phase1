\# CANXplain — Phase 1



Data/feature pipeline, ACHILLES-inspired meta-learning, explanation-aware

model selection, and ablation study for CAN-bus intrusion detection.

CPU-only. No counterfactual generation, no DBC decoding, no payload

bytes anywhere in this phase.



For experimental findings, results, and known issues, see

\[`HANDOFF.md`](HANDOFF.md). This file covers setup and reproduction only.



\---



\## 1. Requirements



\- Python 3.10+

\- CPU only — no GPU required or used

\- \~2 GB disk for real datasets (car\_hacking + can\_ids + road)



```bash

pip install -r requirements.txt

```



Optional, only needed for Markdown table output in some scripts:

```bash

pip install tabulate

```



\---



\## 2. Project layout



```

canxplain/                  # the package

&#x20; data/                      adapters, schema, windowing

&#x20; features/                  feature extraction

&#x20; models/                    RF / DNN registry

&#x20; explain/                   SHAP + consistency scoring

&#x20; meta/                      ACHILLES-inspired search, fitness

&#x20; evaluation/                metrics, tables, plots, tracking

&#x20; experiments/               entry-point scripts (run\_\*.py)

configs/

&#x20; base.yaml                  shared defaults

&#x20; debug.yaml                 fast smoke-test config (1 seed, tiny data)

&#x20; full.yaml                  the reported-experiment config

scripts/

&#x20; make\_synthetic\_data.py     synthetic data for smoke-testing ONLY

&#x20; check\_leakage.py           fairness/leakage assertions — run after any data change

&#x20; convert\_road.py            converts raw ROAD candump logs to labelled CSV

data/raw/<dataset>/          expected location for each dataset's files

data/cache/                  parquet cache, auto-created, safe to delete

results/<output\_dir>/        runs.csv, runs.jsonl, tables/, figures/, models/

```



\---



\## 3. Dataset setup



Paths are set under `data.datasets` in `configs/base.yaml` (inherited

by `debug.yaml` and `full.yaml`). Override any path with `--set

data.datasets.<name>.path=...` instead of editing the file, if

preferred.



\### Car-Hacking



Download: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset



Place the per-attack CSVs directly in `data/raw/car\_hacking/`:

`DoS\_dataset.csv`, `Fuzzy\_dataset.csv`, `gear\_dataset.csv`,

`RPM\_dataset.csv`, `normal\_run\_data.txt`. Each file is treated as one

capture.



\### CAN-IDS (OTIDS)



Download: https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset



Place files in `data/raw/can\_ids/`. \*\*Warning:\*\* the standard OTIDS

release ships as raw `.txt` candump-style logs with \*\*no per-row

labels\*\* — three of the four files get labelled wholesale by filename

(`DoS\_attack\_dataset.txt` → every row "attack"). This produces a flat

0.75 attack rate across every split and, empirically, causes

catastrophic cross-dataset transfer failure (see HANDOFF.md §3.0c).

Use this dataset with that caveat, or substitute a labelled-CSV source

if one becomes available.



\### ROAD (ORNL)



Download: https://0xsam.com/road/



ROAD ships as raw SocketCAN candump logs (`ambient/\*.log`,

`attacks/\*.log`) plus `capture\_metadata.json` with per-capture

injection intervals — not directly usable by the `ornl` adapter, which

expects labelled CSVs. Convert first:



```bash

python scripts/convert\_road.py --road-root /path/to/road --out data/raw/road\_csv

```



This parses `(timestamp) can0 ID#DATA` lines, rebases timestamps to

capture start, and labels each frame from the metadata's

`injection\_interval` (given in seconds elapsed since capture start,

not absolute time). Ambient captures are labelled entirely normal.

\*\*Four captures are skipped automatically\*\*

(`accelerator\_attack\_drive\_1/2`, `accelerator\_attack\_reverse\_1/2`) —

their metadata has `injection\_interval: null` because they are

physical/sensor-level attacks with no CAN-frame-level ground truth.

The script reports this; it is expected, not an error.



Then point the config at the converted output:

```bash

\--set data.datasets.road.adapter=ornl --set data.datasets.road.path=data/raw/road\_csv

```

(`configs/full.yaml` already has this wired in if you're using the

provided config.)



\### SAD



Download: https://ocslab.hksecurity.net/Datasets/survival-ids



Adapter (`sad`) exists but was not exercised in the reported run.

Adding it is a config entry under `data.datasets`, no pipeline change.



\---



\## 4. Verify before running anything



```bash

python scripts/check\_leakage.py -c configs/full.yaml

```



Run this after adding or changing any dataset. It asserts: purge gap

≥ `window\_size - 1` at every split boundary, train/val/test drawn from

disjoint (capture, message) spans, the scaler is fitted on train only

and stays frozen when transforming val/test/cross-dataset targets, and

feature columns match order between source and target. All checks

must print `PASS` — if anything fails, do not trust results downstream

of it.



Also check Table 1 (`tables/table1\_dataset\_statistics.\*`) after

`run\_pipeline` — attack rate must be strictly between 0 and 1 in every

split of every dataset. A rate of exactly 0.0 or 1.0 means a

single-class split.



\---



\## 5. Running



\### Fast smoke test (\~1–2 min, uses synthetic data)



```bash

python scripts/make\_synthetic\_data.py --out data/raw --messages 40000

python -m canxplain.experiments.run\_all -c configs/debug.yaml

```



\### The reported experiment



```bash

python -m canxplain.experiments.run\_all -c configs/full.yaml --set run.seeds=\[0,1,2]

```



Runs all three stages in order: data/feature pipeline (Tables 1–2,

Figure 1), the ablation study (all 9 arms × seeds × source datasets,

same-dataset and cross-dataset evaluation, Tables 3–8, statistical

tests), then the fitness-weight sensitivity sweep.



Individual stages, if you need to split the run across sessions:



```bash

python -m canxplain.experiments.run\_pipeline      -c configs/full.yaml

python -m canxplain.experiments.run\_metalearning  -c configs/full.yaml

python -m canxplain.experiments.run\_ablation      -c configs/full.yaml

python -m canxplain.experiments.run\_crossdataset  -c configs/full.yaml

python -m canxplain.experiments.run\_sensitivity   -c configs/full.yaml

```



\*\*Resuming after an interruption:\*\* `results/<output\_dir>/runs.jsonl`

is append-only and safe — a killed process never corrupts or loses

already-completed runs. To resume without re-running finished

combinations, scope the next invocation to only what's missing:



```bash

python -c "

import pandas as pd

d = pd.read\_json('results/full/runs.jsonl', lines=True)

d = d\[d.evaluation\_type=='validation\_holdout']

print(d.groupby(\['dataset\_train','arm'])\['seed'].apply(lambda s: sorted(s.unique())))

"

```



then target just the missing piece, e.g.:

```bash

python -m canxplain.experiments.run\_ablation -c configs/full.yaml \\

&#x20;   --set run.seeds=\[2] --set run.source\_datasets=\[car\_hacking] \\

&#x20;   --arms A8 --set run.output\_dir=results/full

```



Useful overrides (`--set` is repeatable, do not repeat `--set` for the

same run and expect only the last one to apply on older builds — this

codebase supports repeated `--set`):



```bash

\--set run.seeds=\[0,1,2,3,4]              # more seeds

\--arms A0 A2                              # subset of ablation arms

\--set run.output\_dir=results/custom       # separate output directory

\--set meta.rounds=10 --set meta.population=20   # bigger search budget

```



\### DEBUG vs FULL



`configs/debug.yaml`: 1 seed, tiny search budget, small `max\_windows`

caps — for verifying the pipeline runs end-to-end in a couple of

minutes. Not for reporting numbers.



`configs/full.yaml`: the configuration actually reported. Real search

budget, full window counts, 3 datasets. On an Intel i3 (no GPU) with

all 9 arms × 3 seeds × 3 source datasets, expect the ablation stage

alone to take several hours; budget accordingly and keep the machine

from sleeping.



\---



\## 6. Outputs



Everything lands under `run.output\_dir` (default `results/full`):



\- `runs.csv` / `runs.jsonl` — every run record, one row per

&#x20; (arm, dataset, seed, evaluation\_type). `runs.jsonl` is the durable

&#x20; append-only log; `runs.csv` is rebuilt from it on each flush.

\- `resolved\_config.json`, `environment.json` — exact config and

&#x20; environment used, for reproducibility.

\- `SUMMARY.txt` — plain-text aggregate, explicitly does not claim

&#x20; CANXplain improves over its ablations; read Table 6 and

&#x20; `statistical\_tests.csv` and judge for yourself.

\- `tables/` — Tables 1–9 as CSV + Markdown.

\- `figures/` — Figures 1–8 as PNG.

\- `models/` — one `.joblib` bundle per (arm, dataset, seed), each

&#x20; containing the fitted model, scaler, and feature names:

&#x20; ```python

&#x20; import joblib

&#x20; bundle = joblib.load('results/full/models/A0\_car\_hacking\_seed1.joblib')

&#x20; model, scaler, feature\_names = bundle\['model'], bundle\['scaler'], bundle\['feature\_names']

&#x20; ```

\- `ABLATION.md` — generated documentation of each ablation arm's

&#x20; purpose and question.



\---



\## 7. Label provenance



Not every dataset carries genuine per-frame ground truth:



\- \*\*Car-Hacking\*\*: real per-row labels. Trustworthy.

\- \*\*ROAD\*\* (after `convert\_road.py`): real per-row labels derived from

&#x20; documented injection intervals, for captures where ROAD provides

&#x20; them. Trustworthy for the captures that convert; 4 captures are

&#x20; skipped for lacking ground truth (see §3 above).

\- \*\*CAN-IDS (OTIDS)\*\*: as distributed, whole-file labels only. Attack

&#x20; rate is a flat 0.75 across every split as a direct consequence.

&#x20; Treat any result trained on or evaluated against `can\_ids` with this

&#x20; caveat explicitly stated — see HANDOFF.md §3.0c for the measured

&#x20; effect.



Do not label a capture by its filename when adding a new dataset. If

every row in a file is labelled identically because the label came

from the filename rather than a real per-row column, the model will

learn to identify the capture instead of the attack, and cross-dataset

numbers become meaningless. `data/adapters.py` logs a warning whenever

this fallback path is used — do not ignore it.



\---



\## 8. ACHILLES vs CANXplain



This work uses Mowla et al., \*ACHILLES: A Machine Learning Framework

for Explainable and Generalized Automotive Intrusion Detection

System\*, IEEE TITS 27(7), 2026, as methodological reference for the

meta-learning stage — it is not reproduced and the code is not the

authors'.



\*\*From ACHILLES:\*\* population of meta-agents, competition rounds,

pairwise ranking by fitness, the evolution operator (Q-table

exploitation + winner crossover w.p. 1−ε, else random exploration),

elimination stage, final selection.



\*\*Not from ACHILLES — this project's own contribution:\*\* the fitness

function itself (ACHILLES competes on accuracy/precision only); SHAP

consistency as a first-class selection quantity rather than a post-hoc

report; blocked and purged cross-validation inside the search loop.



See `canxplain/meta/achilles.py` for the full docstring-level

separation.

