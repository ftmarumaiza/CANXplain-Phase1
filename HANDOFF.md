# CANXplain Phase 1 — Handoff

Phase 1 = data/feature pipeline + ACHILLES-inspired meta-learning +
explanation-aware model selection + ablation study.
No counterfactual generation. No DBC decoding. No payload bytes.

Everything in this archive runs on CPU. It was developed and validated on
an i3-class machine with no GPU.

---

## 1. Current implementation status

**Phase 1 is COMPLETE.** `configs/full.yaml` ran to completion: 3 seeds
(0, 1, 2), 3 source datasets (car_hacking, can_ids, road), all 9
ablation arms, plus the 8-point fitness-weight sensitivity sweep.
348 total run records in `results/full/runs.jsonl`.

| Component | Status |
|---|---|
| Data adapters (Car-Hacking, CAN-IDS/OTIDS, ORNL/ROAD, SAD, generic CSV) | Implemented. Car-Hacking, CAN-IDS, ROAD all exercised with real data. |
| ROAD conversion (`scripts/convert_road.py`) | Implemented — converts candump `.log` + `capture_metadata.json` to labelled CSV, skips the 4 captures with no CAN-level ground truth |
| Schema normalisation + capture tracking | Implemented, in use |
| Per-capture chronological split with purge gap | Implemented, verified by `scripts/check_leakage.py` on all three real datasets |
| Window/feature extraction (23 vendor-agnostic features) | Implemented, in use |
| BlockedKFold (purged, never shuffled) | Implemented, used inside the search |
| SHAP consistency (Spearman/Kendall + Jaccard diagnostic) | Implemented, in use, comparability-flagged for short feature vectors |
| Fitness `α·Acc + β·Consistency − γ·Complexity` | Implemented, weights swept in the sensitivity stage |
| ACHILLES-inspired population search | Implemented, re-derived from the paper text |
| Ablation arms A0–A8 | **All 9 run to completion**, 3 seeds, 3 source datasets |
| Cross-dataset evaluation | **Complete** — every arm evaluated against both other datasets |
| Fitness-weight sensitivity sweep | **Complete** — 8 weight points × 3 seeds = 24 runs, on car_hacking |
| Tables 1–9, Figures 1–8 | Auto-generated in `results/full/` |
| Experiment tracking (`runs.csv` / `runs.jsonl`) | Complete, 348 records |
| Leakage/fairness assertions | All pass on all three real datasets |
| README.md | **NOT WRITTEN** — this file is still the interim substitute |
| `docs/ABLATION.md` as a static file | **NOT WRITTEN** — generated into `results/full/ABLATION.md` at runtime |

Real data was used throughout the final run: Car-Hacking and CAN-IDS
downloaded from their official sources, ROAD converted from raw
candump logs via `scripts/convert_road.py`. The synthetic generator
(`scripts/make_synthetic_data.py`) was only used for early smoke
testing and is not the source of any reported number.

---

## 2. Completed experiments

Both under `results/`:

**`results/debug/`** — `run_all -c configs/debug.yaml`, 1 seed, ~1.5 min.
57 run records. All nine arms, both datasets, three evaluation types
(`validation_holdout`, `same_dataset`, `cross_dataset`), plus the
sensitivity sweep. 24 table files, 8 figures.

**`results/debug_3seed/`** — `run_ablation` with `run.seeds=[0,1,2]`.
162 run records. This is the run that exercises the multi-seed path:
mean ± SD in Table 6, error bars in the figures, paired t-test and
Wilcoxon with `sufficient_power=True`.

Leakage checks pass on both datasets: purge gap ≥ `window_size − 1` at
every cut, disjoint (capture, message) spans across splits, scaler fitted
on train only and frozen for val/test/cross-target, matching feature
column order between source and target.

---

## 3. Known findings (final, from the complete run)

### 3.0 THE HEADLINE RESULT: A0 does not significantly beat the naive floor (A5)

On the two reliably per-frame-labelled datasets (car_hacking, road;
`can_ids` excluded here, see 3.2), paired t-test on cross-dataset F1,
matched by (dataset_train, dataset_test, seed), n=6:

```
A0: [0.2947, 0.5302, 0.4177, 0.5848, 0.6291, 0.6323]   mean 0.514
A5: [0.5041, 0.5195, 0.5236, 0.6212, 0.6298, 0.6439]   mean 0.574
t = -1.703, p = 0.149   (not significant)
```

**Honest conclusion for the thesis:** explanation-aware,
complexity-penalised selection does NOT demonstrate a statistically
significant improvement in cross-dataset generalisation over a
conventional accuracy-only pipeline, at n=3 seeds. The numeric trend
favours A5, but the study is underpowered (n=6 paired observations) to
confirm or rule out an effect of this size in either direction. Do not
claim CANXplain "improves" generalisation — the honest claim is that
selection strategy did not measurably change cross-dataset transfer at
this sample size, while other benefits (below) hold.

### 3.0b What DOES hold up, across the full run

- **Complexity efficiency, confirmed twice.** In the ablation study,
  A0's models average `complexity: 2704` same-dataset vs A1/A5's
  `414347` — roughly 150x smaller for comparable same-dataset F1
  (0.888 vs 0.911). The sensitivity sweep independently confirms this:
  turning γ off entirely (α=1, β=0, γ=0) causes selected complexity to
  jump to 28,106 from a stable ~400–2,500 everywhere γ>0. **This is
  the strongest, most defensible finding in the study.**
- **Selection terms are not inert.** A0≠A3≠A5 in actual `config_id`
  selected, consistently, across all three source datasets and all
  seeds checked. The sensitivity sweep shows 15 distinct
  configurations selected across only 8 weight points. The debug-scale
  "identical" finding (superseded, see 3.1) never held once real
  search budget and real data were used.
- **Default weights (0.5/0.3/0.2) are unremarkable within the grid** —
  the sensitivity table shows them sitting in the middle of the
  accuracy/complexity trade-off, not specially optimal. Report the
  full sensitivity table alongside the default; do not present
  0.5/0.3/0.2 as derived.

### 3.0c can_ids: a genuine, well-evidenced negative case study

Models trained on `can_ids` and evaluated cross-dataset collapse to
at-or-below-chance performance — e.g. A0/seed2 can_ids→road:
balanced accuracy 0.501, MCC 0.011, false-positive rate 98.7%; A6/seed2
can_ids→road: FPR 1.0 exactly (predicts "attack" on every window). This
is NOT specific to CANXplain's selection — A5 (the naive floor) shows
the same collapse (balanced accuracy 0.492, MCC -0.024). Root cause:
`can_ids`'s attack rate is a flat 0.75 across train/val/test because
three of its four raw OTIDS `.txt` captures have no per-row ground
truth and are labelled wholesale by filename (see 3.2 below and
section 7's "Label provenance" trap). A model trained on a 75%-attack
distribution learns a "predict-attack" shortcut that transfers as an
almost-universal false alarm on datasets with a different, genuine
attack rate.

**ACHILLES itself (Mowla et al.) reports the same *shape* of failure**
without diagnosing the cause: CAR-HACK→CAN-IDS accuracy 0.4467 (below
chance), ORNL→SAD accuracy 0.5048 with zero precision/recall, and
CAN-IDS flagged repeatedly as their weakest-generalising dataset. Citing
this paper's own generalisation tables (their Table II-IV) alongside
your can_ids diagnosis is a legitimate point of novelty: ACHILLES
observes the phenomenon, this thesis explains a mechanistic cause
(coarse, file-level labelling) that the original paper never
investigates.

### 3.0d road: frequency-only features carry near-zero signal

A8 (frequency-only, 7 features) collapses to a majority-class
classifier EVEN SAME-DATASET on road (balanced_accuracy 0.5, FPR 1.0,
precision equals road's base attack rate exactly). This did not happen
on car_hacking or can_ids. Plausible cause: road's masquerade attacks
(payload-only, no extra messages injected) don't perturb message
frequency at all, so a frequency-only feature set has nothing to key
on. Worth checking A7 (temporal-only) against this directly — if A7
does not collapse the same way, that isolates which feature family
carries road's detectable signal.

### 3.1 A3 vs A0, A4 vs A2 — the debug-scale "identical" finding did NOT replicate

Superseded by 3.0b above; kept here for the debugging history only.
At debug scale (tiny search budget), A3 was bit-identical to A0 in
every seed, suggesting the β term was inert. This never held once the
full search budget and real data were used — A0 and A3 (and A2/A4)
selected different `config_id`s in every seed checked, across all
three source datasets. The debug-scale finding was a search-budget
artifact of `configs/debug.yaml`, not a property of the method.

### 3.2 Consistency is not comparable across feature-set arms

A6 (3 features) scores a perfect 1.0000 SHAP consistency. Spearman over
a 3-element rank vector has only six possible orderings, so perfect
agreement is near-free. A6/A7/A8 change the feature vector length and
therefore the scale of the metric. Table 8 now carries `n_features` and
a `comparable_to_A0` flag; rows marked `no` must not be read as
like-for-like. Threshold is `MIN_FEATURES_FOR_COMPARABILITY = 5` in
`canxplain/explain/consistency.py`.

### 3.3 Statistical power

`n_pairs` in `statistical_tests.csv` is seeds × dataset pairs, not
seeds. Each row now also reports `n_seeds`, `n_dataset_pairs` and
`pairing_unit`. At 3 seeds × 2 dataset pairs the tests run and some
reach p < 0.05, but those six observations are not six independent
replicates. Treat `sufficient_power=True` as "the test executed", not as
"this is a publishable significance claim". Use ≥ 5 seeds for anything
going in the thesis.

### 3.4 Transfer gap is the headline quantity

Same-dataset F1 sits well above cross-dataset F1 in every arm. The gap,
not the same-dataset number, is what a generalisable IDS has to shrink.
`SUMMARY.txt` reports it directly.

### 3.5 Bugs fixed during validation (for context, all already applied)

1. `ExperimentTracker.flush()` wrote only the current process's records,
   truncating `runs.csv` when a later stage ran. Now rebuilt from the
   complete `runs.jsonl`.
2. `--set` used `nargs="*"` with no append action, so a second `--set`
   silently discarded the first — a "3-seed" run quietly ran on one
   seed. Now `action="extend"`.
3. `n_seeds` in the tables was the group size (seeds × datasets). Now
   `n_runs` and a true `n_seeds`.
4. No comparability guard on short feature vectors (see 3.2).
5. Stats rows did not disclose the composition of their pairs (see 3.3).

---

## 4. Remaining tasks

1. **Write `README.md`**: install, reproduction commands, DEBUG vs
   FULL mode, label provenance (can_ids caveat + ROAD conversion
   caveat), ACHILLES-vs-CANXplain separation table, dataset links,
   pointer to the final results in `results/full/`.
2. **Write `docs/ABLATION.md`** statically — currently only generated
   at runtime; see `results/full/ABLATION.md` for the generated
   version to base it on.
3. **Write the thesis results/discussion section** using section 3
   above as the factual backbone: lead with the honest null result
   (3.0), the complexity-efficiency finding (3.0b), the can_ids
   negative case study with the ACHILLES citation (3.0c), and the
   road frequency-collapse finding (3.0d) as a limitations/future-work
   point.
4. **Pick the Phase 2 model.** Use the `config_id` ranking method
   (rank A0 across (dataset_train, seed) by mean cross-dataset
   roc_auc) — prefer `car_hacking` or `road` as source, not `can_ids`
   given 3.0c. Load via
   `joblib.load('results/full/models/A0_<dataset>_seed<N>.joblib')`.
5. Optional: confirm A7 vs A8 on `road` (does temporal-only avoid the
   frequency-only collapse from 3.0d?) — quick query, not yet run.
6. Optional: rerun with more seeds if time allows, to move the
   A0-vs-A5 comparison out of "underpowered" territory (would need ~5
   seeds minimum, ideally more).

---

## 5. Exact commands

```bash
# 0. environment
pip install -r requirements.txt

# 1. synthetic smoke-test data (skip once real data is in place)
python scripts/make_synthetic_data.py --out data/raw --messages 40000

# 2. fairness / leakage assertions — run this after any data change
python scripts/check_leakage.py -c configs/debug.yaml

# 3. fast end-to-end check (~1.5 min, 1 seed)
python -m canxplain.experiments.run_all -c configs/debug.yaml

# 4. THE FULL EXPERIMENT (3 seeds, pop 12, 6 rounds, 500k messages)
python -m canxplain.experiments.run_all -c configs/full.yaml
```

Individual stages, if the full run needs to be split up:

```bash
python -m canxplain.experiments.run_pipeline      -c configs/full.yaml
python -m canxplain.experiments.run_metalearning  -c configs/full.yaml
python -m canxplain.experiments.run_ablation      -c configs/full.yaml
python -m canxplain.experiments.run_crossdataset  -c configs/full.yaml
python -m canxplain.experiments.run_sensitivity   -c configs/full.yaml
```

Useful overrides (`--set` is repeatable):

```bash
# more seeds — do this for anything with a p-value
python -m canxplain.experiments.run_ablation -c configs/full.yaml \
    --set run.seeds=[0,1,2,3,4]

# a subset of arms
python -m canxplain.experiments.run_ablation -c configs/full.yaml --arms A0 A2

# separate output directory
python -m canxplain.experiments.run_ablation -c configs/full.yaml \
    --set run.output_dir=results/full_5seed

# bigger search budget
python -m canxplain.experiments.run_ablation -c configs/full.yaml \
    --set meta.rounds=10 --set meta.population=20
```

Outputs land under `run.output_dir`: `runs.csv`, `runs.jsonl`,
`resolved_config.json`, `environment.json`, `SUMMARY.txt`,
`ABLATION.md`, `tables/`, `figures/`, `models/`, `artifacts/`.

`runs.csv` is the source of truth — every table in the thesis should be
derivable from it with a groupby, without re-running anything.

---

## 6. Expected dataset locations

Paths come from `configs/base.yaml` under `data.datasets`. Defaults:

```
data/raw/car_hacking/          # Car-Hacking
data/raw/can_ids/              # CAN-IDS (OTIDS)
data/raw/can_ids_logformat/    # optional, exercises the OTIDS .txt log parser
data/cache/                    # parquet cache, auto-created, safe to delete
```

`data/cache/` is not in this archive — it regenerates on first load.
Delete it whenever the raw data changes, or the old cache will be reused.

To point at data living elsewhere, edit the `path` field in
`configs/base.yaml` or override it:

```bash
--set data.datasets.car_hacking.path=/some/other/place
```

---

## 7. Adding the real Car-Hacking and CAN-IDS data

**Downloads**

- Car-Hacking: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
- CAN-IDS (OTIDS): https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
- ORNL ROAD (later): https://0xsam.com/road/
- SAD (later): https://ocslab.hksecurity.net/Datasets/survival-ids

**Steps**

1. Delete or move the synthetic captures — do not mix them with real
   data:
   ```bash
   rm -rf data/raw data/cache
   ```
2. Unpack Car-Hacking so the per-attack CSVs sit directly in
   `data/raw/car_hacking/` (`DoS_dataset.csv`, `Fuzzy_dataset.csv`,
   `gear_dataset.csv`, `RPM_dataset.csv`, `normal_run_data.txt`). Each
   file is one capture; the filename becomes the capture name.
3. Unpack CAN-IDS into `data/raw/can_ids/` the same way
   (`Attack_free_dataset.txt`, `DoS_dataset.csv`, `Fuzzy_dataset.csv`,
   `Impersonation_attack_dataset.csv`).
4. Raise the message caps — the defaults are smoke-test sized:
   ```bash
   --set data.datasets.car_hacking.max_messages=2000000 \
   --set data.datasets.can_ids.max_messages=2000000
   ```
   or edit `configs/full.yaml`.
5. Re-run the leakage checks and the table-1 statistics first:
   ```bash
   python scripts/check_leakage.py -c configs/full.yaml
   python -m canxplain.experiments.run_pipeline -c configs/full.yaml
   ```
   Check Table 1 before going further: attack rate must be strictly
   between 0 and 1 in **every** split of **every** dataset. A rate of
   0.0000 or 1.0000 means a single-class split and every downstream
   number is meaningless.

**Two traps that already bit once**

- *Headerless CSVs.* Car-Hacking rows carry the DLC data bytes inline,
  so a DLC-8 frame has more fields than a DLC-4 frame. pandas infers the
  column count from the first row and drops the wider ones — which are
  exactly the injected attack frames. `data/adapters.py` pre-scans for
  the maximum field count and passes explicit column names. If you write
  a new adapter, do the same.
- *Label provenance.* Do not label a capture by its filename. If every
  row in `DoS_dataset.csv` is labelled "attack", the model learns to
  identify the capture, not the attack, and cross-dataset numbers become
  fiction. Use the per-row label column. The OTIDS `.txt` logs have no
  per-row labels, which is why they are parsed but not used as a
  labelled source.

---

## 8. ACHILLES vs CANXplain — what is whose

Full statement in the `canxplain/meta/achilles.py` docstring. Summary:

**From ACHILLES (Mowla et al., IEEE TITS 27(7), 2026):** population of
meta-agents, competition rounds, pairwise ranking by fitness, the
evolution operator (Q-table exploitation plus winner crossover with
probability 1−ε, random exploration otherwise), the elimination stage,
final selection.

**CANXplain's own contribution:** the fitness function itself (ACHILLES
competes on accuracy/precision); SHAP consistency as a first-class
selection quantity rather than a post-hoc report; blocked and purged
cross-validation inside the search loop.

This is a re-implementation from the published description. It is not
the authors' code and it does not reproduce their results.

---

## 9. Honest-reporting guardrails already in the code

Do not remove these when writing up:

- `paired_test` sets `sufficient_power=False` below 5 pairs and attaches
  a note saying the p-value must not be reported as significance.
- Plots start bars at zero, draw error bars only when seeds > 1, label
  `n_seeds` on the axis, and never annotate "better" or "significant".
- `SUMMARY.txt` states explicitly that it makes no claim that CANXplain
  improves over its ablations.
- Table 8 flags consistency values that are not comparable to A0.
- The default weights (α=0.50, β=0.30, γ=0.20) are documented as
  defended defaults, not derived optima. Always report Table 9 (the
  sensitivity sweep) alongside them.