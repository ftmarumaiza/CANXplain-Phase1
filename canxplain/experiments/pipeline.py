"""The Phase 1 pipeline.

    raw CAN log
      -> adapter                      (standard 4-column stream)
      -> chronological split w/ purge  (train | val | test)
      -> sliding-window segmentation
      -> vendor-agnostic features
      -> StandardScaler fitted on TRAIN ONLY
      -> model selection (meta-learning, or fixed defaults)
      -> refit on train+val
      -> same-dataset test evaluation
      -> cross-dataset evaluation on completely unseen datasets

LEAKAGE CONTROL -- the invariants this module enforces
------------------------------------------------------
1. The scaler is fitted on the TRAIN block only and then reused
   unchanged for val, test and every cross-dataset target.
2. Model selection (search, fitness, SHAP consistency) sees the TRAIN
   block only, through blocked purged CV.
3. The val block is held out during selection and used as an honest
   check that the search did not overfit its own CV folds.
4. The test block is touched exactly once, at final evaluation.
5. A cross-dataset target is never loaded until after the source model
   is frozen. It contributes nothing to features, scaling, selection or
   thresholds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.adapters import load_dataset
from ..data.windowing import (
    SplitBlock,
    chronological_split,
    describe_stream,
    split_window_starts,
)
from ..evaluation.metrics import compute_metrics, measure_inference_time
from ..explain.shap_utils import shap_importance, shap_importance_frame, subsample
from ..features.extractor import extract_features, resolve_feature_names
from ..meta.achilles import AchillesSearch, SearchResult, select_without_search
from ..meta.evaluator import CandidateEvaluator
from ..meta.fitness import FitnessSpec, normalise_complexity
from ..meta.search_space import complexity_reference_bounds, space_from_config
from ..models.registry import CandidateModel, default_candidate
from ..utils import ensure_dir, get_logger, set_seed, timer

LOG = get_logger()


# --------------------------------------------------------------------------
@dataclass
class PreparedData:
    """Windowed, featurised, scaled views of one dataset."""
    name: str
    feature_names: List[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    scaler: StandardScaler
    blocks: Dict[str, List[Any]]
    stream_stats: Dict[str, Any] = field(default_factory=dict)

    def split_record(self) -> Dict[str, Any]:
        return {
            "split_strategy": "chronological_contiguous_with_purge",
            "split_train_windows": int(len(self.X_train)),
            "split_val_windows": int(len(self.X_val)),
            "split_test_windows": int(len(self.X_test)),
            "split_train_attack_rate": float(np.mean(self.y_train)),
            "split_val_attack_rate": float(np.mean(self.y_val)),
            "split_test_attack_rate": float(np.mean(self.y_test)),
            "split_n_captures": len(self.blocks.get("train", [])),
            "split_blocks": {name: [b.as_dict() for b in blocks]
                             for name, blocks in self.blocks.items()},
        }


def _scale(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def prepare_dataset(
    cfg,
    dataset_name: str,
    feature_groups: List[str],
    fit_scaler: bool = True,
) -> PreparedData:
    """Load, split, featurise and scale one dataset."""
    spec = cfg.data.datasets[dataset_name].to_dict()
    df = load_dataset(dataset_name, spec, cache_dir=cfg.data.get("cache_dir"))
    stats = describe_stream(df, dataset_name)

    window_size = int(cfg.data.window_size)
    step = int(cfg.data.get("step", 1))
    caps = cfg.data.get("max_windows", {})
    caps = caps.to_dict() if hasattr(caps, "to_dict") else dict(caps or {})

    blocks = chronological_split(
        df, window_size,
        train=float(cfg.data.split.train),
        val=float(cfg.data.split.val),
        test=float(cfg.data.split.test),
    )

    featured = {}
    for split_name, split_blocks in blocks.items():
        starts = split_window_starts(split_blocks, window_size, step,
                                     max_windows=caps.get(split_name))
        if len(starts) == 0:
            raise ValueError(f"{dataset_name}/{split_name}: zero windows produced")
        X_split, y_split = extract_features(df, starts, window_size, feature_groups)
        LOG.info("%s/%s: %d windows, attack rate %.4f",
                 dataset_name, split_name, len(X_split), float(y_split.mean()))
        featured[split_name] = (X_split, y_split)

    X_train, y_train = featured["train"]
    X_val, y_val = featured["val"]
    X_test, y_test = featured["test"]

    # Invariant 1: the scaler only ever sees the training block.
    scaler = StandardScaler().fit(X_train)

    return PreparedData(
        name=dataset_name,
        feature_names=list(X_train.columns),
        X_train=_scale(scaler, X_train), y_train=y_train,
        X_val=_scale(scaler, X_val), y_val=y_val,
        X_test=_scale(scaler, X_test), y_test=y_test,
        scaler=scaler, blocks=blocks, stream_stats=stats,
    )


def prepare_target_dataset(
    cfg,
    dataset_name: str,
    feature_groups: List[str],
    source_scaler: StandardScaler,
    source_feature_names: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Featurise a cross-dataset TARGET, entirely as held-out test data.

    The whole target stream is windowed -- there is no train portion,
    because the target is never trained on. Features are aligned to the
    source's column order and transformed with the SOURCE scaler, since
    at deployment time a vehicle's own statistics are not available to
    refit normalisation.
    """
    spec = cfg.data.datasets[dataset_name].to_dict()
    df = load_dataset(dataset_name, spec, cache_dir=cfg.data.get("cache_dir"))
    stats = describe_stream(df, dataset_name)

    window_size = int(cfg.data.window_size)
    step = int(cfg.data.get("step", 1))
    caps = cfg.data.get("max_windows", {})
    caps = caps.to_dict() if hasattr(caps, "to_dict") else dict(caps or {})

    from ..data.windowing import capture_ranges
    whole = [SplitBlock("target", lo, hi, capture_id)
             for capture_id, lo, hi in capture_ranges(df)]
    starts = split_window_starts(
        whole, window_size, step,
        max_windows=caps.get("cross_target", caps.get("test")))
    X, y = extract_features(df, starts, window_size, feature_groups)

    missing = [c for c in source_feature_names if c not in X.columns]
    if missing:
        LOG.warning("target '%s' lacks %s; filled with 0.0 after scaling",
                    dataset_name, missing)
        for column in missing:
            X[column] = 0.0
    X = X[source_feature_names]

    X_scaled = pd.DataFrame(source_scaler.transform(X), columns=X.columns)
    return X_scaled, y, stats


# --------------------------------------------------------------------------
@dataclass
class ArmResult:
    """Everything one (arm, dataset, seed) run produced."""
    arm_code: str
    dataset: str
    seed: int
    search: SearchResult
    final_model: CandidateModel
    same_dataset_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    cross_dataset_metrics: Dict[str, Dict[str, float]]
    shap_importance: pd.DataFrame
    final_complexity: float
    final_complexity_norm: float
    training_time_s: float
    records: List[Dict[str, Any]] = field(default_factory=list)


def run_arm(
    cfg,
    arm,
    dataset_name: str,
    seed: int,
    tracker,
    cross_targets: Optional[List[str]] = None,
    prepared_cache: Optional[Dict[str, PreparedData]] = None,
) -> ArmResult:
    """Run one ablation arm on one source dataset with one seed."""
    set_seed(seed)
    default_groups = list(cfg.features.groups)
    groups = arm.groups(default_groups)

    cache_key = f"{dataset_name}|{','.join(groups)}"
    if prepared_cache is not None and cache_key in prepared_cache:
        data = prepared_cache[cache_key]
    else:
        data = prepare_dataset(cfg, dataset_name, groups)
        if prepared_cache is not None:
            prepared_cache[cache_key] = data

    base_spec = FitnessSpec.from_config(cfg.fitness)
    spec = arm.spec(base_spec)

    space = space_from_config(cfg)
    bounds = complexity_reference_bounds(space, len(data.feature_names))
    backend = cfg.models.dnn.get("backend", "sklearn")

    evaluator = CandidateEvaluator(
        cfg, data.X_train, data.y_train, bounds, seed=seed,
        data_tag=f"{dataset_name}|{','.join(groups)}",
    )

    # ---- model selection (train block only) --------------------------
    LOG.info("[%s] %s on %s (seed %d) | objective: %s",
             arm.code, arm.name, dataset_name, seed, spec.describe())

    if arm.use_meta_learning:
        search = AchillesSearch(
            space=space, evaluator=evaluator, spec=spec,
            population_size=int(cfg.meta.population),
            rounds=int(cfg.meta.rounds),
            survivors=int(cfg.meta.survivors),
            epsilon=float(cfg.meta.epsilon),
            seed=seed, families=list(cfg.meta.get("families", ["rf", "dnn"])),
            backend=backend,
        ).run()
    else:
        defaults = [default_candidate(family, cfg, seed=seed)
                    for family in cfg.meta.get("families", ["rf", "dnn"])]
        search = select_without_search(evaluator, defaults, spec)

    selected = search.best_candidate

    # ---- refit the selected configuration on train + val --------------
    # Both blocks precede test in time, so this uses more data without
    # touching the test set. The scaler is NOT refitted.
    refit_on = cfg.get("run", {}).get("refit_on", "train_val")
    if refit_on == "train_val":
        X_final = pd.concat([data.X_train, data.X_val], ignore_index=True)
        y_final = np.concatenate([data.y_train, data.y_val])
    else:
        X_final, y_final = data.X_train, data.y_train

    final_model = selected.clone(seed=seed)
    with timer() as elapsed:
        final_model.fit(X_final, y_final)
    final_train_time = elapsed()

    final_complexity = final_model.complexity()
    final_complexity_norm = normalise_complexity(
        final_complexity, final_model.family, bounds)

    # ---- evaluation ---------------------------------------------------
    val_metrics = compute_metrics(
        data.y_val, final_model.predict(data.X_val),
        final_model.positive_score(data.X_val))

    same_metrics = compute_metrics(
        data.y_test, final_model.predict(data.X_test),
        final_model.positive_score(data.X_test))
    same_metrics.update(measure_inference_time(final_model, data.X_test))

    # Final SHAP importance is computed on VALIDATION data, not test.
    shap_frame = pd.DataFrame()
    try:
        X_explain = subsample(data.X_val, int(cfg.meta.shap.get("max_explain_sample", 200)),
                              seed=seed)
        X_background = subsample(data.X_train, int(cfg.meta.shap.get("max_background", 100)),
                                 seed=seed)
        importance = shap_importance(
            final_model, X_explain, X_background,
            max_evals=int(cfg.meta.shap.get("dnn_max_evals", 200)), seed=seed)
        shap_frame = shap_importance_frame(importance, data.feature_names)
    except Exception as exc:
        LOG.warning("final SHAP importance failed for %s: %s", arm.code, exc)

    evaluation = search.best_evaluation

    # Arms that do not SELECT on SHAP consistency (A2, A3, A5) skip SHAP
    # during the search to save compute, which leaves their consistency
    # field NaN. Measure it once here, on the selected configuration
    # only, so Table 8 can compare every arm on the same quantity. This
    # is a reporting step: it uses train-derived CV folds exactly as the
    # search does and never touches the test set, and it cannot change
    # which model was selected because selection has already happened.
    if not np.isfinite(float(evaluation.consistency.score)):
        try:
            evaluation = evaluator.evaluate(selected, need_shap=True)
        except Exception as exc:
            LOG.warning("post-hoc consistency failed for %s: %s", arm.code, exc)

    base_record = {
        **arm.as_record(),
        **spec.as_record(),
        **data.split_record(),
        "dataset_train": dataset_name,
        "seed": seed,
        "feature_groups": ",".join(groups),
        "n_features": len(data.feature_names),
        "model_family": final_model.family,
        "config_id": final_model.config_id,
        **{f"hp_{k}": v for k, v in final_model.params.items()},
        "complexity": final_complexity,
        "complexity_norm": final_complexity_norm,
        "fitness": search.best_fitness,
        "shap_consistency": float(evaluation.consistency.score),
        **{k: v for k, v in evaluation.consistency.to_record().items()
           if k != "shap_consistency"},
        "cv_accuracy": evaluation.accuracy,
        "search_time_s": search.search_time_s,
        "search_evaluations": search.n_evaluations,
        "training_time_s": final_train_time,
        "window_size": int(cfg.data.window_size),
        "window_step": int(cfg.data.get("step", 1)),
    }

    records = []
    records.append(tracker.log({
        **base_record, "dataset_test": dataset_name,
        "evaluation_type": "same_dataset", **same_metrics,
    }))
    records.append(tracker.log({
        **base_record, "dataset_test": dataset_name,
        "evaluation_type": "validation_holdout", **val_metrics,
    }))

    # ---- cross-dataset (target stays unseen until this point) ---------
    cross_metrics: Dict[str, Dict[str, float]] = {}
    for target in (cross_targets or []):
        if target == dataset_name:
            continue
        try:
            X_t, y_t, _ = prepare_target_dataset(
                cfg, target, groups, data.scaler, data.feature_names)
        except Exception as exc:
            LOG.warning("cross-dataset target '%s' unavailable: %s", target, exc)
            continue

        metrics = compute_metrics(y_t, final_model.predict(X_t),
                                  final_model.positive_score(X_t))
        metrics.update(measure_inference_time(final_model, X_t))
        cross_metrics[target] = metrics
        records.append(tracker.log({
            **base_record, "dataset_test": target,
            "evaluation_type": "cross_dataset",
            "cross_target_windows": int(len(X_t)),
            "cross_target_attack_rate": float(np.mean(y_t)),
            **metrics,
        }))

    LOG.info("[%s] %s/seed%d -> %s | same-dataset F1 %.4f | consistency %.4f",
             arm.code, dataset_name, seed, final_model.family,
             same_metrics["f1"], float(evaluation.consistency.score))

    return ArmResult(
        arm_code=arm.code, dataset=dataset_name, seed=seed, search=search,
        final_model=final_model, same_dataset_metrics=same_metrics,
        val_metrics=val_metrics, cross_dataset_metrics=cross_metrics,
        shap_importance=shap_frame, final_complexity=final_complexity,
        final_complexity_norm=final_complexity_norm,
        training_time_s=final_train_time, records=records,
    )


def save_model_bundle(result: ArmResult, data: PreparedData, out_dir: str) -> str:
    """Persist the model, the scaler and the feature contract together.

    Saving them separately is how a deployment ends up feeding a model
    features in the wrong order six months later. One bundle, one file.
    """
    ensure_dir(out_dir)
    path = os.path.join(
        out_dir, f"{result.arm_code}_{result.dataset}_seed{result.seed}.joblib")
    joblib.dump({
        "model": result.final_model.estimator,
        "family": result.final_model.family,
        "hyperparameters": result.final_model.params,
        "scaler": data.scaler,
        "feature_names": data.feature_names,
        "window_size": None,
        "arm": result.arm_code,
        "dataset": result.dataset,
        "seed": result.seed,
        "note": ("Apply scaler.transform to features in feature_names order, "
                 "then model.predict. Features must be built by "
                 "canxplain.features.extractor with the same groups."),
    }, path, compress=3)
    return path
