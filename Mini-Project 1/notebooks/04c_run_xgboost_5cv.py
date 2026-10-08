"""Run descriptive five-fold CV for the already selected XGBoost settings.

The selected settings came from this same dataset. The five-fold scores below
are useful for describing fold-to-fold variation, but are not an independent
estimate of performance after hyperparameter selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_auc_score,
)

from home_credit.csv_data import _source_record, load_prepared_csv
from home_credit.xgboost_model import make_xgboost


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "xgboost_final"
EXPECTED_FOLDS = np.arange(5)
NOTE = (
    "Hyperparameters were selected using this same dataset before this five-fold run. "
    "These post-selection cross-validation scores describe fold variation; they are "
    "not an independent test estimate. Kaggle test labels are unavailable."
)
PARAMETER_NAMES = (
    "objective", "eval_metric", "tree_method", "n_estimators", "learning_rate",
    "max_depth", "min_child_weight", "subsample", "colsample_bytree",
    "reg_lambda", "max_bin", "random_state", "n_jobs", "verbosity",
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_signature(train_csv: Path, test_csv: Path) -> str:
    train = _source_record(train_csv)
    test = _source_record(test_csv)
    return (
        f"csv|{train['size']}:{train['sha256']}|"
        f"{test['size']}:{test['sha256']}"
    )


def _valid_probabilities(probability: np.ndarray, expected: int) -> bool:
    return (
        probability.ndim == 1 and len(probability) == expected
        and np.isfinite(probability).all()
        and not np.any((probability < 0) | (probability > 1))
    )


def _reuse_fold0_predictions(
    output_dir: Path, *, source_signature: str, config: dict, seed: int,
    feature_count: int, train_ids: np.ndarray, targets: np.ndarray,
    fit_rows: np.ndarray, val_rows: np.ndarray,
) -> np.ndarray | None:
    """Reuse old FOLD=0 probabilities only when data, labels and config agree."""
    try:
        context = json.loads((output_dir / "search_context.json").read_text(encoding="utf-8"))
        outer = json.loads((output_dir / "outer_comparison.json").read_text(encoding="utf-8"))
        metrics = json.loads((output_dir / "xgboost_validation_metrics.json").read_text(encoding="utf-8"))
        prediction = pd.read_csv(output_dir / "validation_predictions.csv")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    if (context.get("source_signature") != source_signature
            or context.get("feature_count") != feature_count
            or context.get("comparison_fold") != 0
            or context.get("seed") != seed
            or outer.get("config") != config
            or metrics.get("validation_fold") != 0
            or metrics.get("n_features") != feature_count
            or metrics.get("n_train") != len(fit_rows)
            or metrics.get("n_validation") != len(val_rows)):
        return None
    params = metrics.get("parameters", {})
    if not isinstance(params, dict) or params.get("random_state") != seed:
        return None
    if any(params.get(key) != value for key, value in config.items()):
        return None
    if list(prediction.columns) != ["SK_ID_CURR", "TARGET", "predicted_probability"]:
        return None
    if (not np.array_equal(prediction["SK_ID_CURR"].to_numpy(), train_ids[val_rows])
            or not np.array_equal(prediction["TARGET"].to_numpy(), targets[val_rows])):
        return None
    probability = prediction["predicted_probability"].to_numpy(dtype=np.float64)
    if not _valid_probabilities(probability, len(val_rows)):
        return None
    auc = float(roc_auc_score(targets[val_rows], probability))
    ap = float(average_precision_score(targets[val_rows], probability))
    if (not np.isclose(auc, float(outer["tuned_auc"]), rtol=0, atol=1e-8)
            or not np.isclose(auc, float(metrics["roc_auc"]), rtol=0, atol=1e-8)
            or not np.isclose(ap, float(metrics["average_precision"]), rtol=0, atol=1e-8)):
        return None
    return probability


def _report_text(summary: dict, report_path: Path, output_dir: Path) -> str:
    def relative(name: str) -> str:
        return Path(os.path.relpath(output_dir / name, report_path.parent)).as_posix()

    rows = [
        f"| {item['fold']} | {item['n_train']:,} | {item['n_validation']:,} | "
        f"{item['positive_rate']:.4%} | {item['roc_auc']:.6f} | "
        f"{item['average_precision']:.6f} | {item['pr_auc_trapezoid']:.6f} | "
        f"{item['prediction_source']} |"
        for item in summary["per_fold_details"]
    ]
    return "\n".join([
        "# XGBoost five-fold validation", "",
        f"This run used **{summary['n_features']} encoded features** from the new "
        "`final_train_encoded.csv`/`final_test_encoded.csv` tables. For each fold, "
        "the model was fitted on the other four folds and scored on the held-out fold.",
        "", "| Fold | Train rows | Validation rows | Positive rate | ROC-AUC | Average precision | Trapezoidal PR-AUC | Prediction source |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *rows, "",
        f"Mean ROC-AUC: **{summary['mean_roc_auc']:.6f} ± {summary['std_roc_auc']:.6f}** "
        "(sample standard deviation, ddof=1).",
        f"Mean average precision: **{summary['mean_average_precision']:.6f} ± "
        f"{summary['std_average_precision']:.6f}** (sample standard deviation, ddof=1).",
        f"Mean trapezoidal PR-AUC: **{summary['mean_pr_auc_trapezoid']:.6f} ± "
        f"{summary['std_pr_auc_trapezoid']:.6f}** (sample standard deviation, ddof=1).",
        "", "## Interpretation", "", NOTE, "",
        "Average precision integrates the precision-recall curve with a stepwise "
        "rule. Trapezoidal PR-AUC interpolates linearly between curve points, so "
        "the values can differ. The random-forest reference reports average "
        "precision, making that column the direct comparison.", "",
        "The mean is the unweighted average of the five fold scores. Each training "
        "example has one out-of-fold probability; the test CSV has no labels, so no "
        "test AUC is calculated here.", "",
        f"- [Machine-readable summary]({relative('xgboost_5cv_summary.json')})",
        f"- [Out-of-fold predictions]({relative('xgboost_5cv_oof_predictions.csv')})",
        f"- [Selected settings]({relative('selection.json')})", "",
    ])


def run_five_fold(
    train_csv: Path, test_csv: Path, cache_dir: Path, output_dir: Path,
    report_path: Path, *, seed: int = 42, n_jobs: int = 4,
    reuse_fold0: bool = True,
) -> dict:
    """Run all five fixed-parameter fits and write OOF, summary and report."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    train_csv, test_csv = Path(train_csv), Path(test_csv)
    output_dir, report_path = Path(output_dir), Path(report_path)
    selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    config = selection["config"]
    if not isinstance(config, dict) or int(config.get("n_estimators", 0)) < 1:
        raise ValueError("selection.json must contain a fixed positive n_estimators")
    source_signature = _source_signature(train_csv, test_csv)
    data = load_prepared_csv(train_csv, test_csv, cache_dir)
    if not np.array_equal(np.unique(data.folds), EXPECTED_FOLDS):
        raise ValueError("This report requires exactly FOLD=0,1,2,3,4")
    model_params = make_xgboost(random_state=seed, n_jobs=n_jobs, **config).get_params()
    parameters = {key: model_params[key] for key in PARAMETER_NAMES}
    oof = np.full(len(data.targets), np.nan, dtype=np.float64)
    details: list[dict] = []

    for fold in EXPECTED_FOLDS:
        fit_rows = np.flatnonzero(data.folds != fold)
        val_rows = np.flatnonzero(data.folds == fold)
        y_fit, y_val = data.targets[fit_rows], data.targets[val_rows]
        if set(np.unique(y_fit)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
            raise ValueError(f"FOLD={fold} or its training complement lacks a TARGET class")
        started = perf_counter()
        probability = None
        source = "new training"
        if fold == 0 and reuse_fold0:
            probability = _reuse_fold0_predictions(
                output_dir, source_signature=source_signature, config=config,
                seed=seed, feature_count=len(data.feature_names),
                train_ids=data.train_ids, targets=data.targets,
                fit_rows=fit_rows, val_rows=val_rows,
            )
            if probability is not None:
                source = "verified saved FOLD=0 prediction"
        if probability is None:
            x_fit = data.train_features[fit_rows]
            x_val = data.train_features[val_rows]
            model = make_xgboost(random_state=seed, n_jobs=n_jobs, **config)
            model.fit(x_fit, y_fit)
            probability = model.predict_proba(x_val)[:, 1]
            del model, x_fit, x_val
            gc.collect()
        if not _valid_probabilities(np.asarray(probability), len(val_rows)):
            raise ValueError(f"Invalid probabilities for FOLD={fold}")
        oof[val_rows] = probability
        precision, recall, _ = precision_recall_curve(y_val, probability)
        detail = {
            "fold": int(fold),
            "n_train": int(len(fit_rows)),
            "n_validation": int(len(val_rows)),
            "positive_rate": float(np.mean(y_val)),
            "roc_auc": float(roc_auc_score(y_val, probability)),
            "average_precision": float(average_precision_score(y_val, probability)),
            "pr_auc_trapezoid": float(auc(recall, precision)),
            "prediction_source": source,
            "seconds": round(perf_counter() - started, 2),
        }
        details.append(detail)
        print(
            f"FOLD={fold}: AUC={detail['roc_auc']:.6f}, "
            f"AP={detail['average_precision']:.6f} ({source})", flush=True,
        )

    if not _valid_probabilities(oof, len(data.targets)):
        raise ValueError("Out-of-fold predictions are incomplete")
    aucs = np.asarray([item["roc_auc"] for item in details])
    aps = np.asarray([item["average_precision"] for item in details])
    pr_aucs = np.asarray([item["pr_auc_trapezoid"] for item in details])
    summary = {
        "n_folds": 5,
        "mean_roc_auc": float(np.mean(aucs)),
        "std_roc_auc": float(np.std(aucs, ddof=1)),
        "mean_average_precision": float(np.mean(aps)),
        "std_average_precision": float(np.std(aps, ddof=1)),
        "mean_pr_auc_trapezoid": float(np.mean(pr_aucs)),
        "std_pr_auc_trapezoid": float(np.std(pr_aucs, ddof=1)),
        "per_fold_details": details,
        "model": "XGBClassifier",
        "parameters": parameters,
        "n_features": len(data.feature_names),
        "source_signature": source_signature,
        "provenance_note": NOTE,
    }
    _write_csv(
        output_dir / "xgboost_5cv_oof_predictions.csv",
        pd.DataFrame({
            "SK_ID_CURR": data.train_ids, "TARGET": data.targets,
            "FOLD": data.folds, "predicted_probability": oof,
        }),
    )
    _write_json(output_dir / "xgboost_5cv_summary.json", summary)
    _write_text(report_path, _report_text(summary, report_path, output_dir))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=PROJECT_DIR.parent / "final_train_encoded.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_DIR.parent / "final_test_encoded.csv")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / ".cache_final_csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=PROJECT_DIR / "reports" / "xgboost_5cv_report.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--no-reuse-fold0", action="store_true")
    args = parser.parse_args()
    run_five_fold(
        args.train_csv, args.test_csv, args.cache_dir, args.output_dir, args.report,
        seed=args.seed, n_jobs=args.n_jobs, reuse_fold0=not args.no_reuse_fold0,
    )


if __name__ == "__main__":
    main()
