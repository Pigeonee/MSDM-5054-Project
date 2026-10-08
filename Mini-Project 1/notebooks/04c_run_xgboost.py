"""Run XGBoost by itself on Project 1.zip, one visible step at a time.

This teaching run writes to outputs/xgboost_standalone. It does not overwrite
the five-model comparison in outputs/project1.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from home_credit.prepared_data import load_prepared
from home_credit.xgboost_model import make_xgboost


PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_MEMBER = "Project 1/data/raw/sample_submission.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone XGBoost walkthrough")
    parser.add_argument("--data-zip", type=Path, default=PROJECT_DIR.parent / "Project 1.zip")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / ".cache_project1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "xgboost_standalone")
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.trees < 1 or args.n_jobs < 1:
        parser.error("--trees and --n-jobs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read the prepared numeric features. The loader excludes ID, TARGET,
    #    FOLD from X and checks/drops the 232 duplicate _y feature columns.
    data = load_prepared(args.data_zip, args.cache_dir)
    print(f"Step 1: {len(data.targets):,} labeled rows; {len(data.feature_names)} features.", flush=True)

    # 2. Keep fold 0 completely unseen during this first fit. TARGET is the
    #    answer (0=no default, 1=default); FOLD only identifies the split.
    fit_rows = np.flatnonzero(data.folds != args.validation_fold)
    val_rows = np.flatnonzero(data.folds == args.validation_fold)
    if not len(fit_rows) or not len(val_rows):
        raise ValueError("The selected validation fold leaves an empty split")
    y_fit = data.targets[fit_rows]
    y_val = data.targets[val_rows]
    if set(np.unique(y_fit)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Both splits must contain TARGET=0 and TARGET=1")
    x_fit = data.train_features[fit_rows]
    x_val = data.train_features[val_rows]
    print(f"Step 2: fit={len(fit_rows):,}; held-out validation={len(val_rows):,}.", flush=True)

    # 3. Create 250 small trees one after another (or --trees N). The factory
    #    contains every parameter so it can be read without other model code.
    model = make_xgboost(random_state=args.seed, n_jobs=args.n_jobs, n_estimators=args.trees)
    started = perf_counter()
    model.fit(x_fit, y_fit)
    fit_seconds = perf_counter() - started
    print(f"Step 3: trained {args.trees} boosting trees in {fit_seconds:.1f}s.", flush=True)

    # 4. predict_proba returns two columns: [P(TARGET=0), P(TARGET=1)].
    #    Kaggle requires the second column, a probability between 0 and 1.
    val_probability = model.predict_proba(x_val)[:, 1]
    auc = float(roc_auc_score(y_val, val_probability))
    brier = float(brier_score_loss(y_val, val_probability))
    pd.DataFrame({
        "SK_ID_CURR": data.train_ids[val_rows],
        "TARGET": y_val,
        "predicted_probability": val_probability,
    }).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    print(f"Step 4: held-out ROC-AUC={auc:.6f}; Brier={brier:.6f}.", flush=True)
    del model, x_fit, x_val
    gc.collect()

    # 5. Validation is finished. Start a fresh model with the SAME settings
    #    and fit it on all labeled rows to make the final Kaggle submission.
    final_model = make_xgboost(
        random_state=args.seed, n_jobs=args.n_jobs, n_estimators=args.trees,
    )
    final_model.fit(data.train_features, data.targets)
    final_model.save_model(args.output_dir / "model.json")
    del final_model
    gc.collect()
    print("Step 5: refitted all labeled rows; saved model.json.", flush=True)

    # 6. Load the saved model and predict the unknown TARGET for test rows.
    restored = make_xgboost(n_jobs=args.n_jobs)
    restored.load_model(args.output_dir / "model.json")
    test_probability = restored.predict_proba(data.test_features)[:, 1]
    if (len(test_probability) != len(data.test_ids)
            or not np.isfinite(test_probability).all()
            or np.any((test_probability < 0) | (test_probability > 1))):
        raise ValueError("Invalid test probabilities")
    with ZipFile(args.data_zip) as archive, archive.open(SAMPLE_MEMBER) as source:
        sample = pd.read_csv(source)
    if list(sample.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Unexpected sample submission columns")
    predictions = pd.DataFrame({"SK_ID_CURR": data.test_ids, "TARGET": test_probability})
    submission = sample[["SK_ID_CURR"]].merge(
        predictions, on="SK_ID_CURR", how="left", sort=False, validate="one_to_one",
    )
    if len(submission) != len(data.test_ids) or submission["TARGET"].isna().any():
        raise ValueError("Test IDs do not match the sample submission")
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    print(f"Step 6: saved {len(submission):,} test probabilities.", flush=True)

    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model": "XGBoost", "train_rows": len(data.targets),
            "validation_fold": args.validation_fold,
            "validation_rows": len(val_rows),
            "feature_count": len(data.feature_names),
            "feature_names": list(data.feature_names),
            "trees": args.trees, "seed": args.seed, "n_jobs": args.n_jobs,
            "validation_roc_auc": auc, "validation_brier": brier,
            "validation_fit_seconds": round(fit_seconds, 2),
            "data_zip": str(args.data_zip),
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
