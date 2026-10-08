"""XGBoost model settings for the prepared Home Credit dataset.

The caller passes numeric features only: SK_ID_CURR, TARGET, and FOLD must be
kept outside the input matrix. XGBoost accepts the remaining NaN values.
"""

from __future__ import annotations

from typing import Any


def make_xgboost(
    *, random_state: int = 42, n_jobs: int = 4, **overrides: Any
) -> Any:
    """Create the binary classifier used in the five-model comparison.

    Overrides let a quick check use fewer trees without changing the standard
    250-tree configuration. The label is TARGET: 0 or 1.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required; install xgboost in the pytorch-study "
            "environment before training this model."
        ) from exc

    params: dict[str, Any] = {
        "objective": "binary:logistic",  # Binary labels; output P(TARGET=1).
        "eval_metric": "auc",             # Metric known to XGBoost; externally we also calculate ROC-AUC.
        "tree_method": "hist",            # Bin values before considering split points.
        "n_estimators": 250,              # Add 250 small trees, one after another.
        "learning_rate": 0.05,           # Scale down each new tree's contribution.
        "max_depth": 4,                 # Limit interactions learned by one tree.
        "min_child_weight": 20,         # Avoid creating small, unstable leaves.
        "subsample": 0.85,              # Each tree sees a random 85% of training rows.
        "colsample_bytree": 0.85,        # Each tree sees a random 85% of features.
        "reg_lambda": 2.0,              # L2 regularization on leaf scores.
        "max_bin": 128,                 # Number of histogram bins per feature.
        "random_state": random_state,   # Reproducible random sampling.
        "n_jobs": n_jobs,               # CPU workers.
        "verbosity": 0,
    }
    params.update(overrides)
    return XGBClassifier(**params)
