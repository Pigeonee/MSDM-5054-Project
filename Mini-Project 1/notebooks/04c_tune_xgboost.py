"""Stepwise XGBoost tuning with an untouched comparison fold.

FOLD=1 screens candidates while FOLD=0 is excluded. The best candidates are
confirmed on FOLD=2,3,4. Only then is the winner evaluated on FOLD=0 and
refitted on every labeled row. Progress is saved after each experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss, precision_recall_curve,
    roc_auc_score,
)

from home_credit.csv_data import load_prepared_csv
from home_credit.prepared_data import load_prepared
from home_credit.xgboost_model import make_xgboost


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP = PROJECT_DIR.parent / "Project 1.zip"
DEFAULT_CACHE = PROJECT_DIR / ".cache_project1"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "xgboost_tuned"
DEFAULT_REPORT = PROJECT_DIR / "reports" / "xgboost_tuning_report.md"
BASELINE_VALIDATION = PROJECT_DIR / "outputs" / "project1" / "xgboost" / "validation_predictions.csv"
SAMPLE_MEMBER = "Project 1/data/raw/sample_submission.csv"
ENCODED_MEMBERS = (
    "Project 1/data/processed/final_train_encoded.parquet",
    "Project 1/data/processed/final_test_encoded.parquet",
)
BASE_CONFIG = {
    "n_estimators": 250,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
}
STAGE_CHANGES = (
    ("学习率", (("lr_003", {"learning_rate": 0.03}),
               ("lr_008", {"learning_rate": 0.08}))),
    ("树复杂度", (("depth_3", {"max_depth": 3}),
                ("depth_5", {"max_depth": 5}),
                ("child_10", {"min_child_weight": 10}),
                ("child_40", {"min_child_weight": 40}))),
    ("抽样与正则", (("sample_075", {"subsample": 0.75, "colsample_bytree": 0.75}),
                 ("sample_100", {"subsample": 1.0, "colsample_bytree": 1.0}),
                 ("lambda_1", {"reg_lambda": 1.0}),
                 ("lambda_5", {"reg_lambda": 5.0}))),
)


def source_signature(zip_path: Path) -> str:
    with ZipFile(zip_path) as archive:
        parts = []
        for name in ENCODED_MEMBERS:
            member = archive.getinfo(name)
            parts.append(f"{name}:{member.file_size}:{member.CRC}")
    return "|".join(parts)


def input_signature(args) -> str:
    if args.train_csv is None:
        return source_signature(args.data_zip)
    parts = []
    for path in (args.train_csv, args.test_csv):
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        parts.append(f"{stat.st_size}:{digest.hexdigest()}")
    return "csv|" + "|".join(parts)


def scores(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability)
    if (len(labels) != len(probability) or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("Invalid predicted probability")
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
    }


def precision_recall_scores(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, probability)
    return {
        "average_precision": float(average_precision_score(labels, probability)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
    }


def save_csv(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def existing_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def ensure_context(args, data) -> None:
    path = args.output_dir / "search_context.json"
    current = {
        "source_signature": input_signature(args),
        "feature_count": len(data.feature_names),
        "fit_fold_for_screening": 1,
        "confirmation_folds": [2, 3, 4],
        "comparison_fold": 0,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "patience": args.patience,
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != current:
            raise ValueError("This output directory belongs to different data/settings")
    else:
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def make_model(config: dict, args, *, early_stop: bool):
    options = dict(config)
    if early_stop:
        options["n_estimators"] = args.max_rounds
        options["early_stopping_rounds"] = args.patience
    return make_xgboost(random_state=args.seed, n_jobs=args.n_jobs, **options)


def fit_and_score(config, args, x_fit, y_fit, x_val, y_val, *, early_stop):
    model = make_model(config, args, early_stop=early_stop)
    started = perf_counter()
    if early_stop:
        model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
        used_trees = int(model.best_iteration) + 1
    else:
        model.fit(x_fit, y_fit)
        used_trees = int(config["n_estimators"])
    seconds = perf_counter() - started
    probability = model.predict_proba(x_val)[:, 1]
    result = scores(y_val, probability)
    del model, probability
    gc.collect()
    final_config = dict(config)
    final_config["n_estimators"] = used_trees
    return final_config, result, round(seconds, 2)


def screening(args, data) -> None:
    path = args.output_dir / "search_trials.csv"
    records = existing_records(path)
    by_name = {record["trial"]: record for record in records}
    fit_rows = np.flatnonzero(np.isin(data.folds, [2, 3, 4]))
    val_rows = np.flatnonzero(data.folds == 1)
    x_fit = data.train_features[fit_rows]
    y_fit = data.targets[fit_rows]
    x_val = data.train_features[val_rows]
    y_val = data.targets[val_rows]
    print(f"Screening: fit={len(fit_rows):,}, validation={len(val_rows):,} (FOLD=1)", flush=True)

    def trial(name: str, stage: str, config: dict, early_stop: bool) -> dict:
        expected = dict(config)
        expected["n_estimators"] = args.max_rounds if early_stop else config["n_estimators"]
        expected_json = json.dumps(expected, sort_keys=True)
        if name in by_name:
            saved = by_name[name]
            if (saved["fit_config_json"] != expected_json
                    or bool(saved["early_stop"]) != early_stop):
                raise ValueError(f"Saved trial {name} has different parameters")
            print(f"Reuse {name}: AUC={saved['roc_auc']:.6f}", flush=True)
            return saved
        print(f"Start {stage} / {name}: {expected_json}", flush=True)
        final_config, result, seconds = fit_and_score(
            config, args, x_fit, y_fit, x_val, y_val, early_stop=early_stop,
        )
        row = {
            "trial": name, "stage": stage, "early_stop": early_stop,
            "fit_config_json": expected_json,
            "final_config_json": json.dumps(final_config, sort_keys=True),
            "trees": final_config["n_estimators"],
            **result, "fit_seconds": seconds,
        }
        records.append(row)
        by_name[name] = row
        save_csv(path, records)
        print(f"Done {name}: trees={row['trees']}, AUC={row['roc_auc']:.6f}, Brier={row['brier']:.6f}", flush=True)
        return row

    baseline = trial("baseline_250", "原配置", BASE_CONFIG, False)
    rounds = trial("rounds_early_stop", "树数与早停", BASE_CONFIG, True)
    incumbent = max((baseline, rounds), key=lambda row: row["roc_auc"])
    for stage, variations in STAGE_CHANGES:
        base = json.loads(incumbent["final_config_json"])
        contenders = [incumbent]
        for name, change in variations:
            candidate = {**base, **change}
            contenders.append(trial(name, stage, candidate, True))
        incumbent = max(contenders, key=lambda row: row["roc_auc"])
        print(f"After {stage}: current leader={incumbent['trial']} AUC={incumbent['roc_auc']:.6f}", flush=True)
    del x_fit, x_val
    gc.collect()
    print(f"Screening complete: {len(records)} trials in {path}", flush=True)


def confirmation(args, data) -> None:
    trials = existing_records(args.output_dir / "search_trials.csv")
    if len(trials) != 12:
        raise ValueError("Run the complete screening stage before confirmation")
    baseline = next(row for row in trials if row["trial"] == "baseline_250")
    ranked = sorted((row for row in trials if row["trial"] != "baseline_250"),
                    key=lambda row: row["roc_auc"], reverse=True)
    candidates = [baseline, *ranked[:2]]
    path = args.output_dir / "confirmation_folds.csv"
    records = existing_records(path)
    known = {(str(row["candidate"]), int(row["fold"])): row for row in records}
    for fold in (2, 3, 4):
        fit_rows = np.flatnonzero((data.folds != 0) & (data.folds != fold))
        val_rows = np.flatnonzero(data.folds == fold)
        x_fit = data.train_features[fit_rows]
        y_fit = data.targets[fit_rows]
        x_val = data.train_features[val_rows]
        y_val = data.targets[val_rows]
        print(f"Confirmation FOLD={fold}: fit={len(fit_rows):,}, validation={len(val_rows):,}", flush=True)
        for candidate in candidates:
            name = candidate["trial"]
            config = json.loads(candidate["final_config_json"])
            key = (name, fold)
            if key in known:
                if known[key]["config_json"] != json.dumps(config, sort_keys=True):
                    raise ValueError(f"Saved confirmation {key} has different parameters")
                print(f"Reuse {name} FOLD={fold}: AUC={known[key]['roc_auc']:.6f}", flush=True)
                continue
            print(f"Start {name} FOLD={fold} with fixed {config['n_estimators']} trees", flush=True)
            final_config, result, seconds = fit_and_score(
                config, args, x_fit, y_fit, x_val, y_val, early_stop=False,
            )
            row = {
                "candidate": name, "fold": fold,
                "config_json": json.dumps(final_config, sort_keys=True),
                "trees": config["n_estimators"], **result, "fit_seconds": seconds,
            }
            records.append(row)
            known[key] = row
            save_csv(path, records)
            print(f"Done {name} FOLD={fold}: AUC={row['roc_auc']:.6f}, Brier={row['brier']:.6f}", flush=True)
        del x_fit, x_val
        gc.collect()
    summary = []
    for candidate in candidates:
        rows = [row for row in records if row["candidate"] == candidate["trial"]]
        if {int(row["fold"]) for row in rows} != {2, 3, 4}:
            raise ValueError(f"Incomplete confirmation for {candidate['trial']}")
        summary.append({
            "candidate": candidate["trial"],
            "trees": int(candidate["trees"]),
            "mean_auc": float(np.mean([row["roc_auc"] for row in rows])),
            "sd_auc": float(np.std([row["roc_auc"] for row in rows], ddof=1)),
            "mean_brier": float(np.mean([row["brier"] for row in rows])),
            "mean_fit_seconds": float(np.mean([row["fit_seconds"] for row in rows])),
            "config_json": candidate["final_config_json"],
        })
    summary.sort(key=lambda row: (-row["mean_auc"], row["mean_brier"]))
    save_csv(args.output_dir / "confirmation_summary.csv", summary)
    winner = summary[0]
    selection = {
        "selected_candidate": winner["candidate"],
        "config": json.loads(winner["config_json"]),
        "criterion": "Highest mean ROC-AUC on confirmation FOLD=2,3,4; FOLD=0 unused",
        "mean_confirmation_auc": winner["mean_auc"],
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8",
    )
    print(pd.DataFrame(summary)[["candidate", "trees", "mean_auc", "sd_auc", "mean_brier"]].to_string(index=False), flush=True)
    print(f"Selected by inner confirmation: {winner['candidate']}", flush=True)


def final_evaluation(args, data) -> None:
    selection_path = args.output_dir / "selection.json"
    if not selection_path.is_file():
        raise ValueError("Run search and confirmation before final evaluation")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = selection["config"]
    fit_rows = np.flatnonzero(data.folds != 0)
    val_rows = np.flatnonzero(data.folds == 0)
    x_fit = data.train_features[fit_rows]
    y_fit = data.targets[fit_rows]
    x_val = data.train_features[val_rows]
    y_val = data.targets[val_rows]
    print(f"Locked FOLD=0 evaluation of {selection['selected_candidate']}...", flush=True)
    model = make_model(config, args, early_stop=False)
    started = perf_counter()
    model.fit(x_fit, y_fit)
    fit_seconds = perf_counter() - started
    probability = model.predict_proba(x_val)[:, 1]
    tuned = scores(y_val, probability)
    tuned_pr = precision_recall_scores(y_val, probability)
    parameter_names = (
        "objective", "eval_metric", "tree_method", "n_estimators",
        "learning_rate", "max_depth", "min_child_weight", "subsample",
        "colsample_bytree", "reg_lambda", "max_bin", "random_state",
        "n_jobs", "verbosity",
    )
    model_parameters = model.get_params()
    parameters = {name: model_parameters[name] for name in parameter_names}
    validation = pd.DataFrame({
        "SK_ID_CURR": data.train_ids[val_rows],
        "TARGET": y_val,
        "predicted_probability": probability,
    })
    validation.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    validation_metrics = {
        "validation_fold": 0,
        "n_train": len(fit_rows),
        "n_validation": len(val_rows),
        "positive_rate_validation": float(np.mean(y_val)),
        "roc_auc": tuned["roc_auc"],
        "average_precision": float(average_precision_score(y_val, probability)),
        "pr_auc_trapezoid": tuned_pr["pr_auc_trapezoid"],
        "brier_score": tuned["brier"],
        "n_features": len(data.feature_names),
        "model": "XGBClassifier",
        "selected_candidate": selection["selected_candidate"],
        "parameters": parameters,
    }
    (args.output_dir / "xgboost_validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )
    del model, probability
    gc.collect()

    if args.train_csv is not None:
        print("Training a fresh 250-tree baseline on the same CSV rows...", flush=True)
        baseline_model = make_model(BASE_CONFIG, args, early_stop=False)
        baseline_model.fit(x_fit, y_fit)
        baseline_probability = baseline_model.predict_proba(x_val)[:, 1]
        old = scores(y_val, baseline_probability)
        baseline_pr = precision_recall_scores(y_val, baseline_probability)
        pd.DataFrame({
            "SK_ID_CURR": data.train_ids[val_rows],
            "TARGET": y_val,
            "predicted_probability": baseline_probability,
        }).to_csv(args.output_dir / "baseline_validation_predictions.csv", index=False)
        del baseline_model, baseline_probability
    else:
        baseline = pd.read_csv(args.baseline_validation)
        if (not np.array_equal(baseline["SK_ID_CURR"].to_numpy(), validation["SK_ID_CURR"].to_numpy())
                or not np.array_equal(baseline["TARGET"].to_numpy(), y_val)):
            raise ValueError("Prior XGBoost baseline used a different validation set")
        old = scores(y_val, baseline["predicted_probability"].to_numpy())
        baseline_pr = precision_recall_scores(y_val, baseline["predicted_probability"].to_numpy())
    del x_fit, x_val
    gc.collect()
    outer = {
        "selected_candidate": selection["selected_candidate"],
        "config": config,
        "baseline_auc": old["roc_auc"], "baseline_brier": old["brier"],
        "tuned_auc": tuned["roc_auc"], "tuned_brier": tuned["brier"],
        "baseline_average_precision": baseline_pr["average_precision"],
        "baseline_pr_auc_trapezoid": baseline_pr["pr_auc_trapezoid"],
        "tuned_average_precision": tuned_pr["average_precision"],
        "tuned_pr_auc_trapezoid": tuned_pr["pr_auc_trapezoid"],
        "auc_delta": tuned["roc_auc"] - old["roc_auc"],
        "brier_delta": tuned["brier"] - old["brier"],
        "fit_seconds": round(fit_seconds, 2),
        "validation_rows": len(val_rows),
    }
    (args.output_dir / "outer_comparison.json").write_text(
        json.dumps(outer, indent=2), encoding="utf-8",
    )
    print(f"FOLD=0 AUC: old={old['roc_auc']:.6f}, tuned={tuned['roc_auc']:.6f}, delta={outer['auc_delta']:+.6f}", flush=True)

    # A new estimator, with the fixed inner-selected tree count and no
    # validation labels, is fitted on all 307,511 labeled rows.
    print("Refitting selected configuration on all labeled rows...", flush=True)
    final_model = make_model(config, args, early_stop=False)
    final_model.fit(data.train_features, data.targets)
    final_model.save_model(args.output_dir / "model.json")
    test_probability = final_model.predict_proba(data.test_features)[:, 1]
    del final_model
    gc.collect()
    if (len(test_probability) != len(data.test_ids)
            or not np.isfinite(test_probability).all()
            or np.any((test_probability < 0) | (test_probability > 1))):
        raise ValueError("Invalid test probabilities")
    predictions = pd.DataFrame({"SK_ID_CURR": data.test_ids, "TARGET": test_probability})
    if args.train_csv is not None:
        submission = predictions
    else:
        with ZipFile(args.data_zip) as archive, archive.open(SAMPLE_MEMBER) as source:
            sample = pd.read_csv(source)
        if list(sample.columns) != ["SK_ID_CURR", "TARGET"]:
            raise ValueError("Unexpected sample submission schema")
        submission = sample[["SK_ID_CURR"]].merge(
            predictions, on="SK_ID_CURR", how="left", sort=False, validate="one_to_one",
        )
    if len(submission) != len(data.test_ids) or submission["TARGET"].isna().any():
        raise ValueError("Submission IDs do not match the test table")
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    print(f"Saved {len(submission):,} test probabilities", flush=True)
    if args.train_csv is not None:
        write_csv_report(args, data, selection, outer)
    else:
        write_report(args, data, selection, outer)


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_report(args, data, selection: dict, outer: dict) -> None:
    trials = pd.read_csv(args.output_dir / "search_trials.csv")
    cv = pd.read_csv(args.output_dir / "confirmation_summary.csv")
    trial_rows = []
    for _, row in trials.iterrows():
        config = json.loads(row["final_config_json"])
        trial_rows.append([
            row["stage"], row["trial"], int(row["trees"]),
            config["learning_rate"], config["max_depth"], config["min_child_weight"],
            config["subsample"], config["colsample_bytree"], config["reg_lambda"],
            f"{row['roc_auc']:.6f}", f"{row['brier']:.6f}",
        ])
    cv_rows = [[row["candidate"], int(row["trees"]),
                f"{row['mean_auc']:.6f}", f"{row['sd_auc']:.6f}",
                f"{row['mean_brier']:.6f}"] for _, row in cv.iterrows()]
    config_rows = [[key, value] for key, value in selection["config"].items()]
    cap_note = (
        f"\n\n所选树数已接近本轮 {args.max_rounds} 棵的搜索上限；这只说明本轮搜索到此为止，"
        "不能把该树数解释为全局最佳轮数。"
        if selection["config"]["n_estimators"] >= args.max_rounds - 1 else ""
    )
    report = f"""# Home Credit XGBoost 逐步调参报告

## 目标与数据

在用户提供的 `Project 1.zip` 成品数据上，比较原始 XGBoost 配置与逐步调参后的配置。训练集 {len(data.targets):,} 行、测试集 {len(data.test_ids):,} 行；核对并删去 {len(data.dropped_duplicate_features)} 对完全重复的 `_y` 特征后，五模型共享的输入为 {len(data.feature_names)} 列。标签 `TARGET=1` 表示还款困难。评价指标以 ROC-AUC 为主（越高越好），Brier 分数辅助检查概率误差（越低越好）。

原始五模型对比中的 XGBoost 使用固定 250 棵树，`FOLD=0` ROC-AUC 为 **{outer['baseline_auc']:.6f}**，Brier 为 **{outer['baseline_brier']:.6f}**。这次所有调参决策只使用 `FOLD=1–4`；第 0 折仅在确定配置后作一次同口径比较。由于此前已看过第 0 折的基线成绩，它是锁定的比较折，而不能称为从未被团队看过的全新测试集。

## 步骤 1：在第 1 折逐项筛选

用 `FOLD=2,3,4` 训练、`FOLD=1` 验证。先重跑原始 250 棵树作内部参照，再让相同结构最多训练 {args.max_rounds} 棵树并设置 {args.patience} 轮早停；随后依次尝试学习率、树复杂度、抽样比例和正则强度。每一步只根据第 1 折 AUC 保留当前领先方案，下一步从该方案出发，因此试验名表示本次改动，并不代表其他参数恢复原值。早停的最佳轮次是 `best_iteration+1`；筛选结果不能直接当作独立测试成绩。

{markdown_table(['阶段', '试验', '树数', '学习率', '深度', '子节点权重', '样本比例', '特征比例', 'L2 正则', 'FOLD=1 AUC', 'Brier'], trial_rows)}

## 步骤 2：用第 2–4 折确认

把原始配置与筛选 AUC 最高的两个候选拿到 `FOLD=2,3,4` 分别验证；每次用剩余的三个内部折训练。此处树数已固定，**不在确认折上再次早停选树数**。按三折平均 AUC 选定配置，避免只依据第 1 折的一次波动。第 2–4 折曾作为步骤 1 的训练样本，因此这是跨折稳定性检查，并非完全独立的测试；第 0 折仍未参与任何调参。

{markdown_table(['候选', '固定树数', '平均 AUC', 'AUC 标准差', '平均 Brier'], cv_rows)}

选择 **{selection['selected_candidate']}**，依据是三折平均 ROC-AUC 最高。最终参数：

{markdown_table(['参数', '值'], config_rows)}{cap_note}

## 步骤 3：锁定配置后对比第 0 折

用 `FOLD=1–4` 训练所选配置，对 `FOLD=0` 的 {outer['validation_rows']:,} 人预测。原始模型 AUC **{outer['baseline_auc']:.6f}**，本次配置 AUC **{outer['tuned_auc']:.6f}**，差值 **{outer['auc_delta']:+.6f}**；Brier 从 **{outer['baseline_brier']:.6f}** 变为 **{outer['tuned_brier']:.6f}**（差值 {outer['brier_delta']:+.6f}，负值更好）。不依据第 0 折结果倒过来重新选择参数。

## 步骤 4：全量训练与交付

用锁定参数新建模型，在全部 {len(data.targets):,} 条有标签数据上重新训练，并为 {len(data.test_ids):,} 条测试数据输出 `P(TARGET=1)`。Kaggle 测试标签未公开，因此不能从提交文件计算测试 AUC。输出在 `outputs/xgboost_tuned/`：`model.json`、`validation_predictions.csv`、`submission.csv`、各阶段试验 CSV 和 `selection.json`。原始五模型结果仍在 `outputs/project1/`，此前独立 XGBoost 结果仍在 `outputs/xgboost_standalone/`。

这是一轮有边界的参数搜索，不保证全局最优；后续若继续优化，应另行记录更多配置或使用新的验证方案，避免重复利用锁定比较折。

参数含义及早停规则参考 [XGBoost 官方参数文档](https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html) 和 [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)。
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Report: {args.report}", flush=True)


def write_csv_report(args, data, selection: dict, outer: dict) -> None:
    """Describe the current direct-CSV run without mixing in old ZIP results."""
    trials = pd.read_csv(args.output_dir / "search_trials.csv")
    confirmation = pd.read_csv(args.output_dir / "confirmation_summary.csv")
    metrics = json.loads(
        (args.output_dir / "xgboost_validation_metrics.json").read_text(encoding="utf-8")
    )
    trial_rows = [
        [row["stage"], row["trial"], int(row["trees"]),
         f'{row["roc_auc"]:.6f}', f'{row["brier"]:.6f}']
        for _, row in trials.iterrows()
    ]
    confirmation_rows = [
        [row["candidate"], int(row["trees"]), f'{row["mean_auc"]:.6f}',
         f'{row["mean_brier"]:.6f}']
        for _, row in confirmation.iterrows()
    ]
    parameter_rows = [[name, value] for name, value in metrics["parameters"].items()]
    output_rel = Path(os.path.relpath(args.output_dir.resolve(), PROJECT_DIR)).as_posix()
    output_link = Path(os.path.relpath(args.output_dir.resolve(), args.report.parent.resolve())).as_posix()
    report = f"""# XGBoost tuning on the updated final CSV tables

## Data and evaluation

This run reads `{args.train_csv.name}` and `{args.test_csv.name}` directly. The input matrices have shapes **({len(data.targets):,}, {len(data.feature_names)})** and **({len(data.test_ids):,}, {len(data.feature_names)})**. `SK_ID_CURR`, `TARGET`, and `FOLD` are excluded from model features. The CSV feature names and order match between train and test. This is a fresh 511-feature run; scores from the earlier 508-feature ZIP run are not used as its baseline.

The parameter search uses FOLD=1 for screening and FOLD=2–4 for confirmation. FOLD=0 is excluded from parameter selection. A new 250-tree baseline and the selected model are both trained on the same {metrics["n_train"]:,} rows and evaluated on the same {metrics["n_validation"]:,} FOLD=0 rows. The FOLD=0 results are validation results, not a Kaggle test score.

## Screening trials

{markdown_table(["Stage", "Candidate", "Trees", "FOLD=1 ROC-AUC", "Brier"], trial_rows)}

## Confirmation

{markdown_table(["Candidate", "Trees", "Mean ROC-AUC", "Mean Brier"], confirmation_rows)}

Selected **{selection["selected_candidate"]}** using the highest mean confirmation ROC-AUC, **{selection["mean_confirmation_auc"]:.6f}**.

## Locked-fold comparison

{markdown_table(["Configuration", "FOLD=0 ROC-AUC", "Average precision", "PR-AUC (trapezoid)", "Brier"], [
    ["Fresh 250-tree baseline", f'{outer["baseline_auc"]:.6f}', f'{outer["baseline_average_precision"]:.6f}', f'{outer["baseline_pr_auc_trapezoid"]:.6f}', f'{outer["baseline_brier"]:.6f}'],
    ["Selected XGBoost", f'{outer["tuned_auc"]:.6f}', f'{outer["tuned_average_precision"]:.6f}', f'{outer["tuned_pr_auc_trapezoid"]:.6f}', f'{outer["tuned_brier"]:.6f}'],
])}

Average precision (AP) is the stepwise precision-recall summary used by the random-forest reference file. The separately reported PR-AUC uses trapezoidal integration of the precision-recall curve; the two are close but not identical. The validation positive rate is **{metrics["positive_rate_validation"]:.6f}**. The ROC-AUC difference from the fresh baseline is **{outer["auc_delta"]:+.6f}**. The baseline and selected model use different parameters, so this comparison measures the effect of the selected configuration on this fold.

## Final model and files

{markdown_table(["Parameter", "Value"], parameter_rows)}

After validation, a new model with this configuration is fitted on all {len(data.targets):,} labeled rows. Its `model.json`, `submission.csv`, `validation_predictions.csv`, `baseline_validation_predictions.csv`, and `xgboost_validation_metrics.json` are in `{output_rel}/`. `submission.csv` contains `P(TARGET=1)` for the {len(data.test_ids):,} unlabeled test rows. The test labels are unavailable, so no test ROC-AUC is claimed.

The older five-model comparison used different 508-feature inputs and remains a historical reference. This run is a single-model XGBoost result.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    final_result = f"""# Final XGBoost result on updated CSV data

The final run uses **{len(data.feature_names)} features** from `{args.train_csv.name}` and `{args.test_csv.name}`. For FOLD=0 validation, the model trained on {metrics["n_train"]:,} rows and predicted {metrics["n_validation"]:,} rows.

| Model | ROC-AUC | Average precision | PR-AUC (trapezoid) | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Fresh 250-tree XGBoost baseline | {outer["baseline_auc"]:.6f} | {outer["baseline_average_precision"]:.6f} | {outer["baseline_pr_auc_trapezoid"]:.6f} | {outer["baseline_brier"]:.6f} |
| **Selected XGBoost** | **{outer["tuned_auc"]:.6f}** | **{outer["tuned_average_precision"]:.6f}** | **{outer["tuned_pr_auc_trapezoid"]:.6f}** | **{outer["tuned_brier"]:.6f}** |

Average precision is the same metric named in the random-forest five-fold JSON. The separate PR-AUC column uses trapezoidal integration, so the two values differ slightly. The FOLD=0 positive rate is **{metrics["positive_rate_validation"]:.6f}**. The ROC-AUC change from the fresh baseline is **{outer["auc_delta"]:+.6f}**. FOLD=0 was not used for this run's parameter selection; it is a validation fold, not a labeled Kaggle test set.

## Final parameters

{markdown_table(["Parameter", "Value"], parameter_rows)}

The final model was refitted on all {len(data.targets):,} labeled rows. See [model.json]({output_link}/model.json), [submission.csv]({output_link}/submission.csv), [validation metrics and parameters]({output_link}/xgboost_validation_metrics.json), and the [full tuning report]({args.report.name}).

The earlier 508-feature model results are not directly comparable as an isolated algorithm effect because the input data changed.
"""
    final_path = args.report.parent / "xgboost_final_result.md"
    final_path.write_text(final_result, encoding="utf-8")
    print(f"Reports: {args.report}, {final_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "search", "confirm", "final", "report"], default="all")
    parser.add_argument("--data-zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline-validation", type=Path, default=BASELINE_VALIDATION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=600)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    if args.n_jobs < 1 or args.patience < 1 or args.max_rounds < 250:
        parser.error("n-jobs/patience must be positive and max-rounds >= 250")
    if (args.train_csv is None) != (args.test_csv is None):
        parser.error("--train-csv and --test-csv must be provided together")
    if args.train_csv is not None:
        if args.output_dir == DEFAULT_OUTPUT:
            args.output_dir = PROJECT_DIR / "outputs" / "xgboost_final"
        if args.cache_dir == DEFAULT_CACHE:
            args.cache_dir = PROJECT_DIR / ".cache_final_csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = (load_prepared_csv(args.train_csv, args.test_csv, args.cache_dir)
            if args.train_csv is not None else load_prepared(args.data_zip, args.cache_dir))
    ensure_context(args, data)
    print(f"Data: {len(data.targets):,} train, {len(data.test_ids):,} test, {len(data.feature_names)} features", flush=True)
    if args.stage in ("all", "search"):
        screening(args, data)
    if args.stage in ("all", "confirm"):
        confirmation(args, data)
    if args.stage in ("all", "final"):
        final_evaluation(args, data)
    if args.stage == "report":
        selection = json.loads((args.output_dir / "selection.json").read_text(encoding="utf-8"))
        outer = json.loads((args.output_dir / "outer_comparison.json").read_text(encoding="utf-8"))
        if args.train_csv is not None:
            write_csv_report(args, data, selection, outer)
        else:
            write_report(args, data, selection, outer)


if __name__ == "__main__":
    main()
