from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

RANDOM_STATE = 42
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
FOLD_COL = "FOLD"

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    return out

def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_x = add_domain_features(train.drop(columns=[ID_COL, TARGET_COL, FOLD_COL]))
    test_x = add_domain_features(test.drop(columns=[ID_COL]))
    
    test_x = test_x.reindex(columns=train_x.columns)
    
    medians = train_x.median(numeric_only=True)
    
    train_x = train_x.fillna(medians).fillna(-999.0).astype(np.float32)
    test_x = test_x.fillna(medians).fillna(-999.0).astype(np.float32)
    
    return train_x, test_x

def make_model(n_estimators: int) -> RandomForestClassifier:

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=18,
        min_samples_leaf=20,
        max_features=0.50, 
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

data_dir = Path(r"D:/桌面/Project 1/data/processed")

output_dir = Path(r"E:\outputs")
n_estimators = 150
validation_fold = 0
n_folds = 5

output_dir.mkdir(parents=True, exist_ok=True)
print("参数配置完成，输出目录：", output_dir.resolve())

## 输出部分
print("正在读取训练集和测试集...")
train = pd.read_csv(data_dir / "final_train_encoded.csv")
test = pd.read_csv(data_dir / "final_test_encoded.csv")
y = train[TARGET_COL].astype(np.uint8)

train_aug = add_domain_features(train.drop(columns=[ID_COL, TARGET_COL, FOLD_COL]))
test_aug = add_domain_features(test.drop(columns=[ID_COL]))
test_aug = test_aug.reindex(columns=train_aug.columns)

print(f"总训练样本：{len(train)}，测试样本：{len(test)}，特征数：{train_aug.shape[1]}")

## 5折交叉验证
print("\n" + "="*50)
print("开始 5 折交叉验证")
print("="*50)

metrics_list = []

for fold in range(n_folds):
    print(f"\n===== 第 {fold} 折 =====")

    train_mask = train[FOLD_COL] != fold
    valid_mask = train[FOLD_COL] == fold
    
    X_tr = train_aug[train_mask].copy()
    X_val = train_aug[valid_mask].copy()
    y_tr = y[train_mask]
    y_val = y[valid_mask]
    
    medians = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(medians).fillna(-999.0).astype(np.float32)
    X_val = X_val.fillna(medians).fillna(-999.0).astype(np.float32)
    
    model = make_model(n_estimators)
    model.fit(X_tr, y_tr)
    
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    ap = average_precision_score(y_val, val_proba)
    
    fold_metrics = {
        "fold": fold,
        "n_train": int(train_mask.sum()),
        "n_validation": int(valid_mask.sum()),
        "positive_rate": float(y_val.mean()),
        "roc_auc": float(auc),
        "average_precision": float(ap),
    }
    metrics_list.append(fold_metrics)
    
    pd.DataFrame({
        ID_COL: train.loc[valid_mask, ID_COL],
        "pred_prob": val_proba,
        "actual_target": y_val
    }).to_csv(output_dir / f"random_forest_fold{fold}_predictions.csv", index=False)
    
    print(f"AUC: {auc:.4f} | AP: {ap:.4f}")

metrics_df = pd.DataFrame(metrics_list)
mean_auc = metrics_df["roc_auc"].mean()
std_auc = metrics_df["roc_auc"].std()
mean_ap = metrics_df["average_precision"].mean()
std_ap = metrics_df["average_precision"].std()

cv_summary = {
    "n_folds": n_folds,
    "mean_roc_auc": float(mean_auc),
    "std_roc_auc": float(std_auc),
    "mean_average_precision": float(mean_ap),
    "std_average_precision": float(std_ap),
    "per_fold_details": metrics_list,
    "model": "RandomForestClassifier",
    "parameters": make_model(n_estimators).get_params(),
}


(output_dir / "random_forest_5cv_summary.json").write_text(
    json.dumps(cv_summary, indent=2), encoding="utf-8"
)

print("\n" + "="*50)
print("5折交叉验证结果汇总")
print(f"平均 AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"平均 AP : {mean_ap:.4f} ± {std_ap:.4f}")
print("="*50)

## 全量数据训练
print("\n===== 开始全量数据训练最终模型 =====")

final_medians = train_aug.median(numeric_only=True)
X_final = train_aug.fillna(final_medians).fillna(-999.0).astype(np.float32)
X_test_final = test_aug.fillna(final_medians).fillna(-999.0).astype(np.float32)

final_model = make_model(n_estimators)
final_model.fit(X_final, y)

test_probability = final_model.predict_proba(X_test_final)[:, 1]
pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_COL: test_probability
}).to_csv(output_dir / "submission_random_forest.csv", index=False)

importance = pd.DataFrame({
    "feature": train_aug.columns,
    "importance": final_model.feature_importances_
})
importance.sort_values("importance", ascending=False).head(30).to_csv(
    output_dir / "random_forest_top30_features.csv", index=False
)

## 最终输出
print("\n" + "="*50)
print("全部流程完成")
print(f"5折汇总指标：{output_dir / 'random_forest_5cv_summary.json'}")
print(f"逐折预测结果：每折一个CSV文件")
print(f"提交文件：{output_dir / 'submission_random_forest.csv'}")
print(f"Top30特征：{output_dir / 'random_forest_top30_features.csv'}")
print("="*50)