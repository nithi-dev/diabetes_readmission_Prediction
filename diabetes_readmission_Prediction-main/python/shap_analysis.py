import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# Load the exact same XGBoost model evaluated in 07_evaluation.R
# Model was exported as JSON from R via xgb.save()
booster = xgb.Booster()
booster.load_model("outputs/results/model_xgb.json")

# Load cleaned + featured data (same as used in R modeling)
df = pd.read_csv("data/processed/diabetes_featured.csv")
df["readmitted_binary"] = df["readmitted_binary"].astype(int)

X = df.drop(columns=["readmitted_binary"])
y = df["readmitted_binary"]

# One-hot encode to match R's model.matrix encoding
X_enc = pd.get_dummies(X, drop_first=True)

# Load R's xgb_col_names to align columns exactly
# Generate from R: write.csv(data.frame(col_names=readRDS("outputs/results/xgb_col_names.rds")),
#                            "outputs/results/xgb_col_names.csv", row.names=FALSE)
try:
    col_names = pd.read_csv("outputs/results/xgb_col_names.csv")["col_names"].tolist()
    for col in col_names:
        if col not in X_enc.columns:
            X_enc[col] = 0
    X_enc = X_enc[col_names]
    print(f"Aligned to {len(col_names)} R model columns")
except FileNotFoundError:
    print("xgb_col_names.csv not found — using all encoded columns")
    col_names = X_enc.columns.tolist()

# 70/30 split — same seed as R
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.3, random_state=42, stratify=y
)

dtest = xgb.DMatrix(X_test)
preds = booster.predict(dtest)

from sklearn.metrics import roc_auc_score
print(f"Test AUC (R model loaded in Python): {roc_auc_score(y_test, preds):.3f}")

# SHAP analysis
print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_test)

# 22: SHAP bar — mean absolute SHAP (top 15)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, max_display=15,
                  show=False, plot_type="bar")
plt.title("feature importance — mean |SHAP value| (XGBoost)")
plt.tight_layout()
plt.savefig("outputs/figures/22_shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved 22_shap_importance_bar.png")

# 23: SHAP beeswarm — direction and magnitude
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title("SHAP summary — feature impact direction and magnitude")
plt.tight_layout()
plt.savefig("outputs/figures/23_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved 23_shap_beeswarm.png")

# 24: SHAP waterfall — single highest-risk patient
high_risk_idx = int(np.argmax(preds))
shap_exp = shap.Explanation(
    values        = shap_values[high_risk_idx],
    base_values   = explainer.expected_value,
    data          = X_test.iloc[high_risk_idx].values,
    feature_names = X_test.columns.tolist()
)
plt.figure(figsize=(9, 6))
shap.plots.waterfall(shap_exp, max_display=15, show=False)
plt.title("SHAP waterfall — highest-risk patient")
plt.tight_layout()
plt.savefig("outputs/figures/24_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved 24_shap_waterfall.png")

# Save SHAP summary CSV
shap_df = pd.DataFrame({
    "feature":   X_test.columns,
    "mean_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_shap", ascending=False).head(15)

print("\nTop 15 features by |SHAP|:")
print(shap_df.to_string(index=False))
shap_df.to_csv("outputs/results/shap_values.csv", index=False)
print("\nSaved: outputs/results/shap_values.csv")
