"""SHAP analysis of the XGBoost model on the G2 feature combination.

Outputs (into Sources_Codes/images/):
    shap_beeswarm_g2_xgb.png    -- summary beeswarm across all samples
    shap_bar_g2_xgb.png         -- mean |SHAP value| bar summary
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path("results/cleaned_dataset_195.csv")
IMG_DIR = Path("images")
IMG_DIR.mkdir(exist_ok=True)

G2_FEATURES = ["M (%)", "Ash (%)", "VM (%)", "FC (%)",
               "AT(℃)", "RT(℃/min)", "HT(min)", "IR", "Cond(A/g)"]
FEATURE_LABELS = ["M", "Ash", "VM", "FC", "AT", "RT", "HT", "IR", "Cond"]
TARGET = "Cap(F/g)"
SEED = 20260711


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[G2_FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                           random_state=SEED, shuffle=True)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_all_s = scaler.transform(X)

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.1, subsample=0.8,
        objective="reg:squarederror", random_state=SEED,
        tree_method="hist", verbosity=0,
    ).fit(X_tr_s, y_tr)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all_s)

    # --- Beeswarm summary plot ---
    plt.figure(figsize=(8.4, 5.6))
    shap.summary_plot(shap_values, X_all_s, feature_names=FEATURE_LABELS,
                      show=False, color_bar=True, max_display=9)
    fig = plt.gcf()
    fig.suptitle("SHAP summary — XGBoost on G2 (n=195)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "shap_beeswarm_g2_xgb.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --- Mean |SHAP| bar plot ---
    plt.figure(figsize=(7.2, 4.6))
    shap.summary_plot(shap_values, X_all_s, feature_names=FEATURE_LABELS,
                      plot_type="bar", show=False, max_display=9,
                      color="#3E7CB1")
    fig = plt.gcf()
    fig.suptitle("Mean |SHAP value| — XGBoost on G2",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "shap_bar_g2_xgb.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranking = pd.DataFrame({
        "feature": FEATURE_LABELS,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("Feature ranking by mean |SHAP value|:")
    print(ranking.to_string(index=False))

    ranking.to_csv("results/shap_g2_xgb_ranking.csv", index=False)


if __name__ == "__main__":
    main()
