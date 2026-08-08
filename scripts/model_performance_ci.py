"""Repeated random-split CIs for the five model families on feature combination G2.

For each of N_REPEATS random 80/20 splits, all five models are trained on the
G2 feature block (M, Ash, VM, FC, AT, RT, HT, IR, Cond) and evaluated on the
held-out test partition. The script reports mean +- std of R^2, MAE, RMSE for
train and test partitions, and paired Wilcoxon signed-rank tests between
XGBoost and each competitor on per-repeat test MAE.

Hyperparameters mirror the values reported in the manuscript.

Outputs (results/ inside Sources_Codes):
    model_ci_g2_raw.csv       -- per-repeat metrics for every model
    model_ci_g2_summary.csv   -- mean +/- std summary
    model_ci_g2_wilcoxon.csv  -- paired Wilcoxon results vs XGBoost
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import wilcoxon
import lightgbm as lgb
import xgboost as xgb

# The main manuscript workflow uses a Keras DNN (four hidden layers 256/128/64/32,
# ReLU, Adam, dropout, validation-loss early stopping). For repeated-split CI
# estimation we use scikit-learn's MLPRegressor with the same architecture,
# activation, optimiser, and early-stopping strategy. This substitution is
# purely for reproducible CI estimation across 30 splits without a full Keras
# deployment and is documented in the manuscript.

DATA_PATH = Path("results/cleaned_dataset_195.csv")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

G2_FEATURES = ["M (%)", "Ash (%)", "VM (%)", "FC (%)",
               "AT(℃)", "RT(℃/min)", "HT(min)", "IR", "Cond(A/g)"]
TARGET = "Cap(F/g)"
N_REPEATS = 30
TEST_SIZE = 0.2
BASE_SEED = 20260711


def make_dnn(seed: int):
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        max_iter=200,
        batch_size=32,
        random_state=seed,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {"R2": r2, "MAE": mae, "RMSE": rmse}


def score(model_name: str, y_tr, yhat_tr, y_te, yhat_te) -> dict[str, float]:
    tr = metrics(y_tr, yhat_tr)
    te = metrics(y_te, yhat_te)
    return {
        "model": model_name,
        "R2_train": tr["R2"], "MAE_train": tr["MAE"], "RMSE_train": tr["RMSE"],
        "R2_test":  te["R2"], "MAE_test":  te["MAE"], "RMSE_test":  te["RMSE"],
    }


def run_repeat(rep: int, X: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
    seed = BASE_SEED + rep
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, shuffle=True
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rows: list[dict[str, float]] = []

    # --- SVR ---
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale").fit(X_tr_s, y_tr)
    rows.append(score("SVR", y_tr, svr.predict(X_tr_s), y_te, svr.predict(X_te_s)))

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_split=2,
        n_jobs=-1, random_state=seed,
    ).fit(X_tr_s, y_tr)
    rows.append(score("Random Forest", y_tr, rf.predict(X_tr_s),
                      y_te, rf.predict(X_te_s)))

    # --- LightGBM ---
    lgbm = lgb.LGBMRegressor(
        n_estimators=500, num_leaves=31, max_depth=-1,
        learning_rate=0.1, random_state=seed, verbosity=-1,
    ).fit(X_tr_s, y_tr)
    rows.append(score("LightGBM", y_tr, lgbm.predict(X_tr_s),
                      y_te, lgbm.predict(X_te_s)))

    # --- XGBoost ---
    xgbr = xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.1, subsample=0.8,
        objective="reg:squarederror", random_state=seed,
        tree_method="hist", verbosity=0,
    ).fit(X_tr_s, y_tr)
    rows.append(score("XGBoost", y_tr, xgbr.predict(X_tr_s),
                      y_te, xgbr.predict(X_te_s)))

    # --- DNN (sklearn MLPRegressor substitute; see header comment) ---
    dnn = make_dnn(seed).fit(X_tr_s, y_tr)
    rows.append(score("DNN", y_tr, dnn.predict(X_tr_s),
                      y_te, dnn.predict(X_te_s)))

    # --- Ensemble (LightGBM + XGBoost mean) ---
    yhat_tr_ens = 0.5 * (lgbm.predict(X_tr_s) + xgbr.predict(X_tr_s))
    yhat_te_ens = 0.5 * (lgbm.predict(X_te_s) + xgbr.predict(X_te_s))
    rows.append(score("Ensemble (XGB+LGBM)", y_tr, yhat_tr_ens, y_te, yhat_te_ens))

    for r in rows:
        r["rep"] = rep
    return rows


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["R2_train", "MAE_train", "RMSE_train",
                "R2_test",  "MAE_test",  "RMSE_test"]
    summary = (
        raw.groupby("model")[agg_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    return summary.reset_index()


def wilcoxon_vs_xgb(raw: pd.DataFrame) -> pd.DataFrame:
    ref = "XGBoost"
    ref_mae = (raw[raw["model"] == ref]
               .sort_values("rep")["MAE_test"].to_numpy())
    rows = []
    for m in sorted(raw["model"].unique()):
        if m == ref:
            continue
        other = (raw[raw["model"] == m]
                 .sort_values("rep")["MAE_test"].to_numpy())
        stat, p = wilcoxon(ref_mae, other, alternative="less",
                           zero_method="wilcox", method="auto")
        rows.append({
            "comparison": f"{ref} vs {m}",
            "n_pairs": int(len(other)),
            "median_MAE_XGBoost": float(np.median(ref_mae)),
            f"median_MAE_{m}": float(np.median(other)),
            "wilcoxon_statistic": float(stat),
            "p_value_one_sided": float(p),
            "significant_at_0.05": bool(p < 0.05),
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in G2_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in dataset: {missing}")

    X = df[G2_FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    all_rows: list[dict[str, float]] = []
    for rep in range(N_REPEATS):
        rows = run_repeat(rep, X, y)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(f"rep {rep+1:02d}/{N_REPEATS} done  ({elapsed:6.1f}s elapsed)")

    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT_DIR / "model_ci_g2_raw.csv", index=False)

    summary = summarise(raw)
    summary.to_csv(OUT_DIR / "model_ci_g2_summary.csv", index=False)

    w = wilcoxon_vs_xgb(raw)
    w.to_csv(OUT_DIR / "model_ci_g2_wilcoxon.csv", index=False)

    print("\n--- Summary (mean ± std) ---")
    for _, row in summary.iterrows():
        print(
            f"{row['model']:<22} "
            f"R2 test = {row['R2_test_mean']:.3f} ± {row['R2_test_std']:.3f}   "
            f"MAE test = {row['MAE_test_mean']:.3f} ± {row['MAE_test_std']:.3f}   "
            f"RMSE test = {row['RMSE_test_mean']:.3f} ± {row['RMSE_test_std']:.3f}"
        )

    print("\n--- Paired Wilcoxon (XGBoost vs others, one-sided) ---")
    for _, row in w.iterrows():
        star = "*" if row["significant_at_0.05"] else " "
        print(f"{star} {row['comparison']:<32}  "
              f"p = {row['p_value_one_sided']:.4f}")

    print(f"\nTotal wall clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
