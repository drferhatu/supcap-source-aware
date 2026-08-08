from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path('results/cleaned_dataset_195.csv')
OUT_METRICS = Path('results/split_protocol_metrics.csv')
OUT_SUMMARY = Path('results/split_protocol_summary.txt')
OUT_FIG = Path('images/split_protocol_sensitivity.png')

RNG = np.random.default_rng(42)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


def fit_ridge(X, y, alpha=1.0):
    # closed-form ridge with intercept handled by centered features
    XtX = X.T @ X
    I = np.eye(X.shape[1])
    beta = np.linalg.solve(XtX + alpha * I, X.T @ y)
    return beta


def predict_ridge(X, beta):
    return X @ beta


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=123):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    r2_vals, mae_vals, rmse_vals = [], [], []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        r2_vals.append(r2_score(yt, yp))
        mae_vals.append(mae(yt, yp))
        rmse_vals.append(rmse(yt, yp))
    def q(arr):
        a = np.array(arr)
        return np.quantile(a, 0.025), np.quantile(a, 0.975)
    return q(r2_vals), q(mae_vals), q(rmse_vals)


def main():
    df = pd.read_csv(DATA_PATH)
    target_col = 'Cap(F/g)'
    group_col = 'Reference'
    drop_cols = [target_col, group_col, 'Biomass materials', 'Family']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    groups = df[group_col].astype(str).to_numpy()

    # Random split repeated evaluation
    n = len(df)
    n_test = int(round(0.2 * n))
    random_rows = []
    for rep in range(250):
        idx = RNG.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]

        mu, sigma = standardize_fit(Xtr)
        Xtr_s = standardize_apply(Xtr, mu, sigma)
        Xte_s = standardize_apply(Xte, mu, sigma)

        y_mu = ytr.mean()
        ytr_c = ytr - y_mu

        beta = fit_ridge(Xtr_s, ytr_c, alpha=1.0)
        yhat = predict_ridge(Xte_s, beta) + y_mu

        random_rows.append({
            'protocol': 'Random (80/20 repeated)',
            'rep': rep,
            'R2': r2_score(yte, yhat),
            'MAE': mae(yte, yhat),
            'RMSE': rmse(yte, yhat),
        })

    random_df = pd.DataFrame(random_rows)

    # Group-based leave-one-reference-out evaluation
    group_preds = np.zeros_like(y)
    unique_groups = np.unique(groups)
    fold_rows = []
    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask

        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]

        mu, sigma = standardize_fit(Xtr)
        Xtr_s = standardize_apply(Xtr, mu, sigma)
        Xte_s = standardize_apply(Xte, mu, sigma)

        y_mu = ytr.mean()
        ytr_c = ytr - y_mu
        beta = fit_ridge(Xtr_s, ytr_c, alpha=1.0)
        yhat = predict_ridge(Xte_s, beta) + y_mu

        group_preds[test_mask] = yhat

        fold_rows.append({
            'protocol': 'Group-based (Leave-One-Reference-Out)',
            'group': g,
            'n_test': int(test_mask.sum()),
            'R2': r2_score(yte, yhat),
            'MAE': mae(yte, yhat),
            'RMSE': rmse(yte, yhat),
        })

    group_fold_df = pd.DataFrame(fold_rows)

    group_global = {
        'protocol': 'Group-based (Leave-One-Reference-Out)',
        'rep': -1,
        'R2': r2_score(y, group_preds),
        'MAE': mae(y, group_preds),
        'RMSE': rmse(y, group_preds),
    }

    # CI computations
    r_r2_ci = np.quantile(random_df['R2'], [0.025, 0.975])
    r_mae_ci = np.quantile(random_df['MAE'], [0.025, 0.975])
    r_rmse_ci = np.quantile(random_df['RMSE'], [0.025, 0.975])

    g_r2_ci, g_mae_ci, g_rmse_ci = bootstrap_ci(y, group_preds, n_boot=2000, seed=7)

    summary_lines = [
        f"Random_mean_R2: {random_df['R2'].mean():.3f}",
        f"Random_CI95_R2: [{r_r2_ci[0]:.3f}, {r_r2_ci[1]:.3f}]",
        f"Random_mean_MAE: {random_df['MAE'].mean():.3f}",
        f"Random_CI95_MAE: [{r_mae_ci[0]:.3f}, {r_mae_ci[1]:.3f}]",
        f"Random_mean_RMSE: {random_df['RMSE'].mean():.3f}",
        f"Random_CI95_RMSE: [{r_rmse_ci[0]:.3f}, {r_rmse_ci[1]:.3f}]",
        f"Group_global_R2: {group_global['R2']:.3f}",
        f"Group_CI95_R2: [{g_r2_ci[0]:.3f}, {g_r2_ci[1]:.3f}]",
        f"Group_global_MAE: {group_global['MAE']:.3f}",
        f"Group_CI95_MAE: [{g_mae_ci[0]:.3f}, {g_mae_ci[1]:.3f}]",
        f"Group_global_RMSE: {group_global['RMSE']:.3f}",
        f"Group_CI95_RMSE: [{g_rmse_ci[0]:.3f}, {g_rmse_ci[1]:.3f}]",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines))

    out_df = pd.concat([
        random_df[['protocol', 'rep', 'R2', 'MAE', 'RMSE']],
        pd.DataFrame([group_global])
    ], ignore_index=True)
    out_df.to_csv(OUT_METRICS, index=False)

    # figure
    sns.set_theme(style='whitegrid', context='talk')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['R2', 'MAE', 'RMSE']
    random_color = '#2E86DE'
    group_color = '#E74C3C'

    group_vals = {'R2': group_global['R2'], 'MAE': group_global['MAE'], 'RMSE': group_global['RMSE']}
    group_cis = {'R2': g_r2_ci, 'MAE': g_mae_ci, 'RMSE': g_rmse_ci}

    for ax, m in zip(axes, metrics):
        sns.violinplot(y=random_df[m], color=random_color, alpha=0.35, inner='quartile', ax=ax)
        ax.scatter(0, group_vals[m], s=80, color=group_color, zorder=4)
        low, high = group_cis[m]
        ax.vlines(0, low, high, color=group_color, linewidth=3, zorder=3)
        ax.set_xticks([0])
        ax.set_xticklabels(['Random split\n(distribution)'])
        ax.set_title(m)
        ax.text(0.02, 0.95, f'Group-based: {group_vals[m]:.3f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, color=group_color)

    fig.suptitle('Split-Protocol Sensitivity: Random Split vs Group-Based Validation', y=1.03)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
    plt.close()

    # save fold-level details
    group_fold_df.to_csv('results/group_fold_metrics.csv', index=False)


if __name__ == '__main__':
    main()