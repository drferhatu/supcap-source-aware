"""Generate two new dataset-description figures for R2:M2.

Outputs (written to Sources_Codes/images/):
    dataset_target_distribution.png
    dataset_process_variable_distributions.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path('results/cleaned_dataset_195.csv')
IMG_DIR = Path('images')
IMG_DIR.mkdir(exist_ok=True)


def target_distribution(df: pd.DataFrame) -> None:
    cap = df['Cap(F/g)'].to_numpy(dtype=float)
    mean = float(np.mean(cap))
    median = float(np.median(cap))
    p25 = float(np.percentile(cap, 25))
    p75 = float(np.percentile(cap, 75))

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    counts, bins, _ = ax.hist(cap, bins=22, color='#3E7CB1', edgecolor='#1F3E63',
                              alpha=0.85, linewidth=0.9)
    ymax = max(counts) * 1.18

    ax.axvline(mean, color='#B23A48', linestyle='--', linewidth=1.6,
               label=f'Mean = {mean:.1f} F g$^{{-1}}$')
    ax.axvline(median, color='#2A9D8F', linestyle='-.', linewidth=1.6,
               label=f'Median = {median:.1f} F g$^{{-1}}$')
    ax.axvspan(p25, p75, color='#F1C453', alpha=0.18,
               label=f'IQR: {p25:.1f} - {p75:.1f} F g$^{{-1}}$')

    ax.set_xlabel('Specific capacitance, Cap (F g$^{-1}$)', fontsize=11)
    ax.set_ylabel('Number of samples', fontsize=11)
    ax.set_title(f'Target-variable distribution (n = {len(cap)})', fontsize=12)
    ax.set_ylim(0, ymax)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(IMG_DIR / 'dataset_target_distribution.png', dpi=220,
                bbox_inches='tight')
    plt.close(fig)


def process_variable_distributions(df: pd.DataFrame) -> None:
    variables = [
        ('AT(℃)',        'Activation temperature (°C)',      '#3E7CB1'),
        ('RT(℃/min)',    'Heating ramp rate (°C min$^{-1}$)', '#8E7CC3'),
        ('HT(min)',      'Holding time (min)',                     '#E29578'),
        ('IR',           'Impregnation ratio',                     '#2A9D8F'),
        ('Cond(A/g)',    'Current density (A g$^{-1}$)',           '#B23A48'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.8))
    axes = axes.ravel()

    for ax, (col, label, color) in zip(axes, variables):
        vals = df[col].to_numpy(dtype=float)
        ax.hist(vals, bins=18, color=color, edgecolor='#333', alpha=0.85,
                linewidth=0.8)
        median = float(np.median(vals))
        ax.axvline(median, color='#111', linestyle='--', linewidth=1.2)
        ax.set_xlabel(label, fontsize=10.5)
        ax.set_ylabel('Count', fontsize=10.5)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.set_title(f'median = {median:g}', fontsize=9.5, color='#555')

    axes[-1].axis('off')  # blank sixth panel

    fig.suptitle('Process-variable distributions across the cleaned dataset '
                 f'(n = {len(df)})',
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'dataset_process_variable_distributions.png',
                dpi=220, bbox_inches='tight')
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    print('unique_references:', df['Reference'].nunique())
    print('unique_biomass:', df['Biomass materials'].nunique())
    print('unique_families:', df['Family'].nunique())
    print('n_samples:', len(df))
    cap = df['Cap(F/g)']
    print('Cap mean:', float(cap.mean()))
    print('Cap median:', float(cap.median()))
    print('Cap min:', float(cap.min()))
    print('Cap max:', float(cap.max()))


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print_summary(df)
    target_distribution(df)
    process_variable_distributions(df)
    print('Wrote:', IMG_DIR / 'dataset_target_distribution.png')
    print('Wrote:', IMG_DIR / 'dataset_process_variable_distributions.png')


if __name__ == '__main__':
    main()
