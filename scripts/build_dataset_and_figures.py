import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

XLSX_PATH = Path('Data/1-s2.0-S2352152X2402560X-mmc1.xlsx')
OUT_CSV = Path('results/cleaned_dataset_195.csv')
FIG_CORR = Path('images/data_corr_heatmap_modern.png')
FIG_COUNTS = Path('images/biomass_family_counts.png')

NS = {
    'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
}


def col_to_idx(col: str) -> int:
    n = 0
    for ch in col:
        if ch.isalpha():
            n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('a:si', NS):
                shared_strings.append(''.join(t.text or '' for t in si.findall('.//a:t', NS)))

        ws = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = ws.findall('a:sheetData/a:row', NS)

        table = []
        for r in rows[1:]:  # skip title row
            cells = [''] * 19
            for c in r.findall('a:c', NS):
                ref = ''.join(ch for ch in c.attrib.get('r', '') if ch.isalpha())
                idx = col_to_idx(ref)
                if idx < 0 or idx >= 19:
                    continue
                v = c.find('a:v', NS)
                if v is None:
                    val = ''
                elif c.attrib.get('t') == 's':
                    val = shared_strings[int(v.text)]
                else:
                    val = v.text
                cells[idx] = val
            if any(cells):
                table.append(cells)

    cols = table[0]
    data = table[1:]
    df = pd.DataFrame(data, columns=cols)
    return df


def family_from_name(name: str) -> str:
    n = str(name).lower()
    if any(k in n for k in ['husk', 'straw', 'grass', 'stem', 'leaf', 'cob', 'bran']):
        return 'Herbaceous residues'
    if any(k in n for k in ['shell', 'kernel', 'pit', 'seed', 'peel', 'pomelo', 'walnut', 'coconut']):
        return 'Fruit/nut residues'
    if any(k in n for k in ['wood', 'sawdust', 'bamboo', 'bagasse']):
        return 'Woody/lignified'
    return 'Other biomass'


def main():
    df = read_xlsx_without_openpyxl(XLSX_PATH)

    numeric_cols = ['C (%)', 'H (%)', 'O (%)', 'N (%)', 'M (%)', 'VM (%)', 'Ash (%)', 'FC (%)',
                    'Cel (%)', 'Hem (%)', 'Lig (%)', 'AT(℃)', 'RT(℃/min)', 'HT(min)', 'IR',
                    'Cond(A/g)', 'Cap(F/g)']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Family'] = df['Biomass materials'].map(family_from_name)

    clean_df = df.dropna(subset=numeric_cols).copy()
    clean_df.to_csv(OUT_CSV, index=False)

    summary = {
        'raw_rows': len(df),
        'clean_rows': len(clean_df),
        'unique_biomass': int(clean_df['Biomass materials'].nunique()),
        'unique_references': int(clean_df['Reference'].nunique()),
    }
    Path('results/dataset_summary.txt').write_text('\n'.join(f'{k}: {v}' for k, v in summary.items()))

    sns.set_theme(style='white', context='talk')

    corr_cols = ['Cap(F/g)', 'Cond(A/g)', 'AT(℃)', 'HT(min)', 'IR', 'Ash (%)', 'VM (%)', 'FC (%)', 'Cel (%)', 'Hem (%)', 'Lig (%)']
    corr = clean_df[corr_cols].corr(method='spearman')

    plt.figure(figsize=(11, 8))
    ax = sns.heatmap(
        corr,
        cmap='mako',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Spearman correlation'}
    )
    ax.set_title('Feature Correlation Landscape for Biomass-Derived Capacitance Dataset')
    plt.tight_layout()
    plt.savefig(FIG_CORR, dpi=300)
    plt.close()

    counts = clean_df.groupby('Family').size().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette('viridis', n_colors=len(counts))
    ax = sns.barplot(x=counts.values, y=counts.index, palette=palette)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.8, i, str(v), va='center', fontsize=11)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Biomass family')
    ax.set_title('Sample Coverage by Biomass Family (Cleaned Dataset)')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(FIG_COUNTS, dpi=300)
    plt.close()


if __name__ == '__main__':
    main()