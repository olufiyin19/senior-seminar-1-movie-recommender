"""Paired seed-by-seed stability analysis of hybrid weights (no model retraining)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path('results/hybrid_weight_experiment_results.csv')
OUT = Path('results')
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Missing {INPUT}. Run hybrid_weight_experiment.py first.')
    df = pd.read_csv(INPUT)
    required = {'availability', 'seed', 'collaborative_weight', 'mean_ndcg'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    if df.duplicated(['availability', 'seed', 'collaborative_weight']).any():
        raise ValueError('Duplicate availability/seed/weight combinations found.')
    pivot = df.pivot(index=['availability', 'seed'], columns='collaborative_weight', values='mean_ndcg')
    if pivot.isna().any().any() or not all(w in pivot.columns for w in WEIGHTS):
        raise ValueError('Incomplete weights across training conditions.')
    if len(pivot) != 21:
        raise ValueError(f'Expected 21 training conditions, found {len(pivot)}.')
    for availability, group in pivot.groupby(level='availability'):
        expected = 1 if availability == 100 else 5
        if len(group) != expected:
            raise ValueError(f'{availability}% has {len(group)} seeds; expected {expected}.')

    # Paired comparisons: same availability and seed, relative to standalone collaborative (weight=1).
    records = []
    for (availability, seed), scores in pivot.iterrows():
        for weight in WEIGHTS[:-1]:
            records.append({
                'availability': int(availability), 'seed': int(seed),
                'collaborative_weight': weight,
                'hybrid_ndcg': float(scores[weight]),
                'collaborative_ndcg': float(scores[1.0]),
                'delta_ndcg': float(scores[weight] - scores[1.0]),
            })
    paired = pd.DataFrame(records)
    paired.to_csv(OUT / 'hybrid_stability_paired.csv', index=False)

    summary = []
    for (availability, weight), group in paired.groupby(['availability', 'collaborative_weight'], sort=True):
        deltas = group['delta_ndcg'].to_numpy()
        n = len(deltas)
        mean = float(deltas.mean())
        sd = float(deltas.std(ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        # Approximate t-based interval for variation across the five paired subsamples.
        # t critical for df=4 (all reduced-data groups have five seeds).
        tcrit = 2.7764451051977987 if n == 5 else np.nan
        summary.append({
            'availability': availability, 'collaborative_weight': weight,
            'n_seeds': n, 'mean_delta_ndcg': mean, 'std_delta_ndcg': sd,
            'min_delta_ndcg': float(deltas.min()),
            'max_delta_ndcg': float(deltas.max()),
            'seeds_improved': int((deltas > 1e-12).sum()),
            'seeds_worsened': int((deltas < -1e-12).sum()),
            'seeds_tied': int((np.abs(deltas) <= 1e-12).sum()),
            'approx_ci95_lower': mean - tcrit * se if n == 5 else np.nan,
            'approx_ci95_upper': mean + tcrit * se if n == 5 else np.nan,
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'hybrid_stability_summary.csv', index=False)

    # Plot only weight=0.75 vs standalone: most relevant near-tie from previous report.
    selected = summary[(summary['collaborative_weight'] == 0.75) & (summary['n_seeds'] == 5)]
    x = selected['availability'].to_numpy()
    y = selected['mean_delta_ndcg'].to_numpy()
    lo = selected['approx_ci95_lower'].to_numpy()
    hi = selected['approx_ci95_upper'].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt='o-', capsize=5, label='Weight 0.75 minus weight 1.00')
    ax.axhline(0, linestyle='--', linewidth=1, color='gray')
    ax.set(xlabel='Training data availability (%)', ylabel='Paired NDCG@10 difference',
           title='Hybrid 0.75 vs collaborative 1.00: paired seed differences')
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'hybrid_stability_comparison.png', dpi=200)
    plt.close(fig)

    print('PASS: 21 complete training conditions and all five weights verified.')
    print('NOTE: 100% availability has one seed, so no variability interval is calculated.')
    print('NOTE: Intervals summarize subsampling variability, not independent-dataset generalization.')
    print('\nHybrid weight 0.75 versus standalone collaborative (weight 1.00):')
    print(summary[summary['collaborative_weight'] == 0.75].to_string(index=False, float_format=lambda v: f'{v:.6f}'))
    print('\nSaved:')
    for filename in ['hybrid_stability_paired.csv', 'hybrid_stability_summary.csv', 'hybrid_stability_comparison.png']:
        print('  results/' + filename)


if __name__ == '__main__':
    main()
