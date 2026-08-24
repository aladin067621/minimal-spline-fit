"""Evaluate model predictions against experimental data.

Input: CSV with columns: series, x, y_exp, sigma_exp, y_sim
Output: E values per series and overall.

The E value quantifies the total error: how far model predictions
deviate from experiments, accounting for both experimental noise
(sigma_exp) and data scatter (sigma_scatter from trendline fitting).
"""

from pathlib import Path
from src.evaluation import evaluate


# --- Point to your data CSV here ---
INPUT_CSV = Path("your_data.csv")
OUTPUT_CSV = Path("evaluation_results.csv")
# ------------------------------------

series_results, e_overall = evaluate(INPUT_CSV, OUTPUT_CSV)

print(f"{'Series':<30} {'N':>5} {'Sigma_sc':>12} {'E':>10}")
print("-" * 60)
for s, r in sorted(series_results.items()):
    print(f"{s:<30} {r['n_points']:>5} {r['sigma_scatter']:>12.4f} {r['E']:>10.4f}")
print("-" * 60)
print(f"{'E_overall':<30} {'':>5} {'':>12} {e_overall:>10.4f}")
print(f"\nResults saved to {OUTPUT_CSV}")
