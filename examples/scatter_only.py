"""Estimate experimental noise (sigma_scatter) from data.

Input: CSV with columns: series, x, y_exp
Output: CSV with columns: series, sigma_scatter

Each series must have at least 4 data points for spline fitting.
"""

from pathlib import Path
from src.scatter import estimate_scatter


# --- Point to your data CSV here ---
INPUT_CSV = Path("your_data.csv")
OUTPUT_CSV = Path("scatter_results.csv")
# ------------------------------------

scatter = estimate_scatter(INPUT_CSV)

with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
    f.write("series,sigma_scatter\n")
    for series, sigma in sorted(scatter.items()):
        f.write(f"{series},{sigma:.8g}\n")

print(f"Computed sigma_scatter for {len(scatter)} series.")
print(f"Results saved to {OUTPUT_CSV}")
