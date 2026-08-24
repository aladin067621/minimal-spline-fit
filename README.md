# Minimal Spline Fit

A Python toolbox for quantitative evaluation of model predictions against experimental data.

Estimates experimental noise (`sigma_scatter`) using Akima spline trendlines with AICc model selection, then computes objective function values (`E`) that quantify the total error of model predictions.

## Method

The approach combines three components:

1. **Minimal Spline Fit** — Fits Akima spline trendlines of increasing complexity to experimental data. The residual scatter from the best-fitting spline estimates `sigma_scatter`, the experimental noise [1].

2. **AICc (Akaike Information Criterion)** — Selects the optimal spline complexity by balancing fit quality against model complexity, preventing overfitting [2].

3. **Objective Function (E)** — Quantifies how far model predictions deviate from experiments, normalized by total uncertainty `sigma_total = sqrt(sigma_exp^2 + sigma_scatter^2)` [3].

### References

- [1] T. Nagy, T. Turanyi, *European Combustion Meeting 2021*, Paper 336.
- [2] H. Akaike, *IEEE Trans. Autom. Control* **19** (1974) 716-723.
- [3] T. Turanyi et al., *Int. J. Chem. Kinetics* **44** (2012) 284-302.

## Installation

```bash
pip install numpy scipy
```

## Usage

### Standalone: Estimate experimental noise

Provide a CSV with columns `series, x, y_exp`:

```python
from pathlib import Path
from src.scatter import estimate_scatter

scatter = estimate_scatter(Path("your_data.csv"))

for series, sigma in sorted(scatter.items()):
    print(f"{series}: sigma_scatter = {sigma:.4f}")
```

### Full evaluation: Compute E values

Provide a CSV with columns `series, x, y_exp, sigma_exp, y_sim`:

```python
from pathlib import Path
from src.evaluation import evaluate

series_results, e_overall = evaluate(
    Path("your_data.csv"),
    Path("results.csv")
)

for s, r in sorted(series_results.items()):
    print(f"{s}: E = {r['E']:.4f} (n={r['n_points']})")

print(f"E_overall = {e_overall:.4f}")
```

### Running the examples

```bash
python examples/scatter_only.py
python examples/full_evaluation.py
```

## Input Format

### For scatter estimation (`scatter.py`)

```csv
series,x,y_exp
series_1,100.0,0.5
series_1,150.0,0.7
series_2,200.0,0.3
```

### For full evaluation (`evaluation.py`)

```csv
series,x,y_exp,sigma_exp,y_sim
series_1,100.0,0.5,0.05,0.48
series_1,150.0,0.7,0.07,0.68
series_2,200.0,0.3,0.03,0.31
```

Each series must have at least 4 data points for spline fitting.

## Output

The E value per series is:

```
E_series = mean( ((y_sim - y_exp) / sigma_total)^2 )
```

where `sigma_total = sqrt(sigma_exp^2 + sigma_scatter^2)`.

`E_overall` is the mean of all pointwise E values across all series.

- `E ≈ 1` means model predictions match experiments within expected uncertainty.
- `E >> 1` means model predictions deviate significantly from experiments.
- `E << 1` means experimental uncertainties are overestimated.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@software{minimal_spline_fit,
  title  = {Minimal Spline Fit: Quantitative Evaluation of Model Predictions},
  year   = {2026},
  url    = {https://github.com/minimal-spline-fit}
}
```
