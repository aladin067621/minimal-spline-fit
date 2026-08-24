import csv
from pathlib import Path

import numpy as np

from .calc_engine import fit_trendline

MIN_POINTS_FOR_FIT = 4
FALLBACK_SCATTER = 1.0
LOG_FLOOR = 1e-15


def estimate_scatter(csv_path):
    """Read a CSV with columns: series, x, y_exp.
    Returns a dict mapping series name -> sigma_scatter."""
    rows = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["series"]
            x = float(row["x"])
            y = float(row["y_exp"])
            rows.setdefault(s, []).append((x, y))

    scatter = {}
    for series, points in rows.items():
        points.sort(key=lambda r: r[0])
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        sigma, func, _ = fit_trendline(xs, ys)

        if np.isnan(sigma) or sigma < LOG_FLOOR:
            sigma = LOG_FLOOR

        scatter[series] = float(sigma)

    return scatter


def estimate_scatter_from_arrays(series_data):
    """Compute sigma_scatter for a dict of {series_name: (x_array, y_array)}.
    Returns dict of {series_name: sigma_scatter}."""
    scatter = {}
    for series, (xs, ys) in series_data.items():
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        sigma, func, _ = fit_trendline(xs, ys)

        if np.isnan(sigma) or sigma < LOG_FLOOR:
            sigma = LOG_FLOOR

        scatter[series] = float(sigma)

    return scatter
