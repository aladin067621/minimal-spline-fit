import csv
import math
from pathlib import Path

import numpy as np

from .scatter import LOG_FLOOR
from .calc_engine import fit_trendline

OUTLIER_E_THRESHOLD = 9.0


def read_csv(csv_path):
    """Read a CSV with columns: series, x, y_exp, sigma_exp, y_sim.
    Returns dict: {series: [(x, y_exp, sigma_exp, y_sim), ...]}"""
    data = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["series"]
            x = float(row["x"])
            y_exp = float(row["y_exp"])
            sigma_exp = float(row["sigma_exp"])
            y_sim = float(row["y_sim"])
            data.setdefault(s, []).append((x, y_exp, sigma_exp, y_sim))
    return data


def compute_sigma_scatter_per_series(data):
    """Fit trendlines to y_exp vs x for each series.
    Returns dict: {series: sigma_scatter}"""
    scatter = {}
    for series, points in data.items():
        points_sorted = sorted(points, key=lambda r: r[0])
        xs = np.array([p[0] for p in points_sorted])
        ys = np.array([p[1] for p in points_sorted])
        sigma, func, _ = fit_trendline(xs, ys)

        if np.isnan(sigma) or sigma < LOG_FLOOR:
            sigma = LOG_FLOOR

        scatter[series] = float(sigma)
    return scatter


def compute_e_values(data, sigma_scatter_dict):
    """Compute pointwise and per-series E values.
    Returns: (pointwise_rows, series_e_dict, e_overall)"""
    all_e = []
    series_results = {}

    for series, points in data.items():
        sigma_scat = sigma_scatter_dict[series]
        e_points = []

        for x, y_exp, sigma_exp, y_sim in points:
            sigma_total = math.sqrt(sigma_exp ** 2 + sigma_scat ** 2)
            e = ((y_sim - y_exp) / sigma_total) ** 2
            e_points.append(e)
            all_e.append(e)

        series_e = float(np.mean(e_points)) if e_points else float("nan")
        series_results[series] = {
            "n_points": len(e_points),
            "sigma_scatter": sigma_scat,
            "E": series_e,
        }

    e_overall = float(np.mean(all_e)) if all_e else float("nan")
    return series_results, e_overall


def evaluate(csv_path, output_path=None):
    """Full evaluation: reads CSV, computes sigma_scatter and E values.
    Prints results to stdout and optionally writes output CSV."""
    data = read_csv(csv_path)
    sigma_scatter = compute_sigma_scatter_per_series(data)
    series_results, e_overall = compute_e_values(data, sigma_scatter)

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["series", "n_points", "sigma_scatter", "E"])
            for s, r in sorted(series_results.items()):
                writer.writerow([s, r["n_points"], f"{r['sigma_scatter']:.8g}", f"{r['E']:.8g}"])
            writer.writerow([])
            writer.writerow(["E_overall", "", "", f"{e_overall:.8g}"])

    return series_results, e_overall
