import warnings

import numpy as np
from scipy.interpolate import Akima1DInterpolator

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _fit_akima(x_data, y_data, k, prev_cx=None, prev_cy=None):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    y_range = max(y_max - y_min, 1e-10)

    n_param_x = k - 2
    n_param_y = k
    n_params = n_param_x + n_param_y

    def decode_params(params):
        if n_param_x > 0:
            raw = np.exp(np.clip(params[:n_param_x], -10, 10))
            padded = np.append(raw, 1.0)
            spacing = padded / np.sum(padded)
            cx = np.empty(k)
            cx[0] = x_min
            for i in range(1, k):
                cx[i] = x_min + np.sum(spacing[:i]) * (x_max - x_min)
        else:
            cx = np.array([x_min, x_max])
        cy = params[n_param_x:n_param_x + n_param_y]
        return cx, cy

    def evaluate(params):
        try:
            cx, cy = decode_params(params)
            if np.any(np.diff(cx) <= 0):
                return np.inf
            akima = Akima1DInterpolator(cx, cy)
            y_fit = akima(x_data)
            if np.any(np.isnan(y_fit)):
                return np.inf
            return float(np.sum((y_data - y_fit) ** 2))
        except Exception:
            return np.inf

    x_bounds = [(-5.0, 5.0)] * n_param_x
    y_lo = y_min - 0.5 * y_range
    y_hi = y_max + 0.5 * y_range
    y_bounds = [(y_lo, y_hi)] * n_param_y
    bounds = x_bounds + y_bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    n_particles = max(10, min(50, 5 * n_params))
    max_iter = max(20, min(150, 10 * n_params))
    w = 0.72
    c1 = 1.49
    c2 = 1.49
    v_max = (ub - lb) * 0.5

    equal_spaced = np.zeros(n_params)
    if n_param_x > 0:
        uniform_x = np.ones(n_param_x) / n_param_x
        equal_spaced[:n_param_x] = np.log(uniform_x * n_param_x + 1e-10)
    y_lin = np.linspace(y_min, y_max, k)
    equal_spaced[n_param_x:] = y_lin

    warm_start = None
    if prev_cx is not None and prev_cy is not None and len(prev_cx) >= 3:
        try:
            prev_akima = Akima1DInterpolator(prev_cx, prev_cy)
            new_x = np.linspace(x_min, x_max, k)
            new_y = prev_akima(new_x)
            ws = np.zeros(n_params)
            if n_param_x > 0:
                cx_spacing = np.diff(new_x) / (x_max - x_min)
                ws[:n_param_x] = np.log(cx_spacing[:k-1] * n_param_x + 1e-10)
            ws[n_param_x:] = new_y
            if not np.any(np.isnan(ws)) and not np.any(np.isinf(ws)):
                warm_start = ws
        except Exception:
            pass

    rng = np.random.RandomState(42)
    positions = rng.uniform(lb, ub, size=(n_particles, n_params))
    positions[0] = np.clip(equal_spaced, lb, ub)
    if warm_start is not None:
        positions[1] = np.clip(warm_start, lb, ub) if n_particles > 1 else positions[1]

    velocities = rng.uniform(-0.1, 0.1, size=(n_particles, n_params)) * (ub - lb)

    personal_best_pos = positions.copy()
    personal_best_val = np.array([evaluate(p) for p in positions])

    global_best_idx = int(np.argmin(personal_best_val))
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_val = personal_best_val[global_best_idx]

    if not np.isfinite(global_best_val):
        best_cx, best_cy = decode_params(equal_spaced)
        akima = Akima1DInterpolator(best_cx, best_cy)
        y_fit = akima(x_data)
        rss = float(np.sum((y_data - y_fit) ** 2))
        return best_cx, best_cy, rss, akima

    stall_count = 0
    for iteration in range(max_iter):
        r1 = rng.random((n_particles, n_params))
        r2 = rng.random((n_particles, n_params))

        velocities = (w * velocities +
                      c1 * r1 * (personal_best_pos - positions) +
                      c2 * r2 * (global_best_pos - positions))
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, lb, ub)

        for i in range(n_particles):
            val = evaluate(positions[i])
            if val < personal_best_val[i]:
                personal_best_val[i] = val
                personal_best_pos[i] = positions[i].copy()

        new_best_idx = int(np.argmin(personal_best_val))
        if personal_best_val[new_best_idx] < global_best_val:
            prev_best = global_best_val
            global_best_val = personal_best_val[new_best_idx]
            global_best_pos = personal_best_pos[new_best_idx].copy()
            if abs(prev_best - global_best_val) < 1e-10 * max(1, global_best_val):
                stall_count += 1
            else:
                stall_count = 0
        else:
            stall_count += 1

        if stall_count >= 15:
            break

    best_cx, best_cy = decode_params(global_best_pos)
    akima = Akima1DInterpolator(best_cx, best_cy)
    y_fit = akima(x_data)
    rss = float(np.sum((y_data - y_fit) ** 2))

    return best_cx, best_cy, rss, akima


def fit_trendline(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)

    if n < 4:
        return np.nan, None, 0

    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    unique_x, unique_indices = np.unique(x, return_inverse=True)
    if len(unique_x) < len(x):
        y_avg = np.array([np.mean(y[unique_indices == i]) for i in range(len(unique_x))])
        x, y = unique_x, y_avg

    best_aicc = np.inf
    best_sigma = np.nan
    best_func = None
    best_p = 0

    for order in range(0, min(4, n - 1)):
        try:
            coeffs = np.polyfit(x, y, order)
            p = order + 1
            y_fit = np.polyval(coeffs, x)
            rss = np.sum((y - y_fit) ** 2)

            if n - p <= 0:
                continue
            rmsd = np.sqrt(rss / n)
            sigma = np.sqrt(rss / (n - p))
            if rmsd > 0:
                aic = 2 * p + n + n * np.log(2 * np.pi * rmsd ** 2 / n)
            else:
                aic = 2 * p + n + n * np.log(2 * np.pi * 1e-30 / n)
            if n - p - 1 > 0:
                aicc = aic + 2 * p * (p + 1) / (n - p - 1)
            else:
                aicc = aic

            if aicc < best_aicc:
                best_aicc = aicc
                best_sigma = sigma
                best_func = lambda xi, c=coeffs, o=order: np.polyval(c, xi)
                best_p = p
        except (np.linalg.LinAlgError, ValueError):
            continue

    if n >= 4:
        best_spline_aicc = np.inf
        best_spline_sigma = np.nan
        best_spline_func = None
        best_spline_p = 0
        max_k = min(n // 2 + 1, n - 2)
        prev_aicc = np.inf
        prev_cx = None
        prev_cy = None

        for k in range(3, max_k + 1):
            try:
                cx, cy, rss, akima = _fit_akima(
                    x, y, k, prev_cx=prev_cx, prev_cy=prev_cy
                )

                p = 2 * k - 2

                if n - p <= 0:
                    prev_cx = cx
                    prev_cy = cy
                    continue

                rmsd = np.sqrt(rss / n)
                sigma = np.sqrt(rss / (n - p))

                if rmsd > 0:
                    aic = 2 * p + n + n * np.log(2 * np.pi * rmsd ** 2 / n)
                else:
                    aic = 2 * p + n + n * np.log(2 * np.pi * 1e-30 / n)

                if n - p - 1 > 0:
                    aicc = aic + 2 * p * (p + 1) / (n - p - 1)
                else:
                    aicc = aic

                if aicc > prev_aicc and best_spline_func is not None:
                    break

                if aicc < best_spline_aicc:
                    best_spline_aicc = aicc
                    best_spline_sigma = sigma
                    best_spline_func = akima
                    best_spline_p = p

                prev_aicc = aicc
                prev_cx = cx
                prev_cy = cy
            except (ValueError, np.linalg.LinAlgError):
                break

        if best_spline_func is not None and best_spline_aicc < best_aicc:
            best_aicc = best_spline_aicc
            best_sigma = best_spline_sigma
            best_func = best_spline_func
            best_p = best_spline_p

    if best_func is None and n >= 2:
        try:
            coeffs = np.polyfit(x, y, 1)
            y_fit = np.polyval(coeffs, x)
            rss = np.sum((y - y_fit) ** 2)
            p = 2
            sigma = np.sqrt(rss / (n - p)) if n - p > 0 else np.nan
            best_sigma = sigma
            best_func = lambda xi, c=coeffs: np.polyval(c, xi)
            best_p = p
        except Exception:
            pass

    return best_sigma, best_func, best_p
