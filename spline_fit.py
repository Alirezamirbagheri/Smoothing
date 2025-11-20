# spline_fit.py

from helpers import prepare_points, rdp, arc_length_param, compute_max_deviation, compute_length
from scipy.interpolate import splprep, splev, interp1d
from data_loader import extract_mover_data
import numpy as np


def clean_and_simplify_path(x_raw, y_raw, rdp_tol):
    # ... (content remains the same as previous clean_and_simplify_path) ...
    try:
        x_cleaned, y_cleaned = prepare_points(x_raw, y_raw)
    except ValueError:
        x_cleaned, y_cleaned = x_raw.astype(float), y_raw.astype(float)

    pts = np.column_stack((x_cleaned, y_cleaned))
    pts_simpl = rdp(pts, rdp_tol)
    if pts_simpl.shape[0] < 3: pts_simpl = pts.copy()
    x_simp, y_simp = pts_simpl[:, 0], pts_simpl[:, 1]

    return x_cleaned, y_cleaned, x_simp, y_simp


def fit_spline(x_simp, y_simp, config):
    # ... (content remains the same as previous fit_spline) ...
    t_simp = arc_length_param(x_simp, y_simp)
    N_res = max(min(config['RESAMPLE_POINTS'], config['MAX_POINTS']), len(x_simp))
    u_uniform = np.linspace(0, 1, N_res)
    fx = interp1d(t_simp, x_simp, kind='linear', fill_value="extrapolate")
    fy = interp1d(t_simp, y_simp, kind='linear', fill_value="extrapolate")
    x_res, y_res = fx(u_uniform), fy(u_uniform)
    t_res = u_uniform

    k_fit = min(config['K_ORDER'], len(x_res) - 1)
    sf = config['SMOOTHING_FACTOR_INIT']
    tck = None

    while True:
        try:
            tck, _ = splprep([x_res, y_res], u=t_res, s=sf, k=k_fit, nest=-1)
            xs_fine, ys_fine = splev(np.linspace(0, 1, N_res), tck)

            xs_fine[0], ys_fine[0] = x_simp[0], y_simp[0]
            xs_fine[-1], ys_fine[-1] = x_simp[-1], y_simp[-1]

            max_dev, mean_dev = compute_max_deviation(x_res, y_res, xs_fine, ys_fine)

            if max_dev <= config['MAX_DEV_ALLOWED'] or sf < 1e-3:
                break
            sf *= 0.5
        except Exception as e:
            xs_fine, ys_fine = x_res, y_res
            tck = None
            max_dev, mean_dev = compute_max_deviation(x_res, y_res, xs_fine, ys_fine)
            break

    path_len = compute_length(xs_fine, ys_fine)

    return {
        'x_smoothed': xs_fine, 'y_smoothed': ys_fine,
        'path_len': path_len, 'tck': tck,
        'max_dev': max_dev, 'mean_dev': mean_dev
    }


def process_all_paths(df_raw, num_movers, config):
    """
    Iterates through all movers, cleans, simplifies, and fits the spline.
    """
    results = []
    print("\n--- Path Cleaning and Fitting ---")
    for i in range(num_movers):
        x_raw, y_raw = extract_mover_data(df_raw, i)

        x_cleaned, y_cleaned, x_simp, y_simp = clean_and_simplify_path(x_raw, y_raw, config['RDP_TOL'])
        fit_data = fit_spline(x_simp, y_simp, config)

        result = {
            'x_raw': x_raw, 'y_raw': y_raw,
            'x_cleaned': x_cleaned, 'y_cleaned': y_cleaned,
            'orig_len': compute_length(x_cleaned, y_cleaned),
            **fit_data
        }
        results.append(result)

        print(f"Mover {i + 1}: raw={len(x_raw)}, cleaned={len(x_cleaned)}, simplified={len(x_simp)}, "
              f"smoothed={len(result['x_smoothed'])}, max_dev={result['max_dev']:.2f}")

    return results