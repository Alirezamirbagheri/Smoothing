# parameter_extractor.py

from helpers import arc_length_param, local_to_global_ascending, compute_length
from scipy.interpolate import PPoly, interp1d
import pandas as pd
import numpy as np
import csv


# --- ONLY ONE EXPORT FUNCTION REMAINS ---
def generate_final_segment_coeffs(results, file_path_final):
    """
    Calculates coefficients, maps raw points to segments, and saves the final
    ParametricSplineCoeff.csv file. This replaces multiple steps.
    """
    final_coeff_rows = []

    for idx, data in enumerate(results):
        x_s, y_s, path_len, tck = data['x_smoothed'], data['y_smoothed'], data['path_len'], data['tck']

        if tck is None: continue

        tx, c, k = tck
        c[0][0] = x_s[0];
        c[1][0] = y_s[0];
        c[0][-1] = x_s[-1];
        c[1][-1] = y_s[-1]

        ppx = PPoly.from_spline((tx, c[0], k))
        ppy = PPoly.from_spline((tx, c[1], k))

        valid_indices = [j for j in range(len(ppx.x) - 1) if abs(ppx.x[j + 1] - ppx.x[j]) > 1e-9]

        # --- RAW POINT COUNTING LOGIC (MERGED) ---
        x_orig_raw, y_orig_raw = data['x_raw'], data['y_raw']
        t_raw = arc_length_param(x_orig_raw, y_orig_raw)
        t_smooth = arc_length_param(x_s, y_s)
        idx_nearest = np.argmin(np.abs(t_raw[:, None] - t_smooth[None, :]), axis=1)
        t_res = arc_length_param(x_s, y_s)
        # --- END MERGED LOGIC ---

        for seg_idx_local, j in enumerate(valid_indices):
            u_start = float(ppx.x[j]);
            u_end = float(ppx.x[j + 1])
            coeffs_x_local = ppx.c[:, j];
            coeffs_y_local = ppy.c[:, j]

            Ax, Bx, Cx, Dx = local_to_global_ascending(coeffs_x_local, u_start)
            Ay, By, Cy, Dy = local_to_global_ascending(coeffs_y_local, u_start)

            # Calculate physical segment length (Length_mm)
            u_vals = np.linspace(u_start, u_end, 200)
            x_vals = np.polyval([Ax, Bx, Cx, Dx][::-1], u_vals)
            y_vals = np.polyval([Ay, By, Cy, Dy][::-1], u_vals)
            seg_length = np.sum(np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2))

            # --- CALCULATE NumRawPoints FOR THIS SEGMENT ---
            mask_smooth = (t_res >= u_start) & (t_res <= u_end)
            smoothed_indices = np.where(mask_smooth)[0]
            mask_raw = np.isin(idx_nearest, smoothed_indices)
            num_raw_points = int(np.sum(mask_raw))
            # ---------------------------------------------

            final_coeff_rows.append([
                idx + 1, seg_idx_local + 1, u_start, u_end,
                Ax, Bx, Cx, Dx, Ay, By, Cy, Dy,
                seg_length,
                num_raw_points  # ADDED directly
            ])

    # Write Final CSV
    header_final = ['Mover', 'Segment', 'U_Start', 'U_End',
                    'Ax', 'Bx', 'Cx', 'Dx', 'Ay', 'By', 'Cy', 'Dy',
                    'Length_mm', 'NumRawPoints']

    with open(file_path_final, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_final)
        writer.writerows(final_coeff_rows)

    print(f"✅ Final spline coefficients generated and saved: {file_path_final}")
    # Return the rows for animation setup if needed, but not required for file cleanup
    # return final_coeff_rows


# --- KEPT FUNCTIONS ---
def save_smoothed_points_csv(results, file_path, resample_points):
    # ... (function body remains the same) ...
    smoothed_dict = {}
    for idx, data in enumerate(results):
        x_s, y_s = data['x_smoothed'], data['y_smoothed']
        N_out = resample_points
        u_src = np.linspace(0, 1, len(x_s))
        u_out = np.linspace(0, 1, N_out)
        fx = interp1d(u_src, x_s, kind='linear', fill_value="extrapolate")
        fy = interp1d(u_src, y_s, kind='linear', fill_value="extrapolate")
        smoothed_dict[f'xMover{idx + 1}'] = fx(u_out)
        smoothed_dict[f'yMover{idx + 1}'] = fy(u_out)
    pd.DataFrame(smoothed_dict).to_csv(file_path, index=False, float_format='%.3f')
    print(f"✅ Smoothed points saved: {file_path}")
