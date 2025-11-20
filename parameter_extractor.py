# parameter_extractor.py

from helpers import arc_length_param, local_to_global_ascending, compute_length
from scipy.interpolate import PPoly, interp1d
import pandas as pd
import numpy as np
import csv


def generate_segment_coeffs(results, file_path_old, file_path_updated):
    """
    Extracts polynomial coefficients for each spline segment and converts them.
    Exports the minimal set of parameters.
    """
    coeff_rows = []

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
        mover_segments = []

        for seg_idx_local, j in enumerate(valid_indices):
            u_start = float(ppx.x[j]);
            u_end = float(ppx.x[j + 1])
            coeffs_x_local = ppx.c[:, j];
            coeffs_y_local = ppy.c[:, j]

            Ax, Bx, Cx, Dx = local_to_global_ascending(coeffs_x_local, u_start)
            Ay, By, Cy, Dy = local_to_global_ascending(coeffs_y_local, u_start)

            # Calculate physical segment length (Length_mm) - KEPT
            u_vals = np.linspace(u_start, u_end, 200)
            x_vals = np.polyval([Ax, Bx, Cx, Dx][::-1], u_vals)
            y_vals = np.polyval([Ay, By, Cy, Dy][::-1], u_vals)
            seg_length = np.sum(np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2))

            # Length_Weight and Time_Scale_Factor are REMOVED

            mover_segments.append([
                idx + 1, seg_idx_local + 1, u_start, u_end,
                Ax, Bx, Cx, Dx, Ay, By, Cy, Dy,
                seg_length  # Only Length_mm is appended here
            ])

        coeff_rows.extend(mover_segments)

    # Write Initial/Old CSV (REDUCED HEADER)
    header_old = [
        'Mover', 'Segment', 'U_Start', 'U_End',
        'Ax', 'Bx', 'Cx', 'Dx', 'Ay', 'By', 'Cy', 'Dy',
        'Length_mm'
    ]
    with open(file_path_old, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_old)
        writer.writerows([row[:len(header_old)] for row in coeff_rows])

    print(f"✅ Spline coefficients saved (old format - reduced): {file_path_old}")
    return coeff_rows


# save_smoothed_points_csv (remains unchanged)
def save_smoothed_points_csv(results, file_path, resample_points):
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


# --- Simplified: update_and_save_raw_points_weight function is DELETED ---
# We keep count_raw_points_per_segment but simplify the export in main.py.

def count_raw_points_per_segment(results, num_movers):
    """
    Counts the number of raw points that map to the smoothed points within
    each spline segment. Only NumRawPoints is returned.
    """
    raw_per_segment_rows = []
    total_raw_points_per_mover = {}

    for mover_idx in range(num_movers):
        data = results[mover_idx]
        x_s, y_s, tck = data['x_smoothed'], data['y_smoothed'], data['tck']

        if tck is None: continue

        x_orig_raw, y_orig_raw = data['x_raw'], data['y_raw']
        t_raw = arc_length_param(x_orig_raw, y_orig_raw)
        t_smooth = arc_length_param(x_s, y_s)
        idx_nearest = np.argmin(np.abs(t_raw[:, None] - t_smooth[None, :]), axis=1)

        tx, c, k = tck
        ppx = PPoly.from_spline((tx, c[0], k))
        valid_indices = [j for j in range(len(ppx.x) - 1) if abs(ppx.x[j + 1] - ppx.x[j]) > 1e-9]
        t_res = arc_length_param(x_s, y_s)

        current_mover_raw_count = 0

        for seg_idx_local, j in enumerate(valid_indices):
            u_start = float(ppx.x[j]);
            u_end = float(ppx.x[j + 1])
            mask_smooth = (t_res >= u_start) & (t_res <= u_end)
            smoothed_indices = np.where(mask_smooth)[0]
            mask_raw = np.isin(idx_nearest, smoothed_indices)
            num_raw_points = np.sum(mask_raw)
            current_mover_raw_count += num_raw_points

            raw_per_segment_rows.append({
                'Mover': mover_idx + 1,
                'Segment': seg_idx_local + 1,
                'NumRawPoints': int(num_raw_points)
            })

        total_raw_points_per_mover[mover_idx + 1] = current_mover_raw_count

    df_raw_seg = pd.DataFrame(raw_per_segment_rows)
    return df_raw_seg, total_raw_points_per_mover


# --- New function to export the simpler segment map ---
def save_raw_segment_map(df_raw_seg, file_path):
    """Saves only the Mover, Segment, and NumRawPoints columns."""
    df_raw_seg[['Mover', 'Segment', 'NumRawPoints']].to_csv(file_path, index=False, float_format='%.6f')
    print(f"✅ Raw segment map saved: {file_path} (NumRawPoints only)")


def finalize_coeff_file(coeff_rows, df_raw_seg, file_path_updated):
    """Appends NumRawPoints to the coefficient rows and saves the final CSV."""
    if df_raw_seg.empty:
        print("Skipping coefficient file finalization as raw segment data is empty.")
        return

    raw_points_lookup = df_raw_seg.set_index(['Mover', 'Segment'])['NumRawPoints'].to_dict()
    updated_coeff_rows = []

    for row in coeff_rows:
        mover, segment = row[0], row[1]
        num_raw_points = raw_points_lookup.get((mover, segment), 0)

        # Append NumRawPoints (now at index 12 since Length_Weight and Time_Scale_Factor were removed)
        updated_coeff_rows.append(row + [num_raw_points])

    header_updated = ['Mover', 'Segment', 'U_Start', 'U_End',
                      'Ax', 'Bx', 'Cx', 'Dx', 'Ay', 'By', 'Cy', 'Dy',
                      'Length_mm', 'NumRawPoints']  # REDUCED HEADER

    with open(file_path_updated, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_updated)
        writer.writerows(updated_coeff_rows)

    print(f"✅ Updated spline coefficients saved (simplified): {file_path_updated}")