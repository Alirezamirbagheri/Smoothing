# parameter_extractor.py

from helpers import arc_length_param, local_to_global_ascending, compute_length
from scipy.interpolate import PPoly, interp1d
import pandas as pd
import numpy as np
import csv


def generate_segment_coeffs(results, file_path_old, file_path_updated):
    """
    Extracts polynomial coefficients for each spline segment and converts them.
    Generates the initial (old format) CSV.
    """
    # ... (content remains the same as previous generate_segment_coeffs) ...
    coeff_rows = []

    for idx, data in enumerate(results):
        x_s, y_s, path_len, tck = data['x_smoothed'], data['y_smoothed'], data['path_len'], data['tck']

        if tck is None: continue

        tx, c, k = tck
        c[0][0] = x_s[0]
        c[1][0] = y_s[0]
        c[0][-1] = x_s[-1]
        c[1][-1] = y_s[-1]

        ppx = PPoly.from_spline((tx, c[0], k))
        ppy = PPoly.from_spline((tx, c[1], k))

        valid_indices = [j for j in range(len(ppx.x) - 1)
                         if abs(ppx.x[j + 1] - ppx.x[j]) > 1e-9]
        mover_segments = []

        for seg_idx_local, j in enumerate(valid_indices):
            u_start = float(ppx.x[j])
            u_end = float(ppx.x[j + 1])
            coeffs_x_local = ppx.c[:, j]
            coeffs_y_local = ppy.c[:, j]

            Ax, Bx, Cx, Dx = local_to_global_ascending(coeffs_x_local, u_start)
            Ay, By, Cy, Dy = local_to_global_ascending(coeffs_y_local, u_start)

            u_vals = np.linspace(u_start, u_end, 200)
            x_vals = np.polyval([Ax, Bx, Cx, Dx][::-1], u_vals)
            y_vals = np.polyval([Ay, By, Cy, Dy][::-1], u_vals)
            seg_length = np.sum(np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2))
            length_weight = seg_length / path_len if path_len > 1e-9 else 0.0

            t_res = arc_length_param(x_s, y_s)
            mask_smooth = (t_res >= u_start) & (t_res <= u_end)
            num_smooth_points = np.sum(mask_smooth)
            time_scale_factor = max(1, num_smooth_points)

            mover_segments.append([
                idx + 1, seg_idx_local + 1, u_start, u_end,
                Ax, Bx, Cx, Dx, Ay, By, Cy, Dy,
                seg_length, length_weight, time_scale_factor
            ])

        total_points = sum(row[-1] for row in mover_segments)
        if total_points > 1e-9:
            for row in mover_segments:
                row[-1] = row[-1] / total_points

        coeff_rows.extend(mover_segments)

    # Write Initial/Old CSV
    header_old = [
        'Mover', 'Segment', 'U_Start', 'U_End',
        'Ax', 'Bx', 'Cx', 'Dx', 'Ay', 'By', 'Cy', 'Dy',
        'Length_mm', 'Length_Weight', 'Time_Scale_Factor'
    ]
    with open(file_path_old, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_old)
        writer.writerows([row[:len(header_old)] for row in coeff_rows])

    print(f"✅ Spline coefficients saved (old format): {file_path_old}")
    return coeff_rows


# ... (all other functions remain the same) ...
def save_smoothed_points_csv(results, file_path, resample_points):
    # ... (content remains the same) ...
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


def save_raw_point_coverage(results, file_path_pattern):
    # ... (content remains the same) ...
    for mover_idx, data in enumerate(results):
        x_raw, y_raw = data['x_raw'], data['y_raw']
        x_s, y_s = data['x_smoothed'], data['y_smoothed']

        t_raw = arc_length_param(x_raw, y_raw)
        t_smooth = arc_length_param(x_s, y_s)

        idx_nearest = np.argmin(np.abs(t_raw[:, None] - t_smooth[None, :]), axis=1)
        dist_to_prev = np.sqrt(np.diff(x_raw, prepend=x_raw[0]) ** 2 + np.diff(y_raw, prepend=y_raw[0]) ** 2)

        df_coverage = pd.DataFrame({
            'RawX': x_raw, 'RawY': y_raw, 'MappedSmoothedIndex': idx_nearest,
            'MappedSmoothedX': x_s[idx_nearest], 'MappedSmoothedY': y_s[idx_nearest],
            'DistToPrev': dist_to_prev
        })

        coverage_file = file_path_pattern.format(mover_idx + 1)
        df_coverage.to_csv(coverage_file, index=False, float_format='%.6f')
        print(f"✅ Raw points mapping saved for Mover {mover_idx + 1}: {coverage_file}")


def count_raw_points_per_segment(results, num_movers):
    # ... (content remains the same) ...
    raw_per_segment_rows = []

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

        for seg_idx_local, j in enumerate(valid_indices):
            u_start = float(ppx.x[j])
            u_end = float(ppx.x[j + 1])

            mask_smooth = (t_res >= u_start) & (t_res <= u_end)
            smoothed_indices = np.where(mask_smooth)[0]

            mask_raw = np.isin(idx_nearest, smoothed_indices)
            num_raw_points = np.sum(mask_raw)

            raw_per_segment_rows.append({
                'Mover': mover_idx + 1,
                'Segment': seg_idx_local + 1,
                'NumRawPoints': int(num_raw_points)
            })

    return pd.DataFrame(raw_per_segment_rows)


def update_and_save_raw_points_weight(df_raw_seg, file_path):
    # ... (content remains the same) ...
    if df_raw_seg.empty:
        print("Skipping weight update as raw segment data is empty.")
        return pd.DataFrame()

    total_per_mover = df_raw_seg.groupby('Mover')['NumRawPoints'].transform('sum')
    df_raw_seg['RawPointsWeight'] = df_raw_seg['NumRawPoints'] / total_per_mover
    df_raw_seg.to_csv(file_path, index=False, float_format='%.6f')
    print(f"✅ Updated {file_path} with RawPointsWeight column")

    return df_raw_seg


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

        updated_coeff_rows.append(row[:15] + [num_raw_points])

    header_updated = ['Mover', 'Segment', 'U_Start', 'U_End',
                      'Ax', 'Bx', 'Cx', 'Dx', 'Ay', 'By', 'Cy', 'Dy',
                      'Length_mm', "Length_Weight", 'Time_Scale_Factor', 'NumRawPoints']

    with open(file_path_updated, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_updated)
        writer.writerows(updated_coeff_rows)

    print(f"✅ Updated spline coefficients saved with NumRawPoints: {file_path_updated}")