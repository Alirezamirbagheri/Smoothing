# collision_check.py

import numpy as np
import pandas as pd
from scipy.interpolate import splev


def _precompute_mover_positions(results, config):
    """
    Replicates the density-scaled time-position precomputation setup from
    animations.py for collision checking purposes.

    Returns:
        mover_pos_at_time (list of [num_frames x 2] arrays): Position (x, y)
            of each mover at every frame.
        T_global (numpy array): The global time array corresponding to the frames.
    """
    # 1. Load Coefficient data
    try:
        df_coeff = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
    except FileNotFoundError:
        print("Error: Coefficient file not found. Cannot perform collision check.")
        return None, None

    mover_total_raw_points = [
        df_coeff[df_coeff['Mover'] == i + 1]['NumRawPoints'].sum()
        for i in range(len(results))
    ]
    max_raw_points = max(mover_total_raw_points) if mover_total_raw_points else 0

    if max_raw_points <= 0: return None, None

    FPS = 25
    T_sim = max_raw_points / FPS
    num_frames = max_raw_points
    T_global = np.linspace(0, T_sim, num_frames)

    mover_pos_at_time = []

    for mover_idx in range(len(results)):
        df_mover = df_coeff[df_coeff['Mover'] == mover_idx + 1].copy()
        data = results[mover_idx]
        tck = data['tck']
        path_len = data['path_len']
        current_mover_raw_points = mover_total_raw_points[mover_idx]

        if tck is None or path_len <= 0:
            # Append array of NaNs or the start position for static non-movers
            start_pos = [data['x_smoothed'][0], data['y_smoothed'][0]] if data['x_smoothed'].size else [0, 0]
            mover_pos_at_time.append(np.tile(start_pos, (num_frames, 1)))
            continue

        # Density Scaling Logic (Time calculation based on NumRawPoints)
        if current_mover_raw_points > 0:
            df_mover['Time_sec'] = T_sim * (df_mover['NumRawPoints'] / max_raw_points)
            df_mover['V_seg_mm_s'] = df_mover['Length_mm'] / df_mover['Time_sec']
        else:
            V_uniform = path_len / T_sim
            df_mover['V_seg_mm_s'] = V_uniform
            df_mover['Time_sec'] = df_mover['Length_mm'] / V_uniform

        df_mover['T_cum_end'] = df_mover['Time_sec'].cumsum()
        df_mover['T_cum_start'] = df_mover['T_cum_end'].shift(1, fill_value=0.0)

        path_at_time = np.zeros((num_frames, 2))

        # Calculate position for every time step
        for t_idx, t_g in enumerate(T_global):
            segment = df_mover[(t_g >= df_mover['T_cum_start']) & (t_g < df_mover['T_cum_end'])]

            if segment.empty:
                pos_xy = np.array(splev(1.0, tck)).flatten()  # End of path
            else:
                u_start = segment['U_Start'].iloc[0]
                u_end = segment['U_End'].iloc[0]
                t_start = segment['T_cum_start'].iloc[0]
                v_seg = segment['V_seg_mm_s'].iloc[0]
                seg_len = segment['Length_mm'].iloc[0]

                time_in_seg = t_g - t_start
                distance_in_seg = v_seg * time_in_seg
                u_seg_length = u_end - u_start

                u_local = distance_in_seg / seg_len if seg_len > 1e-6 else 0
                u_g = np.clip(u_start + u_local * u_seg_length, u_start, u_end)
                pos_xy = np.array(splev(u_g, tck)).flatten()

            path_at_time[t_idx, :] = pos_xy

        mover_pos_at_time.append(path_at_time)

    return mover_pos_at_time, T_global


def check_mover_collisions(results, config):
    """
    Checks for spatial and temporal collisions between all mover pairs
    using the AABB overlap criteria.
    """
    mover_pos_at_time, T_global = _precompute_mover_positions(results, config)
    # ... (Error checking remains the same) ...

    num_movers = len(results)
    # Collision is defined by AABB overlap: distance in X < S AND distance in Y < S
    threshold = config['MOVER_SIZE']
    collisions = []

    print("\n--- Running Inter-Mover Collision Check (AABB) ---")

    for i in range(num_movers):
        for j in range(i + 1, num_movers):
            pos_A = mover_pos_at_time[i]
            pos_B = mover_pos_at_time[j]

            # 1. Calculate absolute distance in X and Y at every time step
            dist_x = np.abs(pos_A[:, 0] - pos_B[:, 0])
            dist_y = np.abs(pos_A[:, 1] - pos_B[:, 1])

            # 2. Collision occurs if both conditions are met (AABB overlap)
            collision_mask = (dist_x < threshold) & (dist_y < threshold)

            collision_frames = np.where(collision_mask)[0]

            if collision_frames.size > 0:
                # Collision found! Report the first instance.
                first_frame = collision_frames[0]
                collision_time = T_global[first_frame]

                collisions.append({
                    'Mover_A': i + 1,
                    'Mover_B': j + 1,
                    'Time_sec': collision_time,
                    'Position_A': pos_A[first_frame, :],
                    'Position_B': pos_B[first_frame, :]
                })
                print(f"🚨 Collision detected between Mover {i + 1} and Mover {j + 1} at T={collision_time:.3f} s.")

    if not collisions:
        print("✅ No inter-mover collisions detected across the simulated time.")

    return collisions