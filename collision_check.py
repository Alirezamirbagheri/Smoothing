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
    mover_u_at_time = []

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
        u_at_time = np.zeros(num_frames)

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
            u_at_time[t_idx] = u_g

        mover_pos_at_time.append(path_at_time)
        mover_u_at_time.append(u_at_time)

    return mover_pos_at_time, T_global, mover_u_at_time


def check_mover_collisions(results, config):
    """
    Checks for spatial and temporal collisions between all mover pairs
    using the AABB overlap criteria and extracts all continuous collision events
    needed for time buffering.
    """
    # NOTE: The return signature MUST match the function definition
    # Ensure _precompute_mover_positions returns all three values
    mover_pos_at_time, T_global, mover_u_at_time = _precompute_mover_positions(results, config)

    num_movers = len(results)
    threshold = config['MOVER_SIZE']
    # CHANGE: 'collisions' now holds detailed event data, not just the first instance
    collisions_info = []

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

            # collision_indices holds the raw point (timestep) indices where collision occurred
            collision_indices = np.where(collision_mask)[0]

            if collision_indices.size > 0:
                # --- NEW LOGIC: Group contiguous raw points into collision events ---
                events = []

                # Identify where contiguous blocks of collision indices start
                # np.diff is > 1 where there is a break (a gap of at least 1 rawpoint)
                diffs = np.diff(collision_indices)

                # Indices in collision_indices where a new contiguous block starts
                event_starts_indices = np.where(diffs > 1)[0] + 1
                event_starts_indices = np.insert(event_starts_indices, 0, 0)  # The first block starts at index 0

                for k in range(len(event_starts_indices)):

                    # 1. Get the Raw Point Index for the start
                    start_idx_raw = collision_indices[event_starts_indices[k]]

                    # 2. Get the Raw Point Index for the end of the block
                    if k < len(event_starts_indices) - 1:
                        # End is the index just before the next block starts
                        end_idx_raw = collision_indices[event_starts_indices[k + 1] - 1]
                    else:
                        # Last block ends at the last recorded collision index
                        end_idx_raw = collision_indices[-1]

                    duration_rp = end_idx_raw - start_idx_raw + 1

                    events.append({
                        'start_rawpoint': start_idx_raw,
                        'end_rawpoint': end_idx_raw,
                        'duration_rp': duration_rp,
                        'start_time_sec': T_global[start_idx_raw]
                    })

                collisions_info.append({
                    'Mover_A': i + 1,
                    'Mover_B': j + 1,
                    'Collision_Events': events
                })
                print(
                    f"🚨 Collision events detected between Mover {i + 1} and Mover {j + 1}. Total events: {len(events)}")

    if not collisions_info:
        print("✅ No inter-mover collisions detected across the simulated time.")

    # CHANGE: Return the detailed collision event data structure
    return collisions_info