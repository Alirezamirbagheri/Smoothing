# collision_check.py

import numpy as np
import pandas as pd
from scipy.interpolate import splev


# --- NEW HELPER FUNCTION: Replicates lookup logic for local use ---
def _find_segment_index_from_rp(df_coeff, mover_id, rp_placement_index):
    """
    Finds the 'Segment' number that contains the given raw point index.
    The collision is detected at rp_placement_index, which falls within a segment.
    """
    # We target the segment that contains the raw point *before* the collision starts.
    target_rp_index = max(0, rp_placement_index)

    # 1. Calculate cumulative raw points if not already present (needed for lookup)
    if 'CumRawPointsEnd' not in df_coeff.columns:
        # NOTE: Using NumRawPoints is safe here as this DF is already loaded/modified.
        df_coeff['CumRawPointsEnd'] = df_coeff.groupby('Mover')['NumRawPoints'].cumsum()
        df_coeff['CumRawPointsStart'] = df_coeff.groupby('Mover')['CumRawPointsEnd'].shift(1, fill_value=0)

    segment_row = df_coeff[
        (df_coeff['Mover'] == mover_id) &
        (df_coeff['CumRawPointsStart'] <= target_rp_index) &
        (df_coeff['CumRawPointsEnd'] > target_rp_index)
        ]

    if not segment_row.empty:
        return segment_row['Segment'].iloc[0]
    return -1


def _precompute_mover_positions(results, config):
    """
    Replicates the density-scaled time-position precomputation setup from
    animations.py for collision checking purposes.

    Returns:
        mover_pos_at_time (list of [num_frames x 2] arrays): Position (x, y)
            of each mover at every frame.
        T_global (numpy array): The global time array corresponding to the frames.
        mover_u_at_time (list of arrays): Path parameter 'u' at every frame.
        df_coeff (DataFrame): The loaded coefficient data (for segment lookup).
    """
    # 🚨 CRITICAL FIX 1: Load Coefficient data using the dynamic animation path
    coeff_file_path = config['COEFF_OUTPUT_FILE_ANIMATION']

    try:
        df_coeff = pd.read_csv(coeff_file_path)
    except FileNotFoundError:
        print(f"Error: Coefficient file not found at {coeff_file_path}. Cannot perform collision check.")
        return None, None, None, None

    mover_total_raw_points = [
        df_coeff[df_coeff['Mover'] == i + 1]['NumRawPoints'].sum()
        for i in range(len(results))
    ]
    max_raw_points = max(mover_total_raw_points) if mover_total_raw_points else 0

    if max_raw_points <= 0: return None, None, None, None

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
            mover_u_at_time.append(np.zeros(num_frames))  # Append zeros for u
            continue

        # Density Scaling Logic (Time calculation based on NumRawPoints)
        if current_mover_raw_points > 0:
            # The NumRawPoints of the mover relative to the max RP determines the time scaling
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
                u_g = 1.0
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

    # 🚨 CRITICAL FIX 2: Ensure the df_coeff used for segment lookup has the necessary columns.
    # The DF loaded at the start already contains the modified NumRawPoints.
    if 'CumRawPointsEnd' not in df_coeff.columns:
        df_coeff['CumRawPointsEnd'] = df_coeff.groupby('Mover')['NumRawPoints'].cumsum()
        df_coeff['CumRawPointsStart'] = df_coeff.groupby('Mover')['CumRawPointsEnd'].shift(1, fill_value=0)

    return mover_pos_at_time, T_global, mover_u_at_time, df_coeff  # Return df_coeff now


def check_mover_collisions(results, config):
    """
    Checks for spatial and temporal collisions between all mover pairs
    using the AABB overlap criteria and extracts all continuous collision events
    needed for time buffering.
    """
    # NOTE: Updated to unpack the returned df_coeff
    mover_pos_at_time, T_global, mover_u_at_time, df_coeff = _precompute_mover_positions(results, config)

    if mover_pos_at_time is None: return []

    num_movers = len(results)
    threshold = config['MOVER_SIZE']
    collisions_info = []

    print("\n--- Running Inter-Mover Collision Check (AABB) ---")

    for i in range(num_movers):
        for j in range(i + 1, num_movers):
            mover_A_id = i + 1
            mover_B_id = j + 1

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
                diffs = np.diff(collision_indices)
                event_starts_indices = np.where(diffs > 1)[0] + 1
                event_starts_indices = np.insert(event_starts_indices, 0, 0)

                for k in range(len(event_starts_indices)):

                    # 1. Get the Raw Point Index for the start
                    start_idx_raw = collision_indices[event_starts_indices[k]]

                    # 2. Get the Raw Point Index for the end of the block
                    if k < len(event_starts_indices) - 1:
                        end_idx_raw = collision_indices[event_starts_indices[k + 1] - 1]
                    else:
                        end_idx_raw = collision_indices[-1]

                    duration_rp = end_idx_raw - start_idx_raw + 1

                    # --- NEW: Find the segment index for the *yielding* mover (Mover A by convention here) ---
                    start_segment_index = _find_segment_index_from_rp(df_coeff, mover_A_id, start_idx_raw)

                    events.append({
                        'start_rawpoint': start_idx_raw,
                        'end_rawpoint': end_idx_raw,
                        'duration_rp': duration_rp,
                        'start_time_sec': T_global[start_idx_raw],
                        'start_segment_index': start_segment_index  # 💡 ADDED TO EVENT DATA 💡
                    })

                collisions_info.append({
                    'Mover_A': mover_A_id,
                    'Mover_B': mover_B_id,
                    'Collision_Events': events
                })
                print(
                    f"🚨 Collision events detected between Mover {mover_A_id} and Mover {mover_B_id}. Total events: {len(events)}")

    if not collisions_info:
        print("✅ No inter-mover collisions detected across the simulated time.")

    return collisions_info