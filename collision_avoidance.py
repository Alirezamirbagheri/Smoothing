# collision_avoidance.py

import numpy as np
import pandas as pd
from config import *
from scipy.interpolate import splev
from collision_check import _precompute_mover_positions  # Kept for consistency

# --- Configuration Constants for Avoidance ---
FPS = 25


# --- Helper Function for Segment Lookup (FIX 1: Targets Collision RP Index) ---
def _find_segment_index_from_rp(df_coeff, mover_id, rp_placement_index):
    """
    Finds the 'Segment' number that contains the given raw point index.
    CORRECTION: Targets the exact RP index where the collision STARTS (e.g., RP 109).
    """
    target_rp_index = max(0, rp_placement_index)

    segment_row = df_coeff[
        (df_coeff['Mover'] == mover_id) &
        (df_coeff['CumRawPointsStart'] <= target_rp_index) &
        (df_coeff['CumRawPointsEnd'] > target_rp_index)
        ]

    if not segment_row.empty:
        return segment_row['Segment'].iloc[0]
    return -1


def _distribute_raw_points(df_coeff_avoidance, mover_id, start_segment_index, total_rp_to_distribute):
    """
    Distributes or reduces the total_rp_to_distribute across ALL segments
    preceding the collision start segment, applying the reverse order and zero-guard logic.
    """

    remaining_deficit = abs(total_rp_to_distribute)
    is_addition = total_rp_to_distribute > 0
    total_rp_applied = 0

    # NEW: Log structure to capture before/after changes
    segment_rp_log = {}

    # Identify all segments *before* the collision starts, sorted in reverse order (closest first)
    preceding_segments_df = df_coeff_avoidance[
        (df_coeff_avoidance['Mover'] == mover_id) &
        (df_coeff_avoidance['Segment'] < start_segment_index)
        ].sort_values(by='Segment', ascending=False)

    preceding_segments_indices = preceding_segments_df.index
    num_preceding_segments = len(preceding_segments_indices)

    if num_preceding_segments == 0:
        print(
            f"Warning: Mover {mover_id} has no segments before collision segment {start_segment_index}. Skipping distribution.")
        return 0, {}  # Return the log as well

    # Capture initial RP for logging
    for global_index in preceding_segments_indices:
        segment_id = df_coeff_avoidance.loc[global_index, 'Segment']
        segment_rp_log[segment_id] = {'old': df_coeff_avoidance.loc[global_index, 'NumRawPoints']}

    # 2. Iterate and Apply Adjustment in reverse order (closest segment first)
    for i, global_index in enumerate(preceding_segments_indices):

        if remaining_deficit <= 0:
            break

        segments_left = num_preceding_segments - i
        current_rp = df_coeff_avoidance.loc[global_index, 'NumRawPoints']

        # Calculate the proportional share of the remaining deficit
        rp_share = int(np.ceil(remaining_deficit / segments_left))

        rp_change = 0

        if is_addition:
            # Case 1: POSITIVE MARGIN (Slowdown/Addition)
            rp_change = rp_share
            df_coeff_avoidance.loc[global_index, 'NumRawPoints'] += rp_change
            total_rp_applied += rp_change
            remaining_deficit -= rp_change

        else:
            # 💡 FIX 2 APPLIED: Robust proportional distribution for reduction (Speed-up)
            rp_can_be_removed = current_rp - 1
            rp_change = min(rp_share, rp_can_be_removed)

            if rp_change > 0:
                df_coeff_avoidance.loc[global_index, 'NumRawPoints'] -= rp_change
                total_rp_applied += rp_change
                remaining_deficit -= rp_change

        # Update log with new RP value
        segment_id = df_coeff_avoidance.loc[global_index, 'Segment']
        segment_rp_log[segment_id]['new'] = df_coeff_avoidance.loc[global_index, 'NumRawPoints']

    return total_rp_applied, segment_rp_log


def resolve_collisions(results, collisions_info, config):
    """
    Applies the Distributed Time Adjustment Collision Avoidance strategy.
    """
    if not collisions_info:
        print("✅ No collision events detected. No avoidance necessary.")
        return None, True

    # Load the base segment coefficients to modify
    try:
        df_coeff = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
    except FileNotFoundError:
        print("Error: Coefficient file not found. Cannot resolve collisions.")
        return None, False

    df_coeff_avoidance = df_coeff.copy()
    total_rp_adjusted = 0

    # 0. Pre-calculate total path length for yield decision
    mover_totals = df_coeff_avoidance.groupby('Mover').agg(
        TotalPathLength=('Length_mm', 'sum'),
        TotalRP=('NumRawPoints', 'sum')
    )
    mover_totals['AvgVelocityMetric'] = mover_totals['TotalPathLength'] / mover_totals['TotalRP']

    # 1. Calculate cumulative raw points for segment lookup
    df_coeff_avoidance['CumRawPointsEnd'] = df_coeff_avoidance.groupby('Mover')['NumRawPoints'].cumsum()
    df_coeff_avoidance['CumRawPointsStart'] = df_coeff_avoidance.groupby('Mover')['CumRawPointsEnd'].shift(1,
                                                                                                           fill_value=0)

    print("\n--- Applying Distributed Time Adjustment Strategy ---")

    time_factor = config.get('TIME_ADJUSTMENT_FACTOR', 1.0)
    mover_adjustments = {}

    for col_pair in collisions_info:
        mover_A_id = col_pair['Mover_A']
        mover_B_id = col_pair['Mover_B']

        # --- DECISION LOGIC: Yield based on Total Pathed Length ---
        length_A = mover_totals.loc[mover_A_id, 'TotalPathLength']
        length_B = mover_totals.loc[mover_B_id, 'TotalPathLength']

        mover_yield_id = mover_A_id if length_A >= length_B else mover_B_id
        mover_pass_id = mover_B_id if length_A >= length_B else mover_A_id
        # -----------------------------------------------

        if mover_yield_id not in mover_adjustments:
            mover_adjustments[mover_yield_id] = []

        for event in col_pair['Collision_Events']:
            duration_rp = event['duration_rp']
            rp_placement_index = event['start_rawpoint']

            # --- TARGET CALCULATION ---
            multiplier_magnitude = 1.0 + abs(time_factor)
            target_rp = np.ceil(duration_rp * multiplier_magnitude).astype(int)
            absolute_adjustment_rp = target_rp
            adjustment_rp = int(np.sign(time_factor) * absolute_adjustment_rp)
            # --------------------------

            if adjustment_rp == 0:
                print(f"Mover {mover_yield_id}: Calculated adjustment is 0 RP for duration {duration_rp}.")
                continue

            # Find the segment index where the collision *starts* (Segment 9)
            start_segment_index = _find_segment_index_from_rp(df_coeff_avoidance, mover_yield_id, rp_placement_index)

            if start_segment_index == -1:
                start_segment_index = event.get('start_segment_index', -1)
                if start_segment_index == -1:
                    print(
                        f"Warning: Could not find starting segment for Mover {mover_yield_id} at RP {rp_placement_index}. Skipping event.")
                    continue

            mover_adjustments[mover_yield_id].append({
                'start_segment_index': start_segment_index,
                'adjustment_rp': adjustment_rp
            })

            action = "Adding" if adjustment_rp > 0 else "Reducing by"
            print(
                f"M{mover_yield_id} yields to M{mover_pass_id}. Conflict at Segment {start_segment_index}. {action} {abs(adjustment_rp)} RP.")
            # Added a placeholder for the log. The actual log is generated in the distribution step.

    # --- 4. Distribute Adjustments to Segments ---

    # Store all logs generated during distribution
    all_segment_logs = {}

    for mover_id, adjustments in mover_adjustments.items():
        for adj in adjustments:
            start_segment_index = adj['start_segment_index']
            adjustment_rp = adj['adjustment_rp']

            # Apply the distributed RP modification, returns RP applied and the log
            rp_applied, segment_rp_log = _distribute_raw_points(df_coeff_avoidance, mover_id,
                                                                start_segment_index, adjustment_rp)

            # Store the log for this adjustment
            all_segment_logs[(mover_id, start_segment_index)] = segment_rp_log

            total_rp_adjusted += rp_applied * (-1 if adjustment_rp < 0 else 1)

            print(f"Mover {mover_id}: Distributed {rp_applied} RP (Target: {abs(adjustment_rp)}).")

            # Recalculate cumulative points after this adjustment
            df_coeff_avoidance['CumRawPointsEnd'] = df_coeff_avoidance.groupby('Mover')['NumRawPoints'].cumsum()
            df_coeff_avoidance['CumRawPointsStart'] = df_coeff_avoidance.groupby('Mover')['CumRawPointsEnd'].shift(1,
                                                                                                                   fill_value=0)

    # --- NEW: REPORT SEGMENT CHANGES ---
    if all_segment_logs:
        print("\n===== SEGMENT RAW POINT CHANGE REPORT =====")
        for (mover_id, segment_index), log in all_segment_logs.items():

            # Filter only the segments that were targeted for adjustment (1 through 8)
            # and sort them in ascending order for clarity.
            target_segments = {k: v for k, v in log.items() if k < segment_index}

            if not target_segments:
                print(f"Mover {mover_id} (Target Segment < {segment_index}): No adjustments made.")
                continue

            print(f"\n📢 Mover {mover_id} (Segments 1-{segment_index - 1} adjusted):")

            # Create a list of tuples (Segment ID, Old RP, New RP) and sort by Segment ID
            segment_data = []
            for seg_id in sorted(target_segments.keys()):
                old_rp = target_segments[seg_id]['old']
                new_rp = target_segments[seg_id].get('new', old_rp)
                segment_data.append((seg_id, old_rp, new_rp))

            # Print as a clean table
            header = ["Segment", "Old RP", "New RP", "Change"]
            print(f"{header[0]:<10} {header[1]:<8} {header[2]:<8} {header[3]}")
            print("-" * 34)
            for seg_id, old_rp, new_rp in segment_data:
                change = new_rp - old_rp
                print(f"{seg_id:<10} {old_rp:<8} {new_rp:<8} {change:+d}")
        print("=========================================")
    # -----------------------------------

    # --- 5. Save the Updated Coefficients ---
    if total_rp_adjusted != 0:
        df_coeff_avoidance['NumRawPoints'] = df_coeff_avoidance['NumRawPoints'].round().astype(int)
        df_coeff_avoidance = df_coeff_avoidance.drop(columns=['CumRawPointsEnd', 'CumRawPointsStart'])
        output_file = config['COEFF_OUTPUT_FILE_AVOIDANCE']
        df_coeff_avoidance.to_csv(output_file, index=False)
        action = "Slowdown" if total_rp_adjusted > 0 else "Speed-up"
        print(f"✅ Avoidance solution ({action} by {abs(total_rp_adjusted)} RP) saved to {output_file}.")
        return df_coeff_avoidance, True
    else:
        print("❌ Collision avoidance made no successful adjustments.")
        return None, False