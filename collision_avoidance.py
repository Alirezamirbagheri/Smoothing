# collision_avoidance.py

import numpy as np
import pandas as pd
from config import *  # The following imports are not strictly needed for this new approach but are kept for module consistency
from scipy.interpolate import splev
from collision_check import _precompute_mover_positions

# --- Configuration Constants for Avoidance ---
# These are no longer used for the calculation but define the margin and FPS
TIME_ADJUSTMENT_FACTOR = 0.05
FPS = 25  # Define the constant FPS here for use in time calculations

def resolve_collisions(results, collisions_info, config):
    """
    Applies the Time Buffering Collision Avoidance strategy:
    1. Calculates the required buffer time (duration * 1.10) for each collision event.
    2. Adds this buffer time to the 'NumRawPoints' of the segment preceding the collision
       for the yielding mover (lower Mover ID).

    Args:
        results (list): List of spline coefficient dictionaries.
        collisions_info (list): Detailed list of collision events (start/end rawpoints).
        config (dict): Configuration parameters.

    Returns:
        df_coeff_avoidance (pd.DataFrame): The new set of segment coefficients.
        collision_resolved (bool): True if adjustments were made.
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
    total_time_added_rp = 0

    # 0. Pre-calculate total length and total raw points (time) for each mover
    mover_totals = df_coeff_avoidance.groupby('Mover').agg(
        TotalLength=('Length_mm', 'sum'),
        TotalRP=('NumRawPoints', 'sum')
    )
    # Calculate the average velocity (or a metric proportional to it)
    mover_totals['AvgVelocityMetric'] = mover_totals['TotalLength'] / mover_totals['TotalRP']

    # 1. Calculate cumulative raw points for segment lookup
    df_coeff_avoidance['CumRawPointsEnd'] = df_coeff_avoidance.groupby('Mover')['NumRawPoints'].cumsum()
    # The start of a segment's raw points is the end of the previous segment
    df_coeff_avoidance['CumRawPointsStart'] = df_coeff_avoidance.groupby('Mover')['CumRawPointsEnd'].shift(1,
                                                                                                           fill_value=0)

    print("\n--- Applying Time Buffering Strategy ---")

    # Group adjustments needed by mover to handle multiple collisions
    mover_adjustments = {}  # {mover_id: [{'rp_index': N, 'buffer_rp': M}, ...]}

    for col_pair in collisions_info:
        # Determine who yields (lower ID yields) and who passes (higher ID passes)
        mover_A_id = col_pair['Mover_A']
        mover_B_id = col_pair['Mover_B']

        # --- NEW DECISION LOGIC: Faster Mover Yields ---
        vel_A = mover_totals.loc[mover_A_id, 'AvgVelocityMetric']
        vel_B = mover_totals.loc[mover_B_id, 'AvgVelocityMetric']

        if vel_A > vel_B:
            mover_yield_id = mover_A_id
            mover_pass_id = mover_B_id
        else:
            # If velocities are equal, or B is faster, B yields (or use a tie-breaker like lower ID)
            # Choosing B yields if V_A <= V_B, to ensure one mover is always chosen.
            mover_yield_id = mover_B_id
            mover_pass_id = mover_A_id
        # -----------------------------------------------

        if mover_yield_id not in mover_adjustments:
            mover_adjustments[mover_yield_id] = []
        for event in col_pair['Collision_Events']:
            duration_rp = event['duration_rp']
            rp_placement_index = event['start_rawpoint']

            # 1. Calculation of adjustment_rp (which we were calling buffer_rp)
            if TIME_ADJUSTMENT_FACTOR >= 1.0:
                adjustment_rp = int(duration_rp * TIME_ADJUSTMENT_FACTOR)
                print(f"Factor >= 1.0. Applying buffer {adjustment_rp} RP.")
            else:
                adjustment_rp = int(duration_rp * (1 / TIME_ADJUSTMENT_FACTOR))
                print(f"Factor < 1.0. Applying larger buffer {adjustment_rp} RP.")

            # --- The variable is now adjustment_rp, not buffer_rp! ---

            # 2. Store the adjustment
            mover_adjustments[mover_yield_id].append({
                'rp_index': rp_placement_index,
                'buffer_rp': adjustment_rp
            })

            # 3. Use the calculated values in a print statement *inside* the event loop
            # Note: We use adjustment_rp here, as that is the local variable holding the buffer size.
            vel_A = mover_totals.loc[mover_A_id, 'AvgVelocityMetric']
            vel_B = mover_totals.loc[mover_B_id, 'AvgVelocityMetric']

            print(
                f"M{mover_yield_id} (V={vel_A if mover_yield_id == mover_A_id else vel_B:.2f}) yields to M{mover_pass_id} (V={vel_B if mover_yield_id == mover_A_id else vel_A:.2f}). Buffering {adjustment_rp} RP.")
         # --- 4. Apply Adjustments to Segments ---

    # NOTE: The logic for merging close buffers (as requested) is complex and heavily relies
    # on detailed segment boundaries. For initial implementation, we apply each buffer
    # individually to the segment immediately preceding the collision start.

    for mover_id, adjustments in mover_adjustments.items():
        # Get the coefficient slice for the current mover
        df_mover_slice = df_coeff_avoidance[df_coeff_avoidance['Mover'] == mover_id]

        for adjustment in adjustments:
            rp_placement_index = adjustment['rp_index']
            buffer_rp = adjustment['buffer_rp']

            # Find the segment index that is active immediately BEFORE the collision point.
            # We look for the last segment that ends *before* or *at* the placement index.
            # Safety margin: rp_placement_index > 0 guaranteed by collision check.

            # Select segments that contain or end just before the placement index.
            # We target the row index where CumRawPointsEnd is >= rp_placement_index (segment containing)
            # and take the row *before* it, or the containing segment itself if we can't find the preceding one.

            # Find the segment that covers the raw point immediately preceding the collision start
            target_rp_index = max(0, rp_placement_index - 1)

            target_segment_rows = df_mover_slice[
                (df_mover_slice['CumRawPointsStart'] <= target_rp_index) &
                (df_mover_slice['CumRawPointsEnd'] > target_rp_index)
                ]

            if not target_segment_rows.empty:
                # Get the global index of the segment row to modify
                target_global_index = target_segment_rows.index[0]

                # 5. Add the buffer time to the segment's NumRawPoints
                df_coeff_avoidance.loc[target_global_index, 'NumRawPoints'] += buffer_rp
                total_time_added_rp += buffer_rp

                print(
                    f"Mover {mover_id}: Added {buffer_rp} RP to Segment {target_segment_rows['Segment'].iloc[0]} (ends at RP {df_coeff_avoidance.loc[target_global_index, 'CumRawPointsEnd']}).")

                # IMPORTANT: Since we modified the raw points of a segment, the cumulative
                # indices for all following segments of this mover are now WRONG.
                # We must recalculate the cumulative points after this adjustment.
                df_coeff_avoidance['CumRawPointsEnd'] = df_coeff_avoidance.groupby('Mover')['NumRawPoints'].cumsum()
                df_coeff_avoidance['CumRawPointsStart'] = df_coeff_avoidance.groupby('Mover')['CumRawPointsEnd'].shift(
                    1, fill_value=0)
            else:
                print(f"Warning: Could not find segment for Mover {mover_id} at RP {rp_placement_index}.")
    df_coeff_avoidance.loc[target_global_index, 'NumRawPoints'] += adjustment_rp
    # --- 6. Save the Updated Coefficients ---
    if total_time_added_rp > 0:
        # Final cleanup before saving: ensure NumRawPoints are integers
        df_coeff_avoidance['NumRawPoints'] = df_coeff_avoidance['NumRawPoints'].round().astype(int)
        output_file = config['COEFF_OUTPUT_FILE_AVOIDANCE']

        # Remove the temporary cumulative columns before saving
        df_coeff_avoidance = df_coeff_avoidance.drop(columns=['CumRawPointsEnd', 'CumRawPointsStart'])

        df_coeff_avoidance.to_csv(output_file, index=False)
        print(f"✅ Avoidance solution (Time Buffering) saved to {output_file}.")
        return df_coeff_avoidance, True
    else:
        print("❌ Collision avoidance made no successful adjustments.")
        return None, False