# main.py
import os
from config import *
from data_loader import load_raw_data
from spline_fit import process_all_paths
from parameter_extractor import (
    generate_final_segment_coeffs, save_smoothed_points_csv
)
from animations import plot_static_paths, animate_single_movers, animate_global_movers
from collision_check import check_mover_collisions
from collision_avoidance import resolve_collisions
import numpy as np
import pandas as pd

# Define the new file path constant
COEFF_OUTPUT_FILE_AVOIDANCE = 'Results/ParametricSplineCoeff_Avoidance.csv'
GLOBAL_ANIMATION_FILE_COLLIDING = 'Results/GlobalMoverAnimation_Colliding.gif'
GLOBAL_ANIMATION_FILE_RESOLVED = 'Results/GlobalMoverAnimation_Resolved.gif'


# --- MODIFIED HELPER FUNCTION for accurate RP calculation (Now accepts DataFrame) ---
def _calculate_total_rp_before(mover_id, target_time_sec, df_coeff, config):
    """
    Finds the total cumulative NumRawPoints for mover_id up to the target_time_sec
    using the provided DataFrame (df_coeff).
    """
    try:
        df_mover = df_coeff[df_coeff['Mover'] == mover_id].copy()

        # Ensure FPS is available, default to 25
        fps = config.get('FPS', 25)

        # 1. Calculate cumulative RP and Time for segment ends
        df_mover['CumRawPointsEnd'] = df_mover['NumRawPoints'].cumsum()
        df_mover['CumRawPointsStart'] = df_mover['CumRawPointsEnd'].shift(1, fill_value=0)
        df_mover['T_cum_start'] = df_mover['CumRawPointsStart'] / fps

        # 2. Find the segment that CONTAINS the target time
        # Check T_cum_start against the target time and the end time against the target time
        containing_segment_row = df_mover[
            (df_mover['T_cum_start'] <= target_time_sec) &
            (df_mover['T_cum_start'] + df_mover['NumRawPoints'] / fps > target_time_sec)
            ]

        # 3. Calculate total RP before target_time_sec
        if containing_segment_row.empty:
            if df_mover.empty or target_time_sec > df_mover['T_cum_start'].iloc[-1] + df_mover['NumRawPoints'].iloc[
                -1] / fps:
                total_rp = df_mover['NumRawPoints'].sum()
            else:
                total_rp = 0
        else:
            # Case: Target time is within a segment
            row = containing_segment_row.iloc[0]

            rp_before_segment = row['CumRawPointsStart']
            time_in_segment = target_time_sec - row['T_cum_start']

            # RP accumulated within the containing segment
            rp_in_segment = max(0, time_in_segment * fps)

            total_rp = rp_before_segment + rp_in_segment

        return int(round(total_rp))

    except Exception as e:
        # print(f"Error calculating RP before for M{mover_id}: {e}")
        return 0


# --- NEW HELPER: Calculates total path RP (for post-resolution report) ---
def _calculate_total_path_rp(mover_id, df_coeff):
    """Calculates the total NumRawPoints for the entire path of the mover."""
    # Ensure the DataFrame is not empty and contains the mover
    if df_coeff.empty or 'Mover' not in df_coeff.columns:
        return 0
    return df_coeff[df_coeff['Mover'] == mover_id]['NumRawPoints'].sum()


# -------------------------------------------------------------------------


def main():
    """The main execution function, orchestrating the simplified modules."""

    # Define a default FPS here if it's not in config.py
    FPS = 25

    # Retrieve TIME_ADJUSTMENT_FACTOR from config if possible
    try:
        from config import TIME_ADJUSTMENT_FACTOR
    except ImportError:
        TIME_ADJUSTMENT_FACTOR = 1.0

    config = {
        'RDP_TOL': RDP_TOL, 'SMOOTHING_FACTOR_INIT': SMOOTHING_FACTOR_INIT,
        'MAX_POINTS': MAX_POINTS, 'RESAMPLE_POINTS': RESAMPLE_POINTS,
        'K_ORDER': K_ORDER, 'MAX_DEV_ALLOWED': MAX_DEV_ALLOWED,
        'MOVER_SIZE': MOVER_SIZE, 'VELOCITY': VELOCITY,
        'TABLE_LENGTH_MM': TABLE_LENGTH_MM,
        'TABLE_WIDTH_MM': TABLE_WIDTH_MM,
        'PLOT_PADDING_MM': PLOT_PADDING_MM,
        'ANIMATE': ANIMATE, 'ANIMATE_ALL': ANIMATE_ALL,
        'INPUT_FILE': INPUT_FILE, 'SMOOTHED_POINTS_FILE': SMOOTHED_POINTS_FILE,
        'COEFF_OUTPUT_FILE_UPDATED': COEFF_OUTPUT_FILE_UPDATED,
        'COEFF_OUTPUT_FILE_AVOIDANCE': COEFF_OUTPUT_FILE_AVOIDANCE,
        # This will point to the original file for the first check
        'COEFF_OUTPUT_FILE_ANIMATION': COEFF_OUTPUT_FILE_UPDATED,

        'GLOBAL_ANIMATION_FILE_COLLIDING': GLOBAL_ANIMATION_FILE_COLLIDING,
        'GLOBAL_ANIMATION_FILE_RESOLVED': GLOBAL_ANIMATION_FILE_RESOLVED,

        'FPS': FPS,
        'TIME_ADJUSTMENT_FACTOR': TIME_ADJUSTMENT_FACTOR
    }

    # --- Load original coefficients once for initial check and report ---
    try:
        # Load the original DF for the initial check metrics
        df_coeff_orig = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
    except FileNotFoundError:
        print("Error: Initial coefficient file not found. Exiting.")
        return

    # --- STEP 1 & 2: Load data and path fitting ---
    df_raw, num_movers = load_raw_data(config['INPUT_FILE'])
    results = process_all_paths(df_raw, num_movers, config)

    # --- STEP 4: Initial Safety Check & Report Data Collection ---
    initial_collisions_info = check_mover_collisions(results, config)

    report_data = {}  # Dictionary to store report metrics
    df_coeff_avoidance = None  # Initialize variable for the modified path DF

    if initial_collisions_info:
        first_pair = initial_collisions_info[0]
        first_event = first_pair['Collision_Events'][0]

        # --- Store Initial Metrics ---
        report_data['Mover_A'] = first_pair['Mover_A']
        report_data['Mover_B'] = first_pair['Mover_B']
        report_data['T_start_orig'] = first_event['start_time_sec']
        report_data['Duration_RP_orig'] = first_event['duration_rp']
        report_data['Start_Segment_Orig'] = first_event.get('start_segment_index', 'N/A')

        # Calculate initial Total RP before collision start time using the loaded DF
        report_data['Total_RP_A_orig'] = _calculate_total_rp_before(
            report_data['Mover_A'], report_data['T_start_orig'], df_coeff_orig, config
        )
        report_data['Total_RP_B_orig'] = _calculate_total_rp_before(
            report_data['Mover_B'], report_data['T_start_orig'], df_coeff_orig, config
        )

    # --- STEP 5: Collision Avoidance (ONE-SHOT TIME BUFFERING) ---
    current_collisions_info = initial_collisions_info
    resolved = False

    if initial_collisions_info:

        print("\n======== Starting Collision Avoidance (One-Shot Attempt) ========")

        # 1. Resolve collisions (saves to COEFF_OUTPUT_FILE_AVOIDANCE AND returns the DF)
        df_coeff_avoidance, avoidance_success = resolve_collisions(results, current_collisions_info, config)

        if not avoidance_success:
            print("🛑 Avoidance failed to find a valid solution in the one-shot attempt.")

        # 2. **CRITICAL FIX:** Ensure the animation path is set to the avoidance file.
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_AVOIDANCE']

        # 3. Check the safety of the NEWLY CREATED path (Final check)
        current_collisions_info = check_mover_collisions(results, config)

        if not current_collisions_info:
            resolved = True
            print("✅ All collisions successfully resolved in one attempt!")
        else:
            resolved = False
            print("⚠️ Collisions still persist after the one-shot adjustment.")

        print(f"\n======== Avoidance Attempt Finished (Resolved: {resolved}) ========")
    else:
        resolved = True
        print("✅ No collisions detected initially. No avoidance needed.")

    # --- Collect Final Report Metrics ---
    if initial_collisions_info:

        # Use the MODIFIED DataFrame (df_coeff_avoidance) for ALL NEW metrics
        df_final_report = df_coeff_avoidance if df_coeff_avoidance is not None else df_coeff_orig

        if not current_collisions_info:
            report_data['T_start_new'] = 'N/A (Resolved)'
            report_data['Duration_RP_new'] = 0
            report_data['Start_Segment_New'] = 'N/A (Resolved)'
        else:
            final_event = current_collisions_info[0]['Collision_Events'][0]
            report_data['T_start_new'] = final_event['start_time_sec']
            report_data['Duration_RP_new'] = final_event['duration_rp']
            report_data['Start_Segment_New'] = final_event.get('start_segment_index', 'N/A')

        # 🚨 FINAL CRITICAL CHANGE: Report the TOTAL PATH RP, not RP up to the old time.
        report_data['Total_RP_A_new'] = _calculate_total_path_rp(
            report_data['Mover_A'], df_final_report
        )
        report_data['Total_RP_B_new'] = _calculate_total_path_rp(
            report_data['Mover_B'], df_final_report
        )

        # --- Print Final Report ---
        print("\n\n=======================================================")
        print(f"REPORT: Collision Resolution for Mover {report_data['Mover_A']} vs Mover {report_data['Mover_B']}")
        print("=======================================================")

        print("\n--- BEFORE RESOLUTION (Colliding Path) ---")
        print(f"  First Collision Start Time: {report_data['T_start_orig']:.3f} s")
        print(f"  Collision Start Segment (Approximate): {report_data['Start_Segment_Orig']}")
        print(f"  Collision Duration: {report_data['Duration_RP_orig']} Raw Points")
        print(
            f"  Mover {report_data['Mover_A']} Total RP BEFORE conflict start: {report_data['Total_RP_A_orig']} (Full Path RP: {df_coeff_orig[df_coeff_orig['Mover'] == report_data['Mover_A']]['NumRawPoints'].sum()})")
        print(
            f"  Mover {report_data['Mover_B']} Total RP BEFORE conflict start: {report_data['Total_RP_B_orig']} (Full Path RP: {df_coeff_orig[df_coeff_orig['Mover'] == report_data['Mover_B']]['NumRawPoints'].sum()})")

        print("\n--- AFTER RESOLUTION (Modified Path) ---")
        if report_data['Duration_RP_new'] == 0:
            print("  Collision Status: ✅ Successfully Resolved.")
            print(f"  Old Conflict Point: {report_data['T_start_orig']:.3f} s (Now collision-free)")
        else:
            print(f"  Collision Status: ⚠️ Persists (Duration: {report_data['Duration_RP_new']} RP)")
            print(f"  New Collision Start Time: {report_data['T_start_new']:.3f} s")
            print(f"  Collision Start Segment (New): {report_data['Start_Segment_New']}")

        # These values now reflect the total path length (e.g., 97 RP vs 108 RP)
        print(f"  Mover {report_data['Mover_A']} Total RP (Full Path): {report_data['Total_RP_A_new']}")
        print(f"  Mover {report_data['Mover_B']} Total RP (Full Path): {report_data['Total_RP_B_new']}")
        print("=======================================================")

    # --- STEP 7: Visualization (Dual Animation) ---
    plot_static_paths(results, num_movers)
    animate_single_movers(results, config)

    # A. Generate the COLLIDING Animation
    if initial_collisions_info:
        print("\n🎬 Generating COLLIDING Animation (Original Path)")
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_UPDATED']
        config['GLOBAL_ANIMATION_FILE'] = config['GLOBAL_ANIMATION_FILE_COLLIDING']
        animate_global_movers(results, num_movers, config)

    # B. Generate the RESOLVED Animation
    avoidance_file_exists = os.path.exists(config['COEFF_OUTPUT_FILE_AVOIDANCE'])

    if avoidance_file_exists and initial_collisions_info:
        print("\n🎬 Generating RESOLVED Animation (AVOIDANCE MODIFIED PATH)")
        # This config line ensures the animation function uses the avoidance file
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_AVOIDANCE']
        config['GLOBAL_ANIMATION_FILE'] = config['GLOBAL_ANIMATION_FILE_RESOLVED']
        animate_global_movers(results, num_movers, config)
    elif avoidance_file_exists and not initial_collisions_info:
        print("No resolved animation needed; original path is collision-free.")
    else:
        print("\n⚠️ No resolved path animation generated (Avoidance function failed to save the coefficient file).")


if __name__ == "__main__":
    main()