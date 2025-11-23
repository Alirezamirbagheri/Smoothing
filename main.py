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


# --- NEW HELPER FUNCTION (REQUIRES IMPLEMENTATION ELSEWHERE) ---
def _calculate_total_rp_before(mover_id, target_time_sec, config_file_key, config):
    """
    CONCEPTUAL FUNCTION: Loads the coefficient file specified by config_file_key,
    finds the total cumulative NumRawPoints for mover_id up to the target_time_sec.

    NOTE: You must implement the actual body of this function, potentially in a
    utility module, to load the CSV, calculate cumulative RP, and find the value.
    Returns 0 if file not found or calculation fails.
    """
    try:
        df = pd.read_csv(config[config_file_key])
        df_mover = df[df['Mover'] == mover_id].copy()

        # Calculate approximate original time for each segment end (based on RP count)
        # Use config.get('FPS', 25) for safety
        df_mover['T_cum_end'] = df_mover['NumRawPoints'].cumsum() / config.get('FPS', 25)

        # Find the last segment that ends before or at the target time
        # This simple logic is a strong approximation, assuming the target time is reached
        preceding_segments = df_mover[df_mover['T_cum_end'] <= target_time_sec]

        if preceding_segments.empty:
            # If target_time_sec is within the first segment, RP before is 0.
            total_rp = 0
        else:
            total_rp = preceding_segments['NumRawPoints'].sum()

        return int(total_rp)

    except Exception as e:
        # print(f"Error calculating RP before for M{mover_id}: {e}")
        return 0


def main():
    """The main execution function, orchestrating the simplified modules."""

    # Define a default FPS here if it's not in config.py
    FPS = 25

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
        # Ensure COEFF_OUTPUT_FILE_AVOIDANCE is in config
        'COEFF_OUTPUT_FILE_AVOIDANCE': COEFF_OUTPUT_FILE_AVOIDANCE,
        # The dynamic key for the animation module (gets overwritten in the loop)
        'COEFF_OUTPUT_FILE_ANIMATION': COEFF_OUTPUT_FILE_UPDATED,

        # Keys for dual animation output files
        'GLOBAL_ANIMATION_FILE_COLLIDING': GLOBAL_ANIMATION_FILE_COLLIDING,
        'GLOBAL_ANIMATION_FILE_RESOLVED': GLOBAL_ANIMATION_FILE_RESOLVED,

        'FPS': FPS  # Add FPS to config for calculation
    }

    # --- STEP 1 & 2: Load data and path fitting ---
    df_raw, num_movers = load_raw_data(config['INPUT_FILE'])
    results = process_all_paths(df_raw, num_movers, config)

    # --- STEP 3: Parameter Extraction & Export ---
    save_smoothed_points_csv(results, config['SMOOTHED_POINTS_FILE'], config['RESAMPLE_POINTS'])
    generate_final_segment_coeffs(results, config['COEFF_OUTPUT_FILE_UPDATED'])

    # --- STEP 4: Initial Safety Check & Report Data Collection ---
    config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_UPDATED']
    initial_collisions_info = check_mover_collisions(results, config)

    report_data = {}  # Dictionary to store report metrics

    if initial_collisions_info:
        first_pair = initial_collisions_info[0]
        first_event = first_pair['Collision_Events'][0]

        # --- Store Initial Metrics ---
        report_data['Mover_A'] = first_pair['Mover_A']
        report_data['Mover_B'] = first_pair['Mover_B']
        report_data['T_start_orig'] = first_event['start_time_sec']
        report_data['Duration_RP_orig'] = first_event['duration_rp']

        # Calculate initial Total RP before collision start time
        report_data['Total_RP_A_orig'] = _calculate_total_rp_before(
            report_data['Mover_A'], report_data['T_start_orig'], 'COEFF_OUTPUT_FILE_UPDATED', config
        )
        report_data['Total_RP_B_orig'] = _calculate_total_rp_before(
            report_data['Mover_B'], report_data['T_start_orig'], 'COEFF_OUTPUT_FILE_UPDATED', config
        )

    # --- STEP 5: Collision Avoidance (ONE-SHOT TIME BUFFERING) ---
    resolved = False
    current_collisions_info = initial_collisions_info  # Start with the initial data

    if initial_collisions_info:

        print("\n======== Starting Collision Avoidance (One-Shot Attempt) ========")

        # 1. Resolve collisions (saves to COEFF_OUTPUT_FILE_AVOIDANCE)
        df_coeff_avoidance, avoidance_success = resolve_collisions(results, current_collisions_info, config)

        if not avoidance_success:
            print("🛑 Avoidance failed to find a valid solution in the one-shot attempt.")

        # 2. Use the new avoidance file for the check
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

        if not current_collisions_info:
            # Collision resolved successfully!
            report_data['T_start_new'] = 'N/A (Resolved)'
            report_data['Duration_RP_new'] = 0
        else:
            # Collision persists (Use the first remaining event)
            final_event = current_collisions_info[0]['Collision_Events'][0]
            report_data['T_start_new'] = final_event['start_time_sec']
            report_data['Duration_RP_new'] = final_event['duration_rp']

        # Calculate final Total RP before collision start time (using original T_start as reference)
        report_data['Total_RP_A_new'] = _calculate_total_rp_before(
            report_data['Mover_A'], report_data['T_start_orig'], 'COEFF_OUTPUT_FILE_AVOIDANCE', config
        )
        report_data['Total_RP_B_new'] = _calculate_total_rp_before(
            report_data['Mover_B'], report_data['T_start_orig'], 'COEFF_OUTPUT_FILE_AVOIDANCE', config
        )

        # --- Print Final Report ---
        print("\n\n=======================================================")
        print(f"REPORT: Collision Resolution for Mover {report_data['Mover_A']} vs Mover {report_data['Mover_B']}")
        print("=======================================================")

        print("\n--- BEFORE RESOLUTION (Colliding Path) ---")
        print(f"  First Collision Start Time: {report_data['T_start_orig']:.3f} s")
        print(f"  Collision Duration: {report_data['Duration_RP_orig']} Raw Points")
        print(f"  Mover {report_data['Mover_A']} Total RP BEFORE conflict start: {report_data['Total_RP_A_orig']}")
        print(f"  Mover {report_data['Mover_B']} Total RP BEFORE conflict start: {report_data['Total_RP_B_orig']}")

        print("\n--- AFTER RESOLUTION (Modified Path) ---")
        if report_data['Duration_RP_new'] == 0:
            print("  Collision Status: ✅ Successfully Resolved.")
            print(f"  New Collision Start Time: {report_data['T_start_orig']:.3f} s (Referenced to original time)")
        else:
            print(f"  Collision Status: ⚠️ Persists (Duration: {report_data['Duration_RP_new']} RP)")
            print(f"  New Collision Start Time: {report_data['T_start_new']:.3f} s")

        print(f"  Mover {report_data['Mover_A']} Total RP BEFORE conflict start: {report_data['Total_RP_A_new']}")
        print(f"  Mover {report_data['Mover_B']} Total RP BEFORE conflict start: {report_data['Total_RP_B_new']}")
        print("=======================================================")

    # --- STEP 7: Visualization (Dual Animation) ---

    plot_static_paths(results, num_movers)
    animate_single_movers(results, config)

    # A. Generate the COLLIDING Animation (if collisions were initially found)
    if initial_collisions_info:
        print("\n🎬 Generating COLLIDING Animation (Original Path)")
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_UPDATED']
        config['GLOBAL_ANIMATION_FILE'] = config['GLOBAL_ANIMATION_FILE_COLLIDING']
        animate_global_movers(results, num_movers, config)

        # B. Generate the RESOLVED Animation (Run if file exists, regardless of final 'resolved' bool)

    # Check if the avoidance file exists, indicating that resolve_collisions successfully saved a file.
    avoidance_file_exists = os.path.exists(config['COEFF_OUTPUT_FILE_AVOIDANCE'])

    if avoidance_file_exists and initial_collisions_info:
        # Use the modified path for animation, even if final resolution failed, as requested.
        print("\n🎬 Generating RESOLVED Animation (AVOIDANCE MODIFIED PATH - regardless of resolution success)")
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_AVOIDANCE']
        config['GLOBAL_ANIMATION_FILE'] = config['GLOBAL_ANIMATION_FILE_RESOLVED']
        animate_global_movers(results, num_movers, config)
    elif avoidance_file_exists and not initial_collisions_info:
        print("No resolved animation needed; original path is collision-free.")
    else:
        print("\n⚠️ No resolved path animation generated (Avoidance function failed to save the coefficient file).")


if __name__ == "__main__":
    main()