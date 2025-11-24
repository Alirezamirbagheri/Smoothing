# main.py
import os
import numpy as np
import pandas as pd
from config import *  # Assumes SAFETY_STEP_START and MAX_ITERATIONS are imported here
from data_loader import load_raw_data
from spline_fit import process_all_paths
from parameter_extractor import (
    generate_final_segment_coeffs, save_smoothed_points_csv
)
from animations import plot_static_paths, animate_single_movers, animate_global_movers
from collision_check import check_mover_collisions
from collision_avoidance import resolve_collisions

# Define the new file path constant
COEFF_OUTPUT_FILE_AVOIDANCE = 'Results/ParametricSplineCoeff_Avoidance.csv'
GLOBAL_ANIMATION_FILE_COLLIDING = 'Results/GlobalMoverAnimation_Colliding.gif'
GLOBAL_ANIMATION_FILE_RESOLVED = 'Results/GlobalMoverAnimation_Resolved.gif'


# --- HELPER FUNCTIONS ---

def _calculate_total_rp_before(mover_id, target_time_sec, df_coeff, config):
    """
    Finds the total cumulative NumRawPoints for mover_id up to the target_time_sec
    using the provided DataFrame (df_coeff). Used for initial collision time reporting.
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


def _calculate_total_path_rp(mover_id, df_coeff):
    """Calculates the total NumRawPoints for the entire path of the mover (used for post-resolution path length)."""
    if df_coeff.empty or 'Mover' not in df_coeff.columns:
        return 0
    return df_coeff[df_coeff['Mover'] == mover_id]['NumRawPoints'].sum()


# --- MAIN EXECUTION ---

def main():
    """The main execution function, orchestrating the simplified modules and the optimization loop."""

    # Define a default FPS here if it's not in config.py
    FPS = 25

    # Retrieve TIME_ADJUSTMENT_FACTOR from config if possible
    try:
        from config import TIME_ADJUSTMENT_FACTOR
    except ImportError:
        TIME_ADJUSTMENT_FACTOR = 1.0

    # 1. Configuration Setup
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
        'COEFF_OUTPUT_FILE_ANIMATION': COEFF_OUTPUT_FILE_UPDATED,  # Initial path for first check

        'GLOBAL_ANIMATION_FILE_COLLIDING': GLOBAL_ANIMATION_FILE_COLLIDING,
        'GLOBAL_ANIMATION_FILE_RESOLVED': GLOBAL_ANIMATION_FILE_RESOLVED,

        'FPS': FPS,
        'TIME_ADJUSTMENT_FACTOR': TIME_ADJUSTMENT_FACTOR
    }

    # Load original dataframes
    try:
        df_coeff_orig = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
    except FileNotFoundError:
        print("Error: Initial coefficient file not found. Exiting.")
        return

    # Load and process paths
    df_raw, num_movers = load_raw_data(config['INPUT_FILE'])
    results = process_all_paths(df_raw, num_movers, config)

    # 2. Initial Safety Check & Report Data Collection
    initial_collisions_info = check_mover_collisions(results, config)

    report_data = {}
    df_coeff_avoidance = None
    resolved_in_loop = False
    best_safety_factor = 0.0

    # Store initial metrics if collisions are found
    if initial_collisions_info:
        first_pair = initial_collisions_info[0]
        first_event = first_pair['Collision_Events'][0]

        report_data['Mover_A'] = first_pair['Mover_A']
        report_data['Mover_B'] = first_pair['Mover_B']
        report_data['T_start_orig'] = first_event['start_time_sec']
        report_data['Duration_RP_orig'] = first_event['duration_rp']
        report_data['Start_Segment_Orig'] = first_event.get('start_segment_index', 'N/A')

        report_data['Total_RP_A_orig'] = _calculate_total_rp_before(
            report_data['Mover_A'], report_data['T_start_orig'], df_coeff_orig, config
        )
        report_data['Total_RP_B_orig'] = _calculate_total_rp_before(
            report_data['Mover_B'], report_data['T_start_orig'], df_coeff_orig, config
        )

    # 3. Collision Avoidance Optimization Loop
    current_collisions_info = initial_collisions_info

    if initial_collisions_info:
        print("\n======== Starting Safety Factor Optimization Loop ========")

        # Define search directions (1.0 for positive/faster, -1.0 for negative/slower)
        search_directions = [1.0, -1.0]
        best_avoidance_df = None
        min_persistent_duration = initial_collisions_info[0]['Collision_Events'][0]['duration_rp']
        last_s_f = 0.0  # Tracks the last safety factor attempted

        for direction in search_directions:

            # If resolution was already achieved in the positive direction, stop.
            if resolved_in_loop:
                break

            for iteration in range(1, MAX_ITERATIONS + 1):

                # Calculate the safety factor step
                current_safety_factor = SAFETY_STEP_START * iteration * direction

                # Pass the time adjustment factor to the config
                config['TIME_ADJUSTMENT_FACTOR'] = current_safety_factor
                last_s_f = current_safety_factor  # Update last attempted factor

                print(f"\n--- Running Iteration {iteration} (Direction: {direction}) ---")
                print(f"Attempting Safety Factor (S_f): {current_safety_factor:.3f}")

                # A. Resolve collisions (creates new avoidance CSV and returns DF)
                df_temp_avoidance, avoidance_success = resolve_collisions(
                    results, initial_collisions_info, config
                )

                if not avoidance_success:
                    print(f"🛑 Avoidance failed for S_f={current_safety_factor:.3f}. Stopping search in this direction.")
                    break

                # B. Check the safety of the NEWLY CREATED path
                config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_AVOIDANCE']
                current_collisions_info = check_mover_collisions(results, config)

                if not current_collisions_info:
                    # ✅ Collision resolved! Stop and save the result.
                    resolved_in_loop = True
                    df_coeff_avoidance = df_temp_avoidance
                    best_safety_factor = current_safety_factor
                    print(f"✅ Success! Collisions resolved with S_f={best_safety_factor:.3f}.")
                    break

                else:
                    # ⚠️ Collision persists, check if improvement was made
                    persistent_duration = current_collisions_info[0]['Collision_Events'][0]['duration_rp']

                    if persistent_duration < min_persistent_duration:
                        # Improvement found: save this DF as the current 'best' persistent path
                        min_persistent_duration = persistent_duration
                        best_avoidance_df = df_temp_avoidance
                        print(f"⏱️ Collision persists ({persistent_duration} RP). Improvement found.")
                    else:
                        print(f"⏳ Collision persists ({persistent_duration} RP). No further improvement.")

            if resolved_in_loop:
                break  # Break outer loop if resolution was found

        # Final state check after the loop
        if resolved_in_loop:
            df_final_report = df_coeff_avoidance
            print(f"\nOptimization Finished: ✅ Resolution achieved with S_f={best_safety_factor:.3f}.")
            # current_collisions_info is empty here
        elif best_avoidance_df is not None:
            # Use the most improved path found
            df_final_report = best_avoidance_df
            print(f"\nOptimization Finished: ⚠️ No resolution found. Using most improved path.")
            # Re-run collision check on the best path found to populate current_collisions_info
            config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_AVOIDANCE']
            current_collisions_info = check_mover_collisions(results, config)
        else:
            # No resolution and no improvement found
            df_final_report = df_coeff_orig
            print("\nOptimization Finished: 🛑 No improvement or resolution found. Using original path status.")
            current_collisions_info = initial_collisions_info

    else:
        # No initial collisions
        resolved_in_loop = True
        df_final_report = df_coeff_orig
        print("✅ No collisions detected initially. No avoidance needed.")

    # --- 4. Collect Final Report Metrics ---
    if initial_collisions_info:

        if not current_collisions_info and resolved_in_loop:
            report_data['T_start_new'] = 'N/A (Resolved)'
            report_data['Duration_RP_new'] = 0
            report_data['Start_Segment_New'] = 'N/A (Resolved)'

        elif current_collisions_info:
            final_event = current_collisions_info[0]['Collision_Events'][0]
            report_data['T_start_new'] = final_event['start_time_sec']
            report_data['Duration_RP_new'] = final_event['duration_rp']
            report_data['Start_Segment_New'] = final_event.get('start_segment_index', 'N/A')
        else:
            # Fallback if current_collisions_info is empty but not marked resolved
            report_data['T_start_new'] = 'N/A (Resolved - Fallback)'
            report_data['Duration_RP_new'] = 0
            report_data['Start_Segment_New'] = 'N/A (Resolved - Fallback)'

        # CORRECTED: Report the TOTAL PATH RP for the modified path
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

        print(f"\n--- BEFORE RESOLUTION (Colliding Path) ---")
        print(f"  First Collision Start Time: {report_data['T_start_orig']:.3f} s")
        print(f"  Collision Start Segment (Approximate): {report_data['Start_Segment_Orig']}")
        print(f"  Collision Duration: {report_data['Duration_RP_orig']} Raw Points")
        print(f"  Mover {report_data['Mover_A']} Total RP BEFORE conflict start: {report_data['Total_RP_A_orig']}")
        print(f"  Mover {report_data['Mover_B']} Total RP BEFORE conflict start: {report_data['Total_RP_B_orig']}")

        print("\n--- AFTER RESOLUTION (Modified Path) ---")
        if report_data['Duration_RP_new'] == 0:
            print("  Collision Status: ✅ Successfully Resolved.")
            print(f"  Optimal Safety Factor Applied (S_f): {best_safety_factor:.3f}")
            print(f"  Old Conflict Point: {report_data['T_start_orig']:.3f} s (Now collision-free)")
        else:
            print(f"  Collision Status: ⚠️ Persists (Duration: {report_data['Duration_RP_new']} RP)")
            print(f"  Safety Factor Used (S_f): {last_s_f:.3f}")
            print(f"  New Collision Start Time: {report_data['T_start_new']:.3f} s")
            print(f"  Collision Start Segment (New): {report_data['Start_Segment_New']}")

        # These values now reflect the total path length
        print(f"  Mover {report_data['Mover_A']} Total RP (Full Path): {report_data['Total_RP_A_new']}")
        print(f"  Mover {report_data['Mover_B']} Total RP (Full Path): {report_data['Total_RP_B_new']}")
        print("=======================================================")

    # --- 5. Visualization (Dual Animation) ---

    plot_static_paths(results, num_movers)
    animate_single_movers(results, config)

    # A. Generate the COLLIDING Animation (Always uses the original file)
    if initial_collisions_info:
        print("\n🎬 Generating COLLIDING Animation (Original Path)")
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_UPDATED']
        config['GLOBAL_ANIMATION_FILE'] = config['GLOBAL_ANIMATION_FILE_COLLIDING']
        animate_global_movers(results, num_movers, config)

    # B. Generate the RESOLVED Animation (Uses the final, best avoidance file)
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