# main.py

from config import *
from data_loader import load_raw_data
from spline_fit import process_all_paths
from parameter_extractor import (
    # Only import the new final generator and the smoothed points exporter
    generate_final_segment_coeffs, save_smoothed_points_csv
)
from animations import plot_static_paths, animate_single_movers, animate_global_movers
from collision_check import check_mover_collisions
from collision_avoidance import resolve_collisions, save_avoidance_points_csv # NEW IMPORT
import numpy as np
import pandas as pd

# Define the new file path constant
COEFF_OUTPUT_FILE_AVOIDANCE = 'Results/ParametricSplineCoeff_Avoidance.csv'

def main():
    """The main execution function, orchestrating the simplified modules."""

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
        # COEFF_OUTPUT_FILE_OLD is still needed in config but ignored here
        'COEFF_OUTPUT_FILE_UPDATED': COEFF_OUTPUT_FILE_UPDATED,
        # RAW_SEG_MAP_FILE is still needed in config but ignored here
        'GLOBAL_ANIMATION_FILE': GLOBAL_ANIMATION_FILE,
        # --- NEW KEY REQUIRED BY COLLISION_AVOIDANCE.PY ---
        'COEFF_OUTPUT_FILE_AVOIDANCE': COEFF_OUTPUT_FILE_AVOIDANCE
        # --------------------------------------------------
    }
    # --- STEP 1: Load data ---
    df_raw, num_movers = load_raw_data(config['INPUT_FILE'])
    results = process_all_paths(df_raw, num_movers, config)

    # --- STEP 3: Parameter Extraction & Export (Simplified) ---

    # A. Initial Exports: Saves the final resampled coordinate points (MoverPoints.csv).
    save_smoothed_points_csv(results, config['SMOOTHED_POINTS_FILE'], config['RESAMPLE_POINTS'])

    # B. Coefficient Generation: Calculates spline coefficients and raw point counts,
    #    merging both into the final file (ParametricSplineCoeff.csv) in a single step.
    generate_final_segment_coeffs(results, config['COEFF_OUTPUT_FILE_UPDATED'])

    # --- STEP 4: Safety Check ---
    collisions = check_mover_collisions(results, config)

    # --- STEP 5: Collision Avoidance (NEW STEP) ---
    df_coeff_avoidance, resolved = None, True
    if collisions:
        df_coeff_avoidance, resolved = resolve_collisions(results, collisions, config)

    # Determine which coefficient file to use for animation
    if resolved and df_coeff_avoidance is not None:
        # If avoidance succeeded, use the new file for animation
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_AVOIDANCE']
        # save_avoidance_points_csv(df_coeff_avoidance, config) # Optional secondary output
        print("Using avoidance path for animation.")
    else:
        # If no collisions or avoidance failed, use the original file
        config['COEFF_OUTPUT_FILE_ANIMATION'] = config['COEFF_OUTPUT_FILE_UPDATED']
        print("Using original path for animation (collision risk).")

    # --- STEP 6: Visualization (animations.py) ---
    plot_static_paths(results, num_movers)
    animate_single_movers(results, config)
    # Pass the selected file to animate_global_movers
    config[
        'GLOBAL_ANIMATION_FILE'] = 'GlobalMoverAnimation_Avoidance.gif' if resolved else 'GlobalMoverAnimation_Original.gif'
    animate_global_movers(results, num_movers, config)


if __name__ == "__main__":
    main()