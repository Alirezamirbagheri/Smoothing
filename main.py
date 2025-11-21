# main.py

from config import *
from data_loader import load_raw_data
from spline_fit import process_all_paths
from parameter_extractor import (
    # Only import the new final generator and the smoothed points exporter
    generate_final_segment_coeffs, save_smoothed_points_csv
)
from animations import plot_static_paths, animate_single_movers, animate_global_movers
import numpy as np
import pandas as pd


def main():
    """The main execution function, orchestrating the simplified modules."""

    config = {
        'RDP_TOL': RDP_TOL, 'SMOOTHING_FACTOR_INIT': SMOOTHING_FACTOR_INIT,
        'MAX_POINTS': MAX_POINTS, 'RESAMPLE_POINTS': RESAMPLE_POINTS,
        'K_ORDER': K_ORDER, 'MAX_DEV_ALLOWED': MAX_DEV_ALLOWED,
        'MOVER_SIZE': MOVER_SIZE, 'VELOCITY': VELOCITY,
        'ANIMATE': ANIMATE, 'ANIMATE_ALL': ANIMATE_ALL,
        'INPUT_FILE': INPUT_FILE, 'SMOOTHED_POINTS_FILE': SMOOTHED_POINTS_FILE,
        # COEFF_OUTPUT_FILE_OLD is still needed in config but ignored here
        'COEFF_OUTPUT_FILE_UPDATED': COEFF_OUTPUT_FILE_UPDATED,
        # RAW_SEG_MAP_FILE is still needed in config but ignored here
        'GLOBAL_ANIMATION_FILE': GLOBAL_ANIMATION_FILE
    }

    df_raw, num_movers = load_raw_data(config['INPUT_FILE'])
    results = process_all_paths(df_raw, num_movers, config)

    # --- STEP 3: Parameter Extraction & Export (Simplified) ---

    # A. Initial Exports: Saves the final resampled coordinate points (MoverPoints.csv).
    save_smoothed_points_csv(results, config['SMOOTHED_POINTS_FILE'], config['RESAMPLE_POINTS'])

    # B. Coefficient Generation: Calculates spline coefficients and raw point counts,
    #    merging both into the final file (ParametricSplineCoeff.csv) in a single step.
    generate_final_segment_coeffs(results, config['COEFF_OUTPUT_FILE_UPDATED'])

    # --- STEP 4: Visualization (animations.py) ---
    plot_static_paths(results, num_movers)
    animate_single_movers(results, config)
    animate_global_movers(results, num_movers, config)


if __name__ == "__main__":
    main()