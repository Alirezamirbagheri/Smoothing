# main.py

from config import *
from data_loader import load_raw_data
from spline_fit import process_all_paths
from parameter_extractor import (
    generate_segment_coeffs, save_smoothed_points_csv,
    count_raw_points_per_segment, save_raw_segment_map, finalize_coeff_file  # Changed imports
)
from animations import plot_static_paths, animate_single_movers, animate_global_movers
import numpy as np
import pandas as pd


def main():
    """The main execution function, orchestrating the modules."""

    config = {
        'RDP_TOL': RDP_TOL, 'SMOOTHING_FACTOR_INIT': SMOOTHING_FACTOR_INIT,
        'MAX_POINTS': MAX_POINTS, 'RESAMPLE_POINTS': RESAMPLE_POINTS,
        'K_ORDER': K_ORDER, 'MAX_DEV_ALLOWED': MAX_DEV_ALLOWED,
        'MOVER_SIZE': MOVER_SIZE, 'VELOCITY': VELOCITY,
        'ANIMATE': ANIMATE, 'ANIMATE_ALL': ANIMATE_ALL,
        'INPUT_FILE': INPUT_FILE, 'SMOOTHED_POINTS_FILE': SMOOTHED_POINTS_FILE,
        'COEFF_OUTPUT_FILE_OLD': COEFF_OUTPUT_FILE_OLD,
        'COEFF_OUTPUT_FILE_UPDATED': COEFF_OUTPUT_FILE_UPDATED,
        'RAW_SEG_MAP_FILE': RAW_SEG_MAP_FILE,
        'GLOBAL_ANIMATION_FILE': GLOBAL_ANIMATION_FILE
    }

    df_raw, num_movers = load_raw_data(config['INPUT_FILE'])
    results = process_all_paths(df_raw, num_movers, config)

    # --- STEP 3: Parameter Extraction & Export (parameter_extractor.py) ---

    # A. Initial Exports
    save_smoothed_points_csv(results, config['SMOOTHED_POINTS_FILE'], config['RESAMPLE_POINTS'])

    # B. Raw Point Counting (Now returns DataFrame and totals)
    df_raw_seg, total_raw_points_per_mover = count_raw_points_per_segment(results, num_movers)

    # C. Raw Segment Map Export (Simplified)
    save_raw_segment_map(df_raw_seg, config['RAW_SEG_MAP_FILE'])

    # D. Coefficient Generation (Generates old CSV)
    coeff_rows = generate_segment_coeffs(results, config['COEFF_OUTPUT_FILE_OLD'], config['COEFF_OUTPUT_FILE_UPDATED'])

    # E. Final Coefficient File Update
    finalize_coeff_file(coeff_rows, df_raw_seg, config['COEFF_OUTPUT_FILE_UPDATED'])

    # --- STEP 4: Visualization (animations.py) ---
    plot_static_paths(results, num_movers)
    animate_single_movers(results, config)
    animate_global_movers(results, num_movers, config)


if __name__ == "__main__":
    main()