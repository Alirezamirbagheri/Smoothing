# data_loader.py

import pandas as pd
import numpy as np


def load_raw_data(input_file):
    """
    Loads raw mover coordinate data from the input CSV file.

    Returns:
        df (pd.DataFrame): DataFrame containing raw coordinates.
        num_movers (int): The number of movers detected.
    """
    df = pd.read_csv(input_file, header=0)
    num_movers = df.shape[1] // 2
    print(f"Data loaded from {input_file}. Detected {num_movers} movers.")
    return df, num_movers


def extract_mover_data(df, mover_index):
    """
    Extracts raw X and Y coordinates for a single mover.
    """
    i = mover_index
    x_raw = df.iloc[:, i * 2].astype(float).values
    y_raw = df.iloc[:, i * 2 + 1].astype(float).values
    return x_raw, y_raw