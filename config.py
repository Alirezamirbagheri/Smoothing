# config.py

import numpy as np
import pandas as pd
import matplotlib
# Ensure Matplotlib is configured before importing pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d, PPoly
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors
import csv

# === CONFIGURATION ===
INPUT_FILE = 'data/New_Coord_20251021.csv'
SMOOTHED_POINTS_FILE = 'Results/MoverPoints.csv'
COEFF_OUTPUT_FILE_UPDATED = 'Results/ParametricSplineCoeff.csv'
SMOOTHED_SEG_MAP_FILE = 'Results/SmoothedPointsPerSplineSegment.csv'
GLOBAL_ANIMATION_FILE = "Results/GlobalAnimation.gif"
COEFF_OUTPUT_FILE_AVOIDANCE = 'Results/ParametricSplineCoeff_Avoidance.csv' # Add to config
SMOOTHED_POINTS_FILE_AVOIDANCE = 'Results/SmoothedPoints_Avoidance.csv' # Add to config

# TUNABLES
RDP_TOL = 5.0            # RDP simplification tolerance
SMOOTHING_FACTOR_INIT = 5000  # initial smoothing factor
MAX_POINTS = 2000
RESAMPLE_POINTS = 500
ANIMATE = True
ANIMATE_ALL = True        # new global animation
K_ORDER = 3
MAX_DEV_ALLOWED = 10
MOVER_SIZE = 155.0  # mm, side length of square mover
VELOCITY = 100.0    # mm/s (assumed for global animation)

# --- TABLE DIMENSIONS FOR PLOTTING LIMITS ---
TABLE_LENGTH_MM = 1785.0
TABLE_WIDTH_MM = 1275.0
PLOT_PADDING_MM = 0.0  # Extra space around the table limits