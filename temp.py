# animations.py (Final version of animate_global_movers function)

from config import *
from helpers import arc_length_param
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import numpy as np
from scipy.interpolate import splev, PPoly


def get_limits(results):
    """Calculates min/max bounds for all paths."""
    all_x = np.concatenate([d['x_smoothed'] for d in results if d['x_smoothed'].size])
    all_y = np.concatenate([d['y_smoothed'] for d in results if d['y_smoothed'].size])

    if all_x.size == 0 or all_y.size == 0:
        # Return sensible defaults if no points are found
        return 0, 100, 0, 100

    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()

    # Add padding
    return min_x - 50, max_x + 50, min_y - 50, max_y + 50


def plot_static_paths(results, num_movers):
    # ... (function body remains the same) ...
    print("\n--- Generating Static Plots ---")
    plt.ioff()
    fig_all, axes = plt.subplots(1, num_movers, figsize=(6 * num_movers, 5))
    if num_movers == 1: axes = [axes]

    for i, data in enumerate(results):
        x_org, y_org = data['x_cleaned'], data['y_cleaned']
        x_s, y_s = data['x_smoothed'], data['y_smoothed']
        path_len, max_dev, mean_dev = data['path_len'], data['max_dev'], data['mean_dev']

        ax = axes[i]
        ax.plot(x_org, y_org, 'ko', ms=3, label='Original')
        ax.plot(x_s, y_s, 'r-', lw=1.8, label='Smoothed')
        ax.set_title(f'Mover {i + 1}')
        ax.axis('equal')
        ax.legend()
        ax.text(0.5, -0.13,
                f'Length: {path_len:.2f} mm\nMax dev: {max_dev:.2f} mm (avg {mean_dev:.2f})',
                transform=ax.transAxes, ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.show(block=True)
    print("✅ Static plots displayed (close window to continue)")


def animate_single_movers(results, config):
    # ... (function body remains the same) ...
    if not config['ANIMATE']:
        return

    print("\n--- Running Single Mover Animations ---")
    for i, data in enumerate(results):
        x_org, y_org = data['x_cleaned'], data['y_cleaned']
        x_s, y_s = data['x_smoothed'], data['y_smoothed']
        path_len = data['path_len']

        fig_a, ax_a = plt.subplots(figsize=(6, 5))
        ax_a.set_title(f"Mover {i + 1} | length {path_len:.2f} mm")
        margin = 50.0

        all_x = np.concatenate([x_org, x_s])
        all_y = np.concatenate([y_org, y_s])
        ax_a.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        ax_a.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
        ax_a.axis('equal')

        ax_a.plot(x_org, y_org, 'ko', ms=3, label='Original Points')
        spline_line, = ax_a.plot([], [], 'r-', lw=2, label='Smoothed')
        moving_point, = ax_a.plot([], [], 'bo', ms=6, label='Moving')
        ax_a.legend()

        def init():
            spline_line.set_data([], [])
            moving_point.set_data([], [])
            return spline_line, moving_point

        def update(frame):
            spline_line.set_data(x_s[:frame + 1], y_s[:frame + 1])
            moving_point.set_data([x_s[frame]], [y_s[frame]])
            return spline_line, moving_point

        ani = FuncAnimation(fig_a, update, frames=len(x_s), init_func=init,
                            interval=20, blit=True, repeat=False)
        print(f"Showing animation for mover {i + 1} (close window to continue...)")
        plt.show(block=True)


def animate_global_movers(results, num_movers, config):
    """
    Generates and saves a multi-mover animation where segment velocity is scaled
    by NumRawPoints (Time-Based Density Scaling), displaying elapsed time and
    passed length percentage.
    """
    if not config['ANIMATE_ALL']:
        return

    print("\n--- Generating Global Animation (Density-Scaled Time & Progress Display) ---")

    # --- 1. Load Data and Determine Global Time Parameters ---

    try:
        df_coeff = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
    except FileNotFoundError:
        print(f"Warning: {config['COEFF_OUTPUT_FILE_UPDATED']} not found. Skipping global animation.")
        return

    all_x = np.concatenate([d['x_smoothed'] for d in results if d['x_smoothed'].size])
    if all_x.size == 0:
        print("Warning: No smoothed points found. Cannot generate global animation.")
        return

    mover_total_raw_points = []

    for mover_idx in range(num_movers):
        df_mover = df_coeff[df_coeff['Mover'] == mover_idx + 1].copy()
        total_raw_points = df_mover['NumRawPoints'].sum()
        mover_total_raw_points.append(total_raw_points)

    max_raw_points = max(mover_total_raw_points) if mover_total_raw_points else 0

    if config['VELOCITY'] <= 0 or max_raw_points <= 0:
        print("Warning: Velocity or max raw point count is zero. Skipping global animation.")
        return

    FPS = 25
    T_sim = max_raw_points / FPS
    num_frames = max_raw_points

    T_global = np.linspace(0, T_sim, num_frames)

    # --- 2. Precompute Mover Positions vs. Time (Density-Scaled) ---

    mover_pos_at_time = []
    mover_u_at_time = []  # NEW: Store normalized parameter u for length calculation

    for mover_idx in range(num_movers):
        df_mover = df_coeff[df_coeff['Mover'] == mover_idx + 1].copy()
        data = results[mover_idx]
        tck = data['tck']

        if tck is None or data['path_len'] <= 0:
            start_pos = [data['x_smoothed'][0], data['y_smoothed'][0]] if data['x_smoothed'].size else [0, 0]
            mover_pos_at_time.append(np.tile(start_pos, (num_frames, 1)))
            mover_u_at_time.append(np.zeros(num_frames))  # u=0 for static path
            continue

        path_len = data['path_len']
        current_mover_raw_points = mover_total_raw_points[mover_idx]

        if current_mover_raw_points > 0:
            df_mover['Time_sec'] = T_sim * (df_mover['NumRawPoints'] / max_raw_points)
            df_mover['V_seg_mm_s'] = df_mover['Length_mm'] / df_mover['Time_sec']
        else:
            V_uniform = path_len / T_sim
            df_mover['V_seg_mm_s'] = V_uniform
            df_mover['Time_sec'] = df_mover['Length_mm'] / V_uniform

        df_mover['T_cum_end'] = df_mover['Time_sec'].cumsum()
        df_mover['T_cum_start'] = df_mover['T_cum_end'].shift(1, fill_value=0.0)

        path_at_time = np.zeros((num_frames, 2))
        u_at_time = np.zeros(num_frames)

        # Iterate over global time points T_global
        for t_idx, t_g in enumerate(T_global):
            segment = df_mover[(t_g >= df_mover['T_cum_start']) & (t_g < df_mover['T_cum_end'])]

            if segment.empty:
                # Time is past the end of the path
                u_g = 1.0
                pos_xy = np.array(splev(u_g, tck)).flatten()
            else:
                # Segment data
                u_start = segment['U_Start'].iloc[0]
                u_end = segment['U_End'].iloc[0]
                t_start = segment['T_cum_start'].iloc[0]
                v_seg = segment['V_seg_mm_s'].iloc[0]
                seg_len = segment['Length_mm'].iloc[0]

                time_in_seg = t_g - t_start
                distance_in_seg = v_seg * time_in_seg

                u_seg_length = u_end - u_start
                if seg_len > 1e-6:
                    u_local = distance_in_seg / seg_len
                    u_g = u_start + u_local * u_seg_length
                else:
                    u_g = u_start  # Segment is point-like

                u_g = np.clip(u_g, u_start, u_end)

                # Use the spline object to get the (x, y) coordinates at parameter u_g
                pos_xy = np.array(splev(u_g, tck)).flatten()

            path_at_time[t_idx, :] = pos_xy
            u_at_time[t_idx] = u_g  # Store u_g

        mover_pos_at_time.append(path_at_time)
        mover_u_at_time.append(u_at_time)

    # --- 3. Setup Animation Plot ---
    min_x = 0 - config['PLOT_PADDING_MM']
    max_x = config['TABLE_LENGTH_MM'] + config['PLOT_PADDING_MM']
    min_y = 0 - config['PLOT_PADDING_MM']
    max_y = config['TABLE_WIDTH_MM'] + config['PLOT_PADDING_MM']

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Global Mover Animation (Density-Scaled) | T_sim={T_sim:.1f} s | V_ref={config["VELOCITY"]:.0f} mm/s')
    ax.set_xlabel(f'X Position (mm)')
    ax.set_ylabel(f'Y Position (mm)')
    ax.grid(True)

    # Plot all paths
    for i, data in enumerate(results):
        ax.plot(data['x_smoothed'], data['y_smoothed'], linestyle='--', alpha=0.5, linewidth=1,
                color=plt.cm.get_cmap('hsv')(i / num_movers))

    # Create mover rectangles
    rects = []
    for i in range(num_movers):
        x_start, y_start = (mover_pos_at_time[i][0, 0], mover_pos_at_time[i][0, 1])
        rect = Rectangle(
            (x_start - config['MOVER_SIZE'] / 2, y_start - config['MOVER_SIZE'] / 2),
            config['MOVER_SIZE'], config['MOVER_SIZE'],
            color=plt.cm.get_cmap('hsv')(i / num_movers),
            alpha=0.8, fill=True
        )
        ax.add_patch(rect)
        rects.append(rect)

    # --- 4. Define Update Function (Time-Based) ---
    def update(frame_idx):
        artists_to_draw = []
        current_time = T_global[frame_idx]

        for i, data in enumerate(results):
            if i >= len(rects): continue

            xi, yi = mover_pos_at_time[i][frame_idx, :]
            u_progress = mover_u_at_time[i][frame_idx]  # Get stored u value

            # Since the B-spline parameter 'u' is a normalized arc length
            # when fitting is based on chord length parameterization,
            # u_progress * 100 gives the percentage of the path traversed.
            percent_progress = u_progress * 100.0

            # Update Mover Rectangle position
            rects[i].set_xy((xi - config['MOVER_SIZE'] / 2, yi - config['MOVER_SIZE'] / 2))
            artists_to_draw.append(rects[i])

            # --- Update Text Label (Displaying Time AND Percentage) ---
            text_x, text_y = xi, yi + config['MOVER_SIZE'] / 2 + 20
            # Combine time and percentage into a single label
            text_str = (f'Mover {i + 1} | Time: {current_time:.2f} s\n'
                        f'Progress: {percent_progress:.1f} %')

            if not hasattr(rects[i], 'txt'):
                rects[i].txt = ax.text(
                    text_x, text_y,
                    text_str,
                    color='k', fontsize=9, ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)  # Added background for readability
                )
            else:
                rects[i].txt.set_position((text_x, text_y))
                rects[i].txt.set_text(text_str)
            artists_to_draw.append(rects[i].txt)

        return artists_to_draw

    # --- 5. Run and Save Animation ---
    if num_frames > 1:
        ani_global = FuncAnimation(fig, update, frames=num_frames, interval=1000 / FPS, blit=True, repeat=True)

        ani_global.save(config['GLOBAL_ANIMATION_FILE'], writer='pillow', fps=FPS)
        print(f"✅ Global animation saved as {config['GLOBAL_ANIMATION_FILE']}")

        plt.show(block=False)
    else:
        print("Warning: Simulation time is too low. Skipping animation generation.")

    return