# animations.py

from config import *
from helpers import arc_length_param
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import numpy as np


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
    """Generates and displays static plots of original vs. smoothed paths."""
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
    """Generates and displays single-mover path-following animations."""
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
    Generates and saves a multi-mover animation with velocity scaled based on
    NumRawPoints / TotalRawPoints (Density Scaling).
    """
    if not config['ANIMATE_ALL']:
        return

    print("\n--- Generating Global Animation (Using Density Scaling) ---")

    # --- 1. Precompute density factors for animation ---
    mover_segment_density = []

    try:
        # Load the final coefficient file (ParametricSplineCoeff.csv)
        df_coeff = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
    except FileNotFoundError:
        print(f"Warning: {config['COEFF_OUTPUT_FILE_UPDATED']} not found. Skipping global animation.")
        return

    all_x = np.concatenate([d['x_smoothed'] for d in results if d['x_smoothed'].size])
    all_y = np.concatenate([d['y_smoothed'] for d in results if d['y_smoothed'].size])

    if all_x.size == 0 or all_y.size == 0:
        print("Warning: No smoothed points found. Cannot generate global animation.")
        return

    for mover_idx in range(num_movers):
        df_mover = df_coeff[df_coeff['Mover'] == mover_idx + 1].copy()
        data = results[mover_idx]
        x_s, y_s, tck = data['x_smoothed'], data['y_smoothed'], data['tck']

        # Calculate Total Raw Points for this mover
        total_raw_points = df_mover['NumRawPoints'].sum()

        if total_raw_points == 0 or tck is None:
            df_mover['Density_Factor'] = 1.0
            seg_density = np.ones(len(x_s))
            print(f"Warning: Mover {mover_idx + 1} has zero raw points or no spline. Using uniform speed.")
        else:
            # Density_Factor = (Segment Raw Points / Total Raw Points)
            df_mover['Density_Factor'] = df_mover['NumRawPoints'] / total_raw_points

            # Map the density factors onto the smoothed points array
            t_smooth = arc_length_param(x_s, y_s)
            seg_density = np.zeros(len(x_s))

            tx, c, k = tck
            ppx = PPoly.from_spline((tx, c[0], k))
            valid_indices = [j for j in range(len(ppx.x) - 1) if abs(ppx.x[j + 1] - ppx.x[j]) > 1e-9]

            for seg_idx_local, j in enumerate(valid_indices):
                u_start = float(ppx.x[j]);
                u_end = float(ppx.x[j + 1])
                mask = (t_smooth >= u_start) & (t_smooth <= u_end)

                # Retrieve the calculated Density_Factor for this segment
                density = df_mover.loc[df_mover['Segment'] == seg_idx_local + 1, 'Density_Factor'].iloc[0]
                seg_density[mask] = density

        mover_segment_density.append(seg_density)

    # --- 2. Setup Animation (Fixes NameError) ---
    # Assuming get_limits is defined in animations.py
    min_x, max_x, min_y, max_y = get_limits(results)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Global Mover Animation with Raw Point Density Scaling')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.grid(True)

    # Plot all paths
    for i, data in enumerate(results):
        ax.plot(data['x_smoothed'], data['y_smoothed'], linestyle='--', alpha=0.5, linewidth=1,
                color=plt.cm.get_cmap('hsv')(i / num_movers))

    # Create mover rectangles
    rects = []
    for i in range(num_movers):
        if results[i]['x_smoothed'].size == 0:
            continue

        rect = Rectangle(
            (results[i]['x_smoothed'][0] - config['MOVER_SIZE'] / 2,
             results[i]['y_smoothed'][0] - config['MOVER_SIZE'] / 2),
            config['MOVER_SIZE'], config['MOVER_SIZE'],
            color=plt.cm.get_cmap('hsv')(i / num_movers),
            alpha=0.8, fill=True
        )
        ax.add_patch(rect)
        rects.append(rect)

    max_frames = max(len(r['x_smoothed']) for r in results) if results else 0

    # --- 3. Define Update Function (Density Scaling) ---
    def update(frame_idx):
        artists_to_draw = []
        for i, data in enumerate(results):
            if i >= len(rects): continue

            x_s, y_s = data['x_smoothed'], data['y_smoothed']
            if not x_s.size: continue

            # Get current index, clamping to the path length
            idx = min(frame_idx, len(x_s) - 1)
            xi, yi = x_s[idx], y_s[idx]

            rects[i].set_xy((xi - config['MOVER_SIZE'] / 2, yi - config['MOVER_SIZE'] / 2))
            artists_to_draw.append(rects[i])

            # Get the time factor (density) for the current segment
            time_factor = mover_segment_density[i][idx]
            display_factor = time_factor * 100  # Display as percentage of total time

            text_x, text_y = xi, yi + config['MOVER_SIZE'] / 2 + 20
            text_str = f'Time Factor: {display_factor:.1f}% | Mover {i + 1}'

            if hasattr(rects[i], 'txt'):
                rects[i].txt.set_position((text_x, text_y))
                rects[i].txt.set_text(text_str)
            else:
                rects[i].txt = ax.text(
                    text_x, text_y,
                    text_str,
                    color='k', fontsize=9, ha='center'
                )
            artists_to_draw.append(rects[i].txt)

        return artists_to_draw

    # --- 4. Run and Save Animation ---
    if max_frames > 1:
        ani_global = FuncAnimation(fig, update, frames=max_frames, interval=40, blit=True, repeat=True)

        ani_global.save(config['GLOBAL_ANIMATION_FILE'], writer='pillow', fps=25)
        print(f"✅ Global animation saved as {config['GLOBAL_ANIMATION_FILE']}")

        plt.show()
    else:
        print("Warning: Max frames is too low. Skipping animation generation.")