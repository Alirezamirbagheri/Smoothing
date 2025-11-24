# animations.py (Final version of animate_global_movers function)

from config import *
from helpers import arc_length_param
from config import *
from helpers import arc_length_param
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import numpy as np
from scipy.interpolate import splev, PPoly
# --- IMPORT COLLISION CHECKER ---
# Assuming collision_check.py or a helper function is available to import
try:
    from collision_check import _precompute_mover_positions
except ImportError:
    # Fallback definition if the module isn't strictly loaded
    def _precompute_mover_positions(*args): return None, None


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
    Generates and saves a multi-mover animation with collision visualization.
    Movers turn RED when they are involved in a collision.
    """
    if not config['ANIMATE_ALL']:
        return

    print("\n--- Generating Global Animation (Density-Scaled, Collision Visualized) ---")

    # --- 1. Load Data and Determine Global Time Parameters (PRECOMPUTATION) ---
    try:
        df_coeff = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
    except FileNotFoundError:
        print(f"Warning: {config['COEFF_OUTPUT_FILE_UPDATED']} not found. Skipping global animation.")
        return

    # --- FIX: Capture the third return variable (u-progress) ---
    mover_pos_at_time, T_global, mover_u_at_time, df_coeff = _precompute_mover_positions(results, config)    # ---------------------------------------------

    if mover_pos_at_time is None:
        print("Warning: Could not precompute positions. Skipping global animation.")
        return

    # --- FIX 2: Define FPS (kept for reference, should be outside the previous 'if') ---
    FPS = 25  # Define the constant FPS used for animation speed/time calculation
    # -----------------------------------------------------------------------------------

    # T_global is now defined and populated
    num_frames = len(T_global)
    T_sim = T_global[-1]

    # Update: Use threshold for AABB check
    threshold = config['MOVER_SIZE']

    # --- 2. Collision Mapping Setup ---
    collision_flags = [np.zeros(num_frames, dtype=bool) for _ in range(num_movers)]

    # Iterate over all unique pairs (i, j) where i < j
    for i in range(num_movers):
        for j in range(i + 1, num_movers):
            pos_A = mover_pos_at_time[i]
            pos_B = mover_pos_at_time[j]

            # 1. Calculate absolute distance in X and Y
            dist_x = np.abs(pos_A[:, 0] - pos_B[:, 0])
            dist_y = np.abs(pos_A[:, 1] - pos_B[:, 1])

            # 2. Collision occurs if both conditions are met (AABB overlap)
            collision_mask = (dist_x < threshold) & (dist_y < threshold)

            active_collision_frames = np.where(collision_mask)[0]

            if active_collision_frames.size > 0:
                # Set the flag for both movers A and B for all frames where collision is active
                collision_flags[i][active_collision_frames] = True
                collision_flags[j][active_collision_frames] = True
                print(f"Collision frames found between Mover {i + 1} and Mover {j + 1}.")
    # --- 3. Setup Animation Plot ---
    min_x = 0 - config['PLOT_PADDING_MM']
    max_x = config['TABLE_LENGTH_MM'] + config['PLOT_PADDING_MM']
    min_y = 0 - config['PLOT_PADDING_MM']
    max_y = config['TABLE_WIDTH_MM'] + config['PLOT_PADDING_MM']

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Global Mover Animation (Collision Visualized) | T_sim={T_sim:.1f} s')
    ax.set_xlabel(f'X Position (mm)')
    ax.set_ylabel(f'Y Position (mm)')
    ax.grid(True)

    # Plot all paths (Original paths need to be retrieved from results, as in previous versions)
    for i, data in enumerate(results):
        ax.plot(data['x_smoothed'], data['y_smoothed'], linestyle='--', alpha=0.5, linewidth=1,
                color=plt.cm.get_cmap('hsv')(i / num_movers))

    # Store the default color for later use.
    # We add a hue offset (e.g., 0.25 for green/cyan start) to avoid red as the default color for Mover 1.
    color_offset = 0.25
    default_colors = [plt.cm.get_cmap('hsv')((i / num_movers + color_offset) % 1.0)
                      for i in range(num_movers)]

    # Create mover rectangles
    rects = []
    for i in range(num_movers):
        x_start, y_start = (mover_pos_at_time[i][0, 0], mover_pos_at_time[i][0, 1])
        rect = Rectangle(
            (x_start - config['MOVER_SIZE'] / 2, y_start - config['MOVER_SIZE'] / 2),
            config['MOVER_SIZE'], config['MOVER_SIZE'],
            color=default_colors[i],  # Use the newly offset default color
            alpha=0.8, fill=True
        )
        ax.add_patch(rect)
        rects.append(rect)
    # --- 4. Define Update Function (Collision Coloring) ---
    def update(frame_idx):
        artists_to_draw = []
        current_time = T_global[frame_idx]

        for i in range(num_movers):
            if i >= len(rects): continue

            xi, yi = mover_pos_at_time[i][frame_idx, :]

            # --- COLLISION CHECK AND COLOR CHANGE ---
            if collision_flags[i][frame_idx]:
                rects[i].set_color('red')  # Set color to red on collision
            else:
                rects[i].set_color(default_colors[i])  # Revert to default color
            # ----------------------------------------

            # Update Mover Rectangle position
            rects[i].set_xy((xi - config['MOVER_SIZE'] / 2, yi - config['MOVER_SIZE'] / 2))
            artists_to_draw.append(rects[i])

            # --- Update Text Label (Time and Percentage) ---
            # Retrieve u_progress (requires precalculating u_at_time, which is not in this simplified precompute)
            # Reverting to time display only for simplicity or assuming u_at_time is also passed/available.

            # Since the full precompute data is needed, we assume the necessary u_at_time
            # (which was stored in mover_u_at_time in the last implementation) is available
            # or we re-integrate that part of the precompute logic.

            # For this context, I will assume the u-progress array is also calculated
            # within the precomputation and stored in a list of arrays called `mover_u_at_time`.
            # Since I cannot modify all files at once, I will simplify the text display
            # to be safe, or you can ensure mover_u_at_time is passed here.

            # Assuming `mover_u_at_time` is available for calculation:
            try:
                # If the u array was precomputed and passed/available:
                u_progress = mover_u_at_time[i][frame_idx]
                percent_progress = u_progress * 100.0
                text_str = (f'Mover {i + 1} | Time: {current_time:.2f} s\n'
                            f'Progress: {percent_progress:.1f} %')
            except NameError:
                # Fallback if u_at_time wasn't explicitly passed/computed in this scope
                text_str = f'Mover {i + 1} | Time: {current_time:.2f} s'

            text_x, text_y = xi, yi + config['MOVER_SIZE'] / 2 + 20

            if not hasattr(rects[i], 'txt'):
                rects[i].txt = ax.text(
                    text_x, text_y,
                    text_str,
                    color='k', fontsize=9, ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
                )
            else:
                rects[i].txt.set_position((text_x, text_y))
                rects[i].txt.set_text(text_str)
            artists_to_draw.append(rects[i].txt)

        return artists_to_draw

    # --- 5. Run and Save Animation ---
    if num_frames > 1:
        # The line below now has FPS defined:
        ani_global = FuncAnimation(fig, update, frames=num_frames, interval=1000 / FPS, blit=True, repeat=True)

        ani_global.save(config['GLOBAL_ANIMATION_FILE'], writer='pillow', fps=FPS)
        print(f"✅ Global animation saved as {config['GLOBAL_ANIMATION_FILE']}")

        plt.show(block=False)
    else:
        print("Warning: Simulation time is too low. Skipping animation generation.")

    return