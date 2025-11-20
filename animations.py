# animations.py

from config import *
from helpers import arc_length_param
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import numpy as np


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
    """Generates and saves a multi-mover animation with scaled velocity."""
    if not config['ANIMATE_ALL']:
        return

    print("\n--- Generating Global Animation ---")

    # --- 1. Precompute density factors for animation ---
    mover_segment_density = []

    # Load the updated coefficient file for density factor calculation
    try:
        df_coeff = pd.read_csv(config['COEFF_OUTPUT_FILE_UPDATED'])
        # Calculate the density factor (Length_Weight / RawPointsWeight)
        df_coeff['Density_Factor'] = (
                df_coeff['Length_Weight'] / df_coeff['RawPointsWeight']
        ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    except FileNotFoundError:
        print(f"Warning: {config['COEFF_OUTPUT_FILE_UPDATED']} not found. Using default density factor of 1.0.")
        df_coeff = pd.DataFrame()

    for mover_idx, data in enumerate(results):
        x_s, y_s, tck = data['x_smoothed'], data['y_smoothed'], data['tck']

        t_smooth = arc_length_param(x_s, y_s)
        seg_density = np.zeros(len(x_s))

        df_mover = df_coeff[df_coeff['Mover'] == mover_idx + 1].reset_index(drop=True)

        if tck is None or df_mover.empty:
            seg_density[:] = 1.0
        else:
            tx, c, k = tck
            ppx = PPoly.from_spline((tx, c[0], k))
            valid_indices = [j for j in range(len(ppx.x) - 1) if abs(ppx.x[j + 1] - ppx.x[j]) > 1e-9]

            for seg_idx_local, j in enumerate(valid_indices):
                u_start = float(ppx.x[j])
                u_end = float(ppx.x[j + 1])
                mask = (t_smooth >= u_start) & (t_smooth <= u_end)

                density = df_mover.loc[seg_idx_local, 'Density_Factor'] if seg_idx_local < len(df_mover) else 1.0
                seg_density[mask] = density

        mover_segment_density.append(seg_density)

    # --- 2. Setup Animation ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("All Movers (Scaled Velocity)")
    ax.axis('equal')

    all_x = np.concatenate([r['x_smoothed'] for r in results])
    all_y = np.concatenate([r['y_smoothed'] for r in results])
    ax.set_xlim(np.min(all_x) - config['MOVER_SIZE'], np.max(all_x) + config['MOVER_SIZE'])
    ax.set_ylim(np.min(all_y) - config['MOVER_SIZE'], np.max(all_y) + config['MOVER_SIZE'])

    colors = plt.cm.tab10(np.linspace(0, 1, num_movers))
    rects, paths = [], []

    for i, data in enumerate(results):
        x_s, y_s = data['x_smoothed'], data['y_smoothed']
        path, = ax.plot(x_s, y_s, '-', color=colors[i], alpha=0.5)
        rect = Rectangle((x_s[0] - config['MOVER_SIZE'] / 2, y_s[0] - config['MOVER_SIZE'] / 2),
                         config['MOVER_SIZE'], config['MOVER_SIZE'],
                         facecolor=colors[i], alpha=0.3, edgecolor='k')
        ax.add_patch(rect)
        rects.append(rect)
        paths.append(path)

    for i, data in enumerate(results):
        x_s, y_s = data['x_smoothed'], data['y_smoothed']
        ax.plot(x_s[0], y_s[0], 'go', ms=5, label=f'Mover {i + 1} Start' if i == 0 else "")
        ax.plot(x_s[-1], y_s[-1], 'ro', ms=5, label=f'Mover {i + 1} End' if i == 0 else "")
    ax.legend()

    max_frames = max(len(r['x_smoothed']) for r in results)

    # --- 3. Update Function ---
    def update(frame_idx):
        artists_to_draw = []
        for i, data in enumerate(results):
            x_s, y_s = data['x_smoothed'], data['y_smoothed']
            if not x_s.size: continue

            idx = min(frame_idx, len(x_s) - 1)
            xi, yi = x_s[idx], y_s[idx]

            rects[i].set_xy((xi - config['MOVER_SIZE'] / 2, yi - config['MOVER_SIZE'] / 2))
            artists_to_draw.append(rects[i])

            scaled_velocity = config['VELOCITY'] * mover_segment_density[i][idx]

            text_x, text_y = xi, yi + config['MOVER_SIZE'] / 2 + 20
            text_str = f'{scaled_velocity:.1f} | Mover {i + 1}'

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

    # --- 4. Run/Save Animation ---
    ani_global = FuncAnimation(fig, update, frames=max_frames, interval=40, blit=True, repeat=True)

    # Save as GIF
    ani_global.save(config['GLOBAL_ANIMATION_FILE'], writer='pillow', fps=25)
    print(f"✅ Global animation saved as {config['GLOBAL_ANIMATION_FILE']}")

    plt.show()