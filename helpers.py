# helpers.py

import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors


def rdp(points, eps):
    """Ramer-Douglas-Peucker algorithm for path simplification."""
    if len(points) < 3: return points.copy()
    start, end = points[0], points[-1]
    seg = end - start
    seg_len2 = seg.dot(seg)
    if seg_len2 == 0:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        t = np.clip(((points - start).dot(seg)) / seg_len2, 0.0, 1.0)
        proj = start + np.outer(t, seg)
        dists = np.linalg.norm(points - proj, axis=1)

    idx = np.argmax(dists)
    maxd = dists[idx]

    if maxd > eps:
        left = rdp(points[:idx + 1], eps)
        right = rdp(points[idx:], eps)
        return np.vstack((left[:-1], right))
    else:
        return np.vstack((start, end))


def prepare_points(x, y):
    """Removes duplicate points from coordinates."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    coords = np.vstack((x, y)).T
    _, idx = np.unique(np.round(coords, 6), axis=0, return_index=True)
    idx = np.sort(idx)
    x, y = x[idx], y[idx]
    if len(x) < 3:
        raise ValueError("Too few unique points after cleaning")
    return x, y


def arc_length_param(x, y):
    """Calculates a normalized cumulative arc length parameter (u in [0, 1])."""
    dx, dy = np.diff(x), np.diff(y)
    d = np.sqrt(dx ** 2 + dy ** 2)
    t = np.insert(np.cumsum(d), 0, 0.0)
    return t / t[-1] if t[-1] != 0 else np.zeros_like(t)


def compute_length(xs, ys):
    """Computes the total path length."""
    dx, dy = np.diff(xs), np.diff(ys)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def compute_max_deviation(original_x, original_y, smoothed_x, smoothed_y):
    """Computes max and mean deviation using nearest neighbor search."""
    orig = np.vstack((original_x, original_y)).T
    smoothed = np.vstack((smoothed_x, smoothed_y)).T
    nbrs = NearestNeighbors(n_neighbors=1).fit(smoothed)
    distances, _ = nbrs.kneighbors(orig)
    return float(np.max(distances)), float(np.mean(distances))


def local_to_global_ascending(coeffs, u0):
    """
    Converts a cubic polynomial's coefficients from descending powers
    (local coordinate system) to ascending powers (global coordinate system).
    """
    a3, a2, a1, a0 = coeffs
    A3 = a3
    A2 = a2 - 3 * a3 * u0
    A1 = a1 - 2 * a2 * u0 + 3 * a3 * u0 ** 2
    A0 = a0 - a1 * u0 + a2 * u0 ** 2 - a3 * u0 ** 3
    return A0, A1, A2, A3