from typing import Tuple
import numpy as np
import cv2 as cv

def _slope_degrees(height_m: np.ndarray, meters_per_px: float) -> np.ndarray:
    """
    Compute local surface slope in degrees from a heightfield in meters.
    Uses numpy.gradient with physical spacing so values are correct.
    """
    # grad returns dZ/dy, dZ/dx when spacing is given
    dzy, dzx = np.gradient(height_m, meters_per_px, meters_per_px)
    slope_tan = np.hypot(dzx, dzy)  # rise/run
    return np.degrees(np.arctan(slope_tan))

def anti_ripple_bilateral(
    height_m: np.ndarray,
    meters_per_px: float,
    stamp_grid_m: float = 2.0,     # 1.0 or 2.0 typical map scales
    strength: float = 0.6,         # 0..1, how aggressively to smooth
    slope_thresh_deg: float = 2.0, # only smooth where slope >= this
    clamp_window_px: Tuple[int, int] = (3, 31),  # safety clamp
) -> np.ndarray:
    """
    Remove the game stamping ripples while preserving true contours.

    height_m:       2D numpy array of heights in meters.
    meters_per_px:  horizontal resolution of the height grid in meters/pixel.
    stamp_grid_m:   the game's stamping cell size in meters (1.0 or 2.0).
    strength:       0..1, maps to bilateral sigmas.
    slope_thresh_deg: only apply on sloped ground where ripples show.
    clamp_window_px: min,max limits for the bilateral window.
    """
    assert height_m.ndim == 2, "height_m must be HxW array of meters"
    h32 = height_m.astype(np.float32, copy=False)

    # 1) Build a slope mask so we do not soften flats.
    slope_deg = _slope_degrees(h32, meters_per_px)
    slope_mask = slope_deg >= float(slope_thresh_deg)

    # 2) Window size targets ~ ripple frequency (1.5x stamp pitch).
    win_px = int(round((stamp_grid_m / meters_per_px) * 1.5))
    win_px = max(clamp_window_px[0], min(clamp_window_px[1], win_px))
    # force odd for bilateral
    if win_px % 2 == 0:
        win_px += 1

    # 3) Bilateral parameters.
    # sigmaColor is in "height units" (meters). Ripples are usually small.
    # Aim ~ 3-10 cm by default, scaled by strength.
    sigma_color_m = 0.03 + 0.12 * strength
    sigma_space_px = max(1, win_px)

    smoothed = cv.bilateralFilter(
        h32,
        d=win_px,
        sigmaColor=float(sigma_color_m),
        sigmaSpace=float(sigma_space_px),
        borderType=cv.BORDER_REPLICATE,
    )

    # 4) Apply only where sloped. Copy everywhere else.
    out = h32.copy()
    out[slope_mask] = smoothed[slope_mask]
    return out

def ripple_metric(height_m: np.ndarray, meters_per_px: float, slope_min_deg: float = 2.0) -> float:
    """
    Cheap objective metric to gauge ripple strength:
    variance of the Laplacian on sloped areas only (higher == bumpier).
    Use it to compare before/after.
    """
    slope_deg = _slope_degrees(height_m, meters_per_px)
    mask = slope_deg >= slope_min_deg
    lap = cv.Laplacian(height_m.astype(np.float32, copy=False), cv.CV_32F, ksize=3)
    vals = lap[mask]
    if vals.size == 0:
        return 0.0
    return float(vals.var())