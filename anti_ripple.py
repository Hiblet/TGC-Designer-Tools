from typing import Tuple
import numpy as np
import cv2 as cv

def _as_2d(a: np.ndarray):
    """
    Ensure we have a 2D array for processing.
    Returns (two_d_array, reshape_tag).
    reshape_tag is used to restore original shape if it was 3D with a singleton dim.
    """
    arr = np.asarray(a)
    if arr.ndim == 2:
        return arr, ("2d", arr.shape)
    if arr.ndim == 3:
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            if arr.shape[-1] == 1:
                return squeezed, ("expand_last", arr.shape)
            elif arr.shape[0] == 1:
                return squeezed, ("expand_first", arr.shape)
            elif arr.shape[1] == 1:
                return squeezed, ("expand_middle", arr.shape)
    # Fallback: process as-is (may be 1D or >3D, we will no-op later)
    return arr, ("unchanged", arr.shape)

def _restore_shape(a2d: np.ndarray, reshape_tag):
    tag, orig_shape = reshape_tag
    if tag == "expand_last":
        return a2d[..., np.newaxis]
    if tag == "expand_first":
        return a2d[np.newaxis, ...]
    if tag == "expand_middle":
        return a2d[:, np.newaxis, :]
    # "2d" or "unchanged"
    return a2d

def _slope_degrees(height_m: np.ndarray, meters_per_px: float) -> np.ndarray:
    """
    Return slope (deg). Works with old/new NumPy and tiny arrays.
    If array is too small to compute a 2D gradient, returns zeros.
    """
    h2d, tag = _as_2d(height_m)
    h32 = np.ascontiguousarray(h2d.astype(np.float32, copy=False))

    if h32.ndim != 2:
        return np.zeros_like(h32, dtype=np.float32)
    H, W = h32.shape
    if H < 2 or W < 2:
        return np.zeros_like(h32, dtype=np.float32)

    try:
        dzy, dzx = np.gradient(h32, float(meters_per_px), float(meters_per_px))
    except TypeError:
        dzy, dzx = np.gradient(h32)
        inv = 1.0 / float(meters_per_px)
        dzy *= inv
        dzx *= inv

    slope_tan = np.hypot(dzx, dzy)
    slope_deg = np.degrees(np.arctan(slope_tan))
    return slope_deg  # 2D map

def anti_ripple_bilateral(
    height_m: np.ndarray,
    meters_per_px: float,
    stamp_grid_m: float = 2.0,
    strength: float = 0.6,
    slope_thresh_deg: float = 2.0,
    clamp_window_px: Tuple[int, int] = (3, 31),
    # ---- BLEND IMPROVEMENT KNOBS (safe defaults keep old behavior) ----
    blend_width_deg: float = 0.0,   # 0.0 == old hard mask; try 3.0 to enable soft blend
    max_delta_m: float = 0.0,       # 0.0 == no clamp; e.g., 0.10 limits changes to +-10cm
) -> np.ndarray:
    """
    Edge-preserving smoothing targeted at ripple frequency.
    Accepts HxW, HxWx1, 1xHxW. Returns in the same shape it received.

    New knobs:
      blend_width_deg: softens the on/off slope mask across this many degrees.
                       0.0 keeps the legacy hard mask.
      max_delta_m:     optional safety clamp on how far heights may move.
    """
    h2d, tag = _as_2d(height_m)
    if h2d.ndim != 2:
        return height_m

    h32 = np.ascontiguousarray(h2d.astype(np.float32, copy=False))
    H, W = h32.shape
    if H == 0 or W == 0:
        return height_m

    # Window ~1.5x stamping pitch (in px), clamped and odd
    win_px = int(round((stamp_grid_m / float(meters_per_px)) * 1.5))
    win_px = max(clamp_window_px[0], min(clamp_window_px[1], win_px))
    if win_px % 2 == 0:
        win_px += 1

    # Bilateral params
    sigma_color_m = 0.03 + 0.12 * float(strength)  # meters
    sigma_space_px = max(1, win_px)

    # Slope map
    slope_deg = _slope_degrees(h32, meters_per_px)
    slope_mask = (slope_deg >= float(slope_thresh_deg))

    # Run bilateral (positional args for cv2 compatibility)
    smoothed = cv.bilateralFilter(h32, win_px, float(sigma_color_m), float(sigma_space_px))

    # ---- BLEND IMPROVEMENT START --------------------------------------
    if blend_width_deg > 0.0:
        # Softly blend smoothed in where slope rises above the threshold
        t0 = float(slope_thresh_deg)
        t1 = t0 + float(blend_width_deg)
        denom = max(1e-6, (t1 - t0))
        alpha = (slope_deg - t0) / denom
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        out2d = h32 * (1.0 - alpha) + smoothed * alpha
    else:
        # Legacy behavior: hard mask replacement
        if slope_mask.any():
            out2d = h32.copy()
            out2d[slope_mask] = smoothed[slope_mask]
        else:
            out2d = smoothed
    # Optional safety clamp on total change
    if max_delta_m > 0.0:
        diff = out2d - h32
        np.clip(diff, -float(max_delta_m), float(max_delta_m), out=diff)
        out2d = h32 + diff
    # ---- BLEND IMPROVEMENT END ----------------------------------------

    return _restore_shape(out2d, tag)




def ripple_metric(height_m: np.ndarray, meters_per_px: float, slope_min_deg: float = 2.0) -> float:
    """
    Ripple metric: variance of Laplacian over sloped areas.
    If the slope mask is empty (or the array is tiny), fall back to whole-image var.
    """
    h2d, _ = _as_2d(height_m)
    h32 = np.ascontiguousarray(h2d.astype(np.float32, copy=False))

    if h32.ndim != 2:
        return 0.0
    H, W = h32.shape
    if H < 3 or W < 3:
        return 0.0

    slope_deg = _slope_degrees(h32, meters_per_px)
    mask = (slope_deg >= float(slope_min_deg))

    lap = cv.Laplacian(h32, cv.CV_32F, ksize=3)
    if mask.any():
        vals = lap[mask]
        return float(vals.var())
    else:
        # No sloped pixels at this threshold; use whole image as a fallback
        return float(lap.var())

def ripple_diag(height_m: np.ndarray, meters_per_px: float) -> dict:
    """
    Quick diagnostics to understand why the metric might be zero.
    Returns basic slope stats and counts above a few thresholds.
    """
    h2d, _ = _as_2d(height_m)
    out = {"shape": tuple(h2d.shape), "mpp": float(meters_per_px)}
    if h2d.ndim != 2 or min(h2d.shape) < 2:
        out.update(dict(slope_min=0.0, slope_max=0.0, slope_mean=0.0,
                        n_ge_0_5=0, n_ge_1=0, n_ge_2=0, n_ge_4=0))
        return out

    s = _slope_degrees(h2d, meters_per_px)
    out["slope_min"] = float(np.nanmin(s))
    out["slope_max"] = float(np.nanmax(s))
    out["slope_mean"] = float(np.nanmean(s))
    out["n_ge_0_5"] = int((s >= 0.5).sum())
    out["n_ge_1"]   = int((s >= 1.0).sum())
    out["n_ge_2"]   = int((s >= 2.0).sum())
    out["n_ge_4"]   = int((s >= 4.0).sum())
    return out



# ---------- DEBUG + SAFE METRIC HELPERS ----------

def _print_stats(label: str, arr: np.ndarray, printf=print):
    a = np.asarray(arr)
    finite = np.isfinite(a)
    n_total = a.size
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    if n_total == 0:
        printf(f"{label}: EMPTY")
        return
    vmin = float(np.nanmin(a))
    vmax = float(np.nanmax(a))
    vmean = float(np.nanmean(a))
    vstd = float(np.nanstd(a))
    printf(
        f"{label}: shape={a.shape} dtype={a.dtype} "
        f"finite={n_finite}/{n_total} nan={n_nan} inf={n_inf} "
        f"min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f} std={vstd:.6f}"
    )

def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with median of finite values (stable for diagnostics)."""
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if finite.all():
        return a
    if finite.any():
        med = float(np.nanmedian(a))
    else:
        med = 0.0
    out = a.copy()
    out[~finite] = med
    return out

def ripple_metric_safe(height_m: np.ndarray, meters_per_px: float, slope_min_deg: float = 2.0) -> float:
    """
    Safer version used by the debug path; never returns NaN.
    """
    h2d, _ = _as_2d(height_m)
    if h2d.ndim != 2 or min(h2d.shape) < 3:
        return 0.0
    h32 = _fill_nans(h2d.astype(np.float32, copy=False))

    slope = _slope_degrees(h32, meters_per_px)
    mask = (slope >= float(slope_min_deg))
    lap = cv.Laplacian(h32, cv.CV_32F, ksize=3)

    if mask.any():
        vals = lap[mask]
        var = float(np.var(vals))
    else:
        var = float(np.var(lap))
    # Guard against weirdness
    if not np.isfinite(var):
        var = 0.0
    return var

def ripple_metric_debug(height_m: np.ndarray, meters_per_px: float, slope_min_deg: float = 2.0, printf=print) -> float:
    """
    Very chatty metric. Prints stats at each step and returns a finite number.
    """
    printf("=== ripple_metric_debug START ===")
    h2d, _ = _as_2d(height_m)
    _print_stats("raw height", h2d, printf)

    if h2d.ndim != 2:
        printf("height not 2D after squeeze; bailing metric with 0.0")
        printf("=== ripple_metric_debug END ===")
        return 0.0

    h32 = _fill_nans(h2d.astype(np.float32, copy=False))
    _print_stats("height filled", h32, printf)

    slope = _slope_degrees(h32, meters_per_px)
    _print_stats("slope(deg)", slope, printf)

    mask = (slope >= float(slope_min_deg))
    n_mask = int(mask.sum())
    n_pix = mask.size
    printf(f"mask: slope>= {slope_min_deg} deg -> {n_mask}/{n_pix} pixels")

    lap = cv.Laplacian(h32, cv.CV_32F, ksize=3)
    _print_stats("laplacian", lap, printf)

    if n_mask > 0:
        vals = lap[mask]
        _print_stats("laplacian(masked)", vals, printf)
        var = float(np.var(vals))
    else:
        printf("mask empty; using whole image variance")
        var = float(np.var(lap))

    if not np.isfinite(var):
        printf(f"variance is not finite ({var}); forcing to 0.0")
        var = 0.0

    printf(f"metric variance={var:.6f}")
    printf("=== ripple_metric_debug END ===")
    return var
# ---------- END DEBUG HELPERS ----------