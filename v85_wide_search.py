"""
v85_wide_search.py — v8.5 Wide-Area Multi-Scale Search
========================================================
Replaces the fixed coarse grid with an adaptive, multi-phase search
that covers:
  - Many more sizes (tiny 0.004 → large 0.060, 16+ scales)
  - Much wider geographic area (up to 5 km from center, 7×7 grid)
  - Finer rotation steps (10° or less)
  - Adaptive refinement: promising regions get denser sub-searches

Also contains the Human Recognizability Scorer for heart shapes
(0–10 scale based on closure, symmetry, deviation, lobes, smoothness).
"""

import math
import sys
import os
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry import (
    shape_to_latlngs, densify, adaptive_densify,
    haversine, haversine_vector,
    sample_polyline,
)


def log(msg):
    try:
        print(msg, file=sys.stderr, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"),
              file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  1. WIDE-AREA MULTI-SCALE COARSE SEARCH
# ═══════════════════════════════════════════════════════════════════════════

DEG_PER_KM_LAT = 0.009
DEG_PER_KM_LNG = 0.012

# --- Scale presets -------------------------------------------------------
# From very small (~400 m span) to very large (~6 km span)
WIDE_SCALES = [
    0.004, 0.006, 0.008,                           # tiny
    0.010, 0.012, 0.015, 0.018,                     # small
    0.020, 0.023, 0.027, 0.032,                     # medium
    0.038, 0.045, 0.052, 0.060,                     # large
]

# --- Rotation presets ----------------------------------------------------
WIDE_ROTATIONS = list(range(0, 360, 10))             # 36 steps


def wide_offsets(km_range=4.0, steps=4):
    """Generate a dense offset grid covering +-km_range.

    With steps=4 this produces values at 0, ±1.33, ±2.67, ±4.0 km
    → 7 values per axis → 49 offset pairs.
    """
    vals = [0.0]
    for s in range(1, steps + 1):
        v = (s / steps) * km_range
        vals.extend([v, -v])
    return [(dlat * DEG_PER_KM_LAT, dlng * DEG_PER_KM_LNG)
            for dlat in vals for dlng in vals]


def _closest_point_km(route_pts, center):
    """Return distance in km from *center* to the closest point in *route_pts*."""
    cos_lat = math.cos(math.radians(center[0]))
    min_d2 = 1e18
    for pt in route_pts:
        dlat = (pt[0] - center[0]) * 111.32
        dlng = (pt[1] - center[1]) * 111.32 * cos_lat
        d2 = dlat * dlat + dlng * dlng
        if d2 < min_d2:
            min_d2 = d2
    return math.sqrt(min_d2)


def wide_coarse_search(G, pts, center, kdtree_data=None,
                       scales=None, rotations=None,
                       km_range=4.0, offset_steps=4,
                       max_displacement_km=5.0,
                       densify_spacing=200, n_top=200):
    """Phase-1 wide coarse search using proximity scoring only.

    Default grid:  36 rotations × 15 scales × 49 offsets = 26,460 combos.
    Candidates whose closest shape point > *max_displacement_km* from
    the original *center* are discarded.
    Returns the top *n_top* (score, rot, scale, center) tuples, sorted.
    """
    from scoring import coarse_proximity_score

    scales = scales or WIDE_SCALES
    rotations = rotations or WIDE_ROTATIONS
    offsets = wide_offsets(km_range, offset_steps)

    total = len(rotations) * len(scales) * len(offsets)
    log(f"[wide-search] Phase 1: {len(rotations)} rots x {len(scales)} "
        f"scales x {len(offsets)} offsets = {total} combos "
        f"(max_disp={max_displacement_km}km)")
    t0 = time.time()

    skipped = 0
    results = []
    for dlat, dlng in offsets:
        c = [center[0] + dlat, center[1] + dlng]
        for sc in scales:
            for rot in rotations:
                wps = shape_to_latlngs(pts, c, sc, rot)
                # Enforce max displacement: closest shape point must be within limit
                if max_displacement_km > 0:
                    if _closest_point_km(wps, center) > max_displacement_km:
                        skipped += 1
                        continue
                d = densify(wps, spacing_m=densify_spacing)
                score = coarse_proximity_score(G, d, kdtree_data=kdtree_data)
                results.append((score, rot, sc, c))

    results.sort(key=lambda x: x[0])
    elapsed = time.time() - t0
    if skipped:
        log(f"[wide-search] Phase 1: skipped {skipped} combos (>{max_displacement_km}km)")
    if results:
        log(f"[wide-search] Phase 1 done in {elapsed:.1f}s  "
            f"kept={len(results)}  best={results[0][0]:.1f}  "
            f"worst-kept={results[min(n_top-1, len(results)-1)][0]:.1f}")
    else:
        log(f"[wide-search] Phase 1 done in {elapsed:.1f}s  NO valid candidates")
    return results[:n_top]


def adaptive_refine(G, pts, coarse_top, kdtree_data=None,
                    rot_delta=5.0, scale_factors=None, offset_km=0.3,
                    offset_sub_steps=2, densify_spacing=150, n_top=50):
    """Phase-2 adaptive refinement around the best coarse candidates.

    For each candidate in *coarse_top*, try small perturbations:
      - Rotations: ±rot_delta in 2.5° steps
      - Scales: ×[0.90, 0.95, 1.00, 1.05, 1.10]
      - Offsets: ±offset_km in sub-steps

    Returns the top *n_top* refined candidates.
    """
    from scoring import coarse_proximity_score

    scale_factors = scale_factors or [0.88, 0.94, 1.00, 1.06, 1.12]
    rot_steps = np.arange(-rot_delta, rot_delta + 0.1, 2.5)
    sub_offsets = wide_offsets(offset_km, offset_sub_steps)

    seen = set()
    results = []
    t0 = time.time()

    for _, rot, sc, c in coarse_top:
        for dr in rot_steps:
            for sf in scale_factors:
                for dlat, dlng in sub_offsets:
                    r2 = (rot + dr) % 360
                    s2 = sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    key = (round(r2, 1), round(s2, 5),
                           round(c2[0], 5), round(c2[1], 5))
                    if key in seen:
                        continue
                    seen.add(key)

                    wps = shape_to_latlngs(pts, c2, s2, r2)
                    d = densify(wps, spacing_m=densify_spacing)
                    score = coarse_proximity_score(G, d,
                                                   kdtree_data=kdtree_data)
                    results.append((score, r2, s2, c2))

    results.sort(key=lambda x: x[0])
    elapsed = time.time() - t0
    log(f"[wide-search] Phase 2: {len(results)} refined combos in {elapsed:.1f}s  "
        f"best={results[0][0]:.1f}")
    return results[:n_top]


def wide_search_pipeline(G, pts, center, kdtree_data=None, config=None):
    """Full wide-search pipeline: coarse → refine → fine routing.

    Config keys:
        wide_km_range:   float — search radius in km (default 4.0)
        wide_offset_steps: int — offset grid density (default 4)
        max_displacement_km: float — max km from center to closest route point (default 5.0)
        wide_n_coarse:   int — top N from phase 1 (default 200)
        wide_n_refine:   int — top N candidates entering phase 2 (default 30)
        wide_n_fine:     int — top N entering routing phase (default 10)

    Returns: list of (score, rotation, scale, center) sorted by routed score.
    """
    from v83_enhancements import fit_and_score_v83

    config = config or {}
    km_range = config.get('wide_km_range', 4.0)
    offset_steps = config.get('wide_offset_steps', 4)
    max_disp = config.get('max_displacement_km', 5.0)
    n_coarse = config.get('wide_n_coarse', 200)
    n_refine_input = config.get('wide_n_refine', 30)
    n_fine = config.get('wide_n_fine', 10)

    # Phase 1: wide coarse
    coarse = wide_coarse_search(
        G, pts, center, kdtree_data=kdtree_data,
        km_range=km_range, offset_steps=offset_steps,
        max_displacement_km=max_disp,
        n_top=n_coarse)

    # Phase 2: adaptive refinement on top coarse candidates
    refined = adaptive_refine(
        G, pts, coarse[:n_refine_input], kdtree_data=kdtree_data,
        rot_delta=5.0, offset_km=0.3, n_top=n_fine * 3)

    # Deduplicate — keep top n_fine unique parameter sets
    seen = set()
    candidates = []
    for _, rot, sc, c in refined:
        key = (round(rot, 0), round(sc, 4), round(c[0], 4), round(c[1], 4))
        if key not in seen:
            seen.add(key)
            candidates.append((rot, sc, c))
        if len(candidates) >= n_fine:
            break

    log(f"[wide-search] Phase 3: routing {len(candidates)} candidates")

    # Phase 3: actual routing + scoring
    flags = {k: v for k, v in config.items()
             if k not in ('wide_km_range', 'wide_offset_steps',
                          'wide_n_coarse', 'wide_n_refine', 'wide_n_fine',
                          'wide_search')}

    best_score, best_route, best_params = 1e9, None, {}
    t0 = time.time()

    for i, (rot, sc, c) in enumerate(candidates):
        # Small rotation/scale jitter around each candidate
        for dr in [0, -3, 3, -7, 7]:
            for sf in [1.0, 0.93, 1.07]:
                r2 = (rot + dr) % 360
                s2 = sc * sf
                score, route = fit_and_score_v83(
                    G, pts, r2, s2, c,
                    config=flags, kdtree_data=kdtree_data)
                if route and score < best_score:
                    best_score = score
                    best_route = route
                    best_params = {"rotation": r2, "scale": s2, "center": c}

    elapsed = time.time() - t0
    log(f"[wide-search] Phase 3 done in {elapsed:.1f}s  best={best_score:.1f}")
    return best_score, best_route, best_params


# ═══════════════════════════════════════════════════════════════════════════
#  2. HUMAN RECOGNIZABILITY SCORER  v2  (0 – 10)
# ═══════════════════════════════════════════════════════════════════════════
#
# Redesigned for better perceptual alignment.
#
# Key insight: hearts are recognised by a *combination* of critical features,
# not just template matching.  Humans instantly see:
#   1. Two rounded lobes at top                   (25%)
#   2. A V-shaped indent/cleft between the lobes  (20%)
#   3. A pointed cusp at the bottom               (15%)
#   4. Bilateral left-right symmetry              (20%)
#   5. Overall silhouette match (multi-template)  (10%)
#   6. Smooth, continuous curves (not jagged)     (10%)
#
# Multi-template: compare against MULTIPLE heart parametric curves at
# different aspect ratios/styles and use the BEST match.
#
# Score range: 0.0 (not a heart at all) → 10.0 (perfect heart)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_heart_templates():
    """Generate multiple heart templates from different parametric equations.

    Returns list of Nx2 arrays normalised to [0,1]×[0,1], y=0 at top.
    """
    templates = []
    t = np.linspace(0, 2 * np.pi, 200)

    # --- Template 1: Classic parametric (MathWorld #5) ---
    # x = 16 sin³(t),  y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
    x1 = 16 * np.sin(t) ** 3
    y1 = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    templates.append(np.column_stack([x1, -y1]))  # flip y so top is y=0

    # --- Template 2: Modified nephroid heart (MathWorld #7) ---
    x2 = -np.sqrt(2) * np.sin(t) ** 3
    y2 = 2 * np.cos(t) - np.cos(t) ** 2 - np.cos(t) ** 3
    templates.append(np.column_stack([x2, -y2]))

    # --- Template 3: Wide/flat heart (stretched x by 30%) ---
    x3 = 16 * np.sin(t) ** 3 * 1.3
    y3 = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    templates.append(np.column_stack([x3, -y3]))

    # --- Template 4: Tall/narrow heart (stretched y by 30%) ---
    x4 = 16 * np.sin(t) ** 3
    y4 = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) * 1.3
    templates.append(np.column_stack([x4, -y4]))

    # --- Template 5: Shallow indent heart ---
    x5 = 16 * np.sin(t) ** 3
    y5_base = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    indent_reduce = 3.0 * np.exp(-0.5 * ((t - 0) / 0.3) ** 2) + \
                    3.0 * np.exp(-0.5 * ((t - 2 * np.pi) / 0.3) ** 2)
    y5 = y5_base - indent_reduce
    templates.append(np.column_stack([x5, -y5]))

    # --- Template 6: Hand-drawn polygon style ---
    hand = np.array([
        [0.500, 0.050],
        [0.580, 0.000], [0.700, 0.010], [0.830, 0.050],
        [0.950, 0.180], [1.000, 0.330],
        [0.960, 0.480], [0.870, 0.620], [0.750, 0.750],
        [0.620, 0.870], [0.500, 1.000],
        [0.380, 0.870], [0.250, 0.750], [0.130, 0.620],
        [0.040, 0.480], [0.000, 0.330],
        [0.050, 0.180], [0.170, 0.050],
        [0.300, 0.010], [0.420, 0.000], [0.500, 0.050],
    ], dtype=np.float64)
    templates.append(hand)

    # --- Template 7: GPS-art realistic (rounder lobes, slight indent) ---
    gps_real = np.array([
        [0.500, 0.080],
        [0.600, 0.020], [0.720, 0.000], [0.850, 0.040],
        [0.950, 0.160], [1.000, 0.320],
        [0.980, 0.500], [0.900, 0.650], [0.780, 0.780],
        [0.640, 0.900], [0.500, 1.000],
        [0.360, 0.900], [0.220, 0.780], [0.100, 0.650],
        [0.020, 0.500], [0.000, 0.320],
        [0.050, 0.160], [0.150, 0.040],
        [0.280, 0.000], [0.400, 0.020], [0.500, 0.080],
    ], dtype=np.float64)
    templates.append(gps_real)

    # Normalise all to [0,1]×[0,1]
    normalised = []
    for pts in templates:
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = max(mx[0] - mn[0], mx[1] - mn[1])
        if span < 1e-12:
            continue
        norm = (pts - mn) / span
        normalised.append(norm)

    return normalised


_HEART_TEMPLATES = _generate_heart_templates()

# Keep single template reference for backward compat
_IDEAL_HEART = _HEART_TEMPLATES[0] if _HEART_TEMPLATES else np.zeros((2, 2))


def _normalise_to_unit(pts, is_gps=False):
    """Translate and uniformly scale points to fit in [0,1]×[0,1].

    If is_gps=True, input is [lat, lng]: swap to [x, y] and flip y
    so that north (high lat) maps to y≈0 (top of heart).
    If is_gps=False, input is already [x, y] with y=0 at top.
    """
    pa = np.asarray(pts, dtype=np.float64)
    if len(pa) < 2:
        return pa
    mn = pa.min(axis=0)
    mx = pa.max(axis=0)
    span = max(mx[0] - mn[0], mx[1] - mn[1])
    if span < 1e-12:
        return pa - mn
    norm = (pa - mn) / span
    if is_gps:
        # [lat, lng] → [x, y]: x = lng (col 1), y = 1 - lat_norm (col 0 flipped)
        return np.column_stack([norm[:, 1], 1.0 - norm[:, 0]])
    return norm


def _closure_score(pts, is_gps=False):
    """1.0 if first/last point are within 0.01 relative units, else 0.5."""
    pa = np.asarray(pts, dtype=np.float64)
    if len(pa) < 2:
        return 0.0
    dist = np.linalg.norm(pa[0] - pa[-1])
    span = max(pa.max(axis=0) - pa.min(axis=0))
    if span < 1e-12:
        return 0.0
    rel = dist / span
    if rel < 0.01:
        return 1.0
    if rel < 0.05:
        return 0.75
    return 0.5


def _resample_path(pts, n=100):
    """Resample a path to *n* evenly-spaced points along cumulative arc-length."""
    pa = np.asarray(pts, dtype=np.float64)
    if len(pa) <= 2:
        return pa
    diffs = np.diff(pa, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-12:
        return pa
    targets = np.linspace(0, total, n)
    resampled = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum, t, side='right') - 1
        idx = min(idx, len(pa) - 2)
        seg = seg_lens[idx]
        frac = (t - cum[idx]) / seg if seg > 1e-12 else 0.0
        resampled[i] = pa[idx] + frac * diffs[idx]
    return resampled


def _symmetry_score(pts, is_gps=False):
    """Bilateral symmetry about the geometric center axis (x-axis mirror).

    Uses geometric midpoint (not mean) for the mirror axis, and resamples
    to uniform spacing so point-density doesn't bias the split.
    """
    pa = _normalise_to_unit(pts, is_gps=is_gps)
    if len(pa) < 4:
        return 0.0

    # Resample to uniform spacing so dense road segments don't bias the axis
    pa = _resample_path(pa, n=200)

    # Geometric center — not mean — so asymmetric point density doesn't shift it
    x_center = (pa[:, 0].min() + pa[:, 0].max()) / 2.0
    width = pa[:, 0].max() - pa[:, 0].min()
    if width < 1e-12:
        return 0.0

    left = pa[pa[:, 0] <= x_center]
    right = pa[pa[:, 0] >= x_center]
    if len(left) < 2 or len(right) < 2:
        return 0.0

    # Mirror right side across x_center
    right_mirrored = right.copy()
    right_mirrored[:, 0] = 2 * x_center - right_mirrored[:, 0]

    # Directed Hausdorff from left to mirrored-right and vice versa
    d_fwd = directed_hausdorff(left, right_mirrored)[0]
    d_bwd = directed_hausdorff(right_mirrored, left)[0]
    avg_dist = (d_fwd + d_bwd) / 2.0

    score = max(0.0, 1.0 - avg_dist / width)
    return float(score)


def _chamfer_distance(A, B):
    """Approximate Chamfer distance between point sets A and B."""
    if len(A) == 0 or len(B) == 0:
        return 1.0
    tree_b = cKDTree(B)
    d_a2b, _ = tree_b.query(A)
    tree_a = cKDTree(A)
    d_b2a, _ = tree_a.query(B)
    return float((d_a2b.mean() + d_b2a.mean()) / 2.0)


def _deviation_score(pts, is_gps=False):
    """Best Chamfer match against ALL heart templates (multi-template).

    Returns the best (lowest chamfer) match across 7 different heart shapes.
    """
    norm = _normalise_to_unit(pts, is_gps=is_gps)
    norm_resampled = _resample_path(norm, n=150)

    best = 1.0
    for tmpl in _HEART_TEMPLATES:
        tmpl_resampled = _resample_path(tmpl, n=150)
        chamfer = _chamfer_distance(norm_resampled, tmpl_resampled)
        if chamfer < best:
            best = chamfer

    tolerance = 0.18  # tighter than before (was 0.25), multi-template is more forgiving
    score = max(0.0, 1.0 - best / tolerance)
    return float(score)


def _lobes_score(pts, is_gps=False):
    """Detect two distinct rounded upper bulges.

    Uses the angular profile: walk around the normalised shape, measure the
    angle from centroid.  A heart has two prominent "peaks" (max radius)
    in the top half, separated left/right of the midline.  Scoring:
      - 2 peaks on opposite sides → 1.0
      - 2 peaks on same side or 1 peak → partial
      - 0 peaks → 0.0
    Also checks lobe roundedness (curvature at peak regions).
    """
    pa = _normalise_to_unit(pts, is_gps=is_gps)
    pa = _resample_path(pa, n=200)
    if len(pa) < 20:
        return 0.0

    # Centroid
    cx = (pa[:, 0].min() + pa[:, 0].max()) / 2.0
    cy = (pa[:, 1].min() + pa[:, 1].max()) / 2.0

    # Compute angle from centroid and radius for each point
    dx = pa[:, 0] - cx
    dy = pa[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx ** 2 + dy ** 2)

    # Focus on the upper portion: y < cy (angles roughly -pi to 0 in our coord system)
    # Upper = where dy < 0 (y < centroid, remember y=0 is top)
    upper_mask = pa[:, 1] < cy
    if upper_mask.sum() < 10:
        return 0.0

    # Sort upper points by x for a profile
    upper = pa[upper_mask]
    order = upper[:, 0].argsort()
    upper = upper[order]

    # Create binned top-edge profile: for x-bins, find min y (highest point)
    n_bins = 25
    x_min, x_max = pa[:, 0].min(), pa[:, 0].max()
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    profile_x, profile_y = [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (upper[:, 0] >= lo) & (upper[:, 0] <= hi)
        else:
            mask = (upper[:, 0] >= lo) & (upper[:, 0] < hi)
        if mask.any():
            profile_x.append((lo + hi) / 2)
            profile_y.append(upper[mask, 1].min())  # min y = topmost

    if len(profile_y) < 5:
        return 0.0

    profile_y = np.array(profile_y)
    profile_x = np.array(profile_x)

    # Smooth
    kernel = np.ones(3) / 3
    if len(profile_y) > 5:
        y_smooth = np.convolve(profile_y, kernel, mode='same')
    else:
        y_smooth = profile_y

    # Find local minima in y (= peaks going up, since y=0 is top)
    peaks = []
    for i in range(1, len(y_smooth) - 1):
        if y_smooth[i] < y_smooth[i - 1] and y_smooth[i] < y_smooth[i + 1]:
            peaks.append((profile_x[i], y_smooth[i]))

    # Also check the two endpoints as potential peaks
    if len(y_smooth) >= 3:
        if y_smooth[0] < y_smooth[1]:
            peaks.insert(0, (profile_x[0], y_smooth[0]))
        if y_smooth[-1] < y_smooth[-2]:
            peaks.append((profile_x[-1], y_smooth[-1]))

    x_center = (x_min + x_max) / 2.0
    n_peaks = len(peaks)

    if n_peaks >= 2:
        # Find best left and right peaks
        left_peaks = [p for p in peaks if p[0] < x_center]
        right_peaks = [p for p in peaks if p[0] > x_center]
        if left_peaks and right_peaks:
            # Best (highest = lowest y) on each side
            lp = min(left_peaks, key=lambda p: p[1])
            rp = min(right_peaks, key=lambda p: p[1])
            # Check peaks are reasonably high (in top 40% of shape)
            shape_height = pa[:, 1].max() - pa[:, 1].min()
            peak_depth = max(lp[1] - pa[:, 1].min(), rp[1] - pa[:, 1].min())
            if peak_depth < 0.4 * shape_height:
                # Check lobe symmetry (similar heights)
                height_ratio = min(lp[1], rp[1]) / max(lp[1], rp[1]) if max(lp[1], rp[1]) > 0.01 else 1.0
                return min(1.0, 0.7 + 0.3 * height_ratio)
            return 0.5
        return 0.4

    if n_peaks == 1:
        return 0.3

    return 0.0


def _indent_score(pts, is_gps=False):
    """Measure the V-shaped indent/cleft between the two lobes.

    The indent is THE defining feature that distinguishes a heart from
    a circle or an oval.  We measure:
      - Whether there's a local y-maximum (dip downward) at top center
      - How deep that dip is relative to the lobe peaks
    """
    pa = _normalise_to_unit(pts, is_gps=is_gps)
    pa = _resample_path(pa, n=200)
    if len(pa) < 20:
        return 0.0

    x_min, x_max = pa[:, 0].min(), pa[:, 0].max()
    x_center = (x_min + x_max) / 2.0
    width = x_max - x_min
    if width < 1e-12:
        return 0.0

    # Top-edge profile in fine bins
    n_bins = 30
    upper_mask = pa[:, 1] < np.median(pa[:, 1])
    upper = pa[upper_mask]
    if len(upper) < 10:
        return 0.0

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    profile = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (upper[:, 0] >= lo) & (upper[:, 0] <= hi)
        else:
            mask = (upper[:, 0] >= lo) & (upper[:, 0] < hi)
        if mask.any():
            profile.append(((lo + hi) / 2, upper[mask, 1].min()))
    
    if len(profile) < 5:
        return 0.0

    px = np.array([p[0] for p in profile])
    py = np.array([p[1] for p in profile])

    # Find the center region (middle 40% of width)
    center_lo = x_center - 0.2 * width
    center_hi = x_center + 0.2 * width
    center_mask = (px >= center_lo) & (px <= center_hi)

    # Find left/right lobe regions (outer quarters)
    left_mask = px < center_lo
    right_mask = px > center_hi

    if not center_mask.any() or not left_mask.any() or not right_mask.any():
        return 0.0

    # The indent = max y in center (deepest point = away from top)
    # The lobes = min y on left/right (highest point = nearest top)
    center_deepest = py[center_mask].max()  # max y = lowest point in center
    left_highest = py[left_mask].min()      # min y = highest on left
    right_highest = py[right_mask].min()    # min y = highest on right
    lobe_avg = (left_highest + right_highest) / 2.0

    # Indent depth = how much lower the center dips vs the lobe peaks
    indent_depth = center_deepest - lobe_avg
    shape_height = pa[:, 1].max() - pa[:, 1].min()

    if shape_height < 1e-12:
        return 0.0

    relative_depth = indent_depth / shape_height

    # A good heart indent is 5-20% of shape height
    if relative_depth >= 0.15:
        return 1.0
    elif relative_depth >= 0.08:
        return 0.8
    elif relative_depth >= 0.03:
        return 0.5
    elif relative_depth > 0.01:
        return 0.3
    return 0.0


def _cusp_score(pts, is_gps=False):
    """Measure the pointed cusp at the bottom of the heart.

    A heart should come to a point at the bottom (not be rounded).
    We measure the "sharpness" of the bottommost region.
    """
    pa = _normalise_to_unit(pts, is_gps=is_gps)
    pa = _resample_path(pa, n=200)
    if len(pa) < 20:
        return 0.0

    # Bottom region: points in the bottom 20% of y range
    y_max = pa[:, 1].max()
    y_min = pa[:, 1].min()
    y_range = y_max - y_min
    if y_range < 1e-12:
        return 0.0

    bottom_thresh = y_max - 0.2 * y_range
    bottom = pa[pa[:, 1] >= bottom_thresh]
    if len(bottom) < 3:
        return 0.0

    # Width at bottom vs width at middle
    x_width_bottom = bottom[:, 0].max() - bottom[:, 0].min()
    total_width = pa[:, 0].max() - pa[:, 0].min()

    if total_width < 1e-12:
        return 0.0

    narrowness = 1.0 - (x_width_bottom / total_width)

    # Also check the very bottom (bottom 5%) is even narrower
    very_bottom_thresh = y_max - 0.05 * y_range
    very_bottom = pa[pa[:, 1] >= very_bottom_thresh]
    if len(very_bottom) >= 2:
        vb_width = very_bottom[:, 0].max() - very_bottom[:, 0].min()
        tip_narrowness = 1.0 - (vb_width / total_width)
    else:
        tip_narrowness = 1.0  # single point = perfect tip

    # Combine: overall narrowness and tip sharpness
    score = 0.4 * narrowness + 0.6 * tip_narrowness
    return float(max(0.0, min(1.0, score)))


def _smoothness_score(pts, is_gps=False):
    """Evaluate how smooth/curved the outline is (vs jagged)."""
    pa = np.asarray(pts, dtype=np.float64)
    if len(pa) < 4:
        return 0.0

    # Segment vectors
    diffs = np.diff(pa, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    # Angular change between consecutive segments
    d_angles = np.diff(angles)
    # Wrap to [-pi, pi]
    d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi

    avg_change = np.abs(d_angles).mean()

    # Benchmark: 90° = pi/2 is a very jagged shape → score 0
    benchmark = np.pi / 2
    score = max(0.0, 1.0 - avg_change / benchmark)
    return float(score)


def _heart_component_scores(pts, is_gps=False):
    """Return the recognizability component scores used by HR scoring."""
    pa = np.asarray(pts, dtype=np.float64)
    if len(pa) < 10:
        return None

    return {
        'closure': _closure_score(pa, is_gps=is_gps),
        'symmetry': _symmetry_score(pa, is_gps=is_gps),
        'deviation': _deviation_score(pa, is_gps=is_gps),
        'lobes': _lobes_score(pa, is_gps=is_gps),
        'indent': _indent_score(pa, is_gps=is_gps),
        'cusp': _cusp_score(pa, is_gps=is_gps),
        'smoothness': _smoothness_score(pa, is_gps=is_gps),
    }


def _route_retrace_metrics(route):
    """Estimate how much the route visually retraces itself."""
    if not route or len(route) < 2:
        return {'seg_repeat': 0.0, 'pt_repeat': 0.0}

    rounded = [tuple(round(float(coord), 7) for coord in pt) for pt in route]
    unique_pts = len(set(rounded))
    pt_repeat = 1.0 - unique_pts / max(len(rounded), 1)

    segments = [tuple(sorted((a, b))) for a, b in zip(rounded, rounded[1:])]
    unique_segments = len(set(segments)) if segments else 0
    seg_repeat = 1.0 - unique_segments / max(len(segments), 1) if segments else 0.0

    return {
        'seg_repeat': float(max(0.0, seg_repeat)),
        'pt_repeat': float(max(0.0, pt_repeat)),
    }


def heart_recognizability_score(pts, weights=None, is_gps=False):
    """Score how recognizable a shape is as a heart (0.0 – 10.0).

    v2: Uses 6 perceptual components with multi-template matching,
    indent detection, and bottom cusp analysis.

    Components:
        lobes (25%):     Two rounded upper bulges
        indent (20%):    V-cleft between lobes
        cusp (15%):      Pointed bottom
        symmetry (20%):  Bilateral mirror symmetry
        deviation (10%): Best multi-template Chamfer match
        smoothness (10%): Gentle curves, not jagged

    Returns:
        (score: float, explanation: str)
    """
    components = _heart_component_scores(pts, is_gps=is_gps)
    if components is None:
        return (0.0, "Too few points (<10)")

    closure = components['closure']
    symmetry = components['symmetry']
    deviation = components['deviation']
    lobes = components['lobes']
    indent = components['indent']
    cusp = components['cusp']
    smoothness = components['smoothness']

    # Closure is a gate: if not closed, max score is 5.0
    closure_gate = 1.0 if closure >= 0.75 else 0.5

    w = {
        'lobes':      0.25,
        'indent':     0.20,
        'cusp':       0.15,
        'symmetry':   0.20,
        'deviation':  0.10,
        'smoothness': 0.10,
    }
    if weights:
        w.update(weights)

    weighted = (
        w['lobes']      * lobes +
        w['indent']     * indent +
        w['cusp']       * cusp +
        w['symmetry']   * symmetry +
        w['deviation']  * deviation +
        w['smoothness'] * smoothness
    )

    score = round(min(10.0, max(0.0, weighted * 10.0 * closure_gate)), 1)

    explanation = (
        f"lobes={lobes:.2f} indent={indent:.2f} cusp={cusp:.2f} "
        f"sym={symmetry:.2f} dev={deviation:.2f} smooth={smoothness:.2f} "
        f"closure={closure:.2f} => {score}/10"
    )

    return (score, explanation)


# ═══════════════════════════════════════════════════════════════════════════
#  ABSTRACT SMOOTH LINE SCORER
# ═══════════════════════════════════════════════════════════════════════════
#
# The idea: GPS routes follow roads (jagged, many turns).  A human looking
# at a GPS art heart doesn't perceive every road wiggle — they mentally
# "smooth out" the route into an idealised outline.  So we do the same:
#
# 1. Resample route to uniform spacing
# 2. Extract the *convex-ish outline* (alpha shape / concave hull)
# 3. Smooth the outline with a Savitzky-Golay or moving-average filter
# 4. Score that smoothed abstract shape
#
# This "abstract HR" score better matches what a human would judge.
# ═══════════════════════════════════════════════════════════════════════════

def _extract_outline(pts, n_angular_bins=72):
    """Extract the outer outline of a point cloud using angular sweeping.

    Divides 360° around the centroid into *n_angular_bins* sectors and picks
    the outermost point in each sector.  This gives a concave-hull-like
    outline without heavy dependencies.
    """
    if len(pts) < 10:
        return pts

    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0

    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    bin_edges = np.linspace(-np.pi, np.pi, n_angular_bins + 1)
    outline = []

    for i in range(n_angular_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if mask.any():
            # Pick the point with maximum radius in this angular bin
            idx = np.where(mask)[0]
            best = idx[radii[idx].argmax()]
            outline.append(pts[best])

    if len(outline) < 5:
        return pts

    outline = np.array(outline)
    # Close the loop
    if np.linalg.norm(outline[0] - outline[-1]) > 1e-12:
        outline = np.vstack([outline, outline[0:1]])

    return outline


def _smooth_outline(pts, window=7):
    """Smooth a closed outline using a moving average.

    Uses circular padding to handle the wraparound of a closed curve.
    """
    if len(pts) < window + 2:
        return pts

    # Circular padding
    pad = window // 2
    padded = np.vstack([pts[-pad:], pts, pts[:pad]])

    smoothed = np.empty_like(pts)
    for i in range(len(pts)):
        j = i + pad
        smoothed[i] = padded[j - pad:j + pad + 1].mean(axis=0)

    return smoothed


def abstract_heart_score(pts, is_gps=False):
    """Score the *abstract* (smoothed outline) version of the shape.

    Steps:
    1. Normalise and resample the route densely
    2. Extract outer outline via angular sweep
    3. Smooth the outline (moving average)
    4. Score the smoothed outline with the full heart scorer

    Returns (score, explanation, smooth_outline_pts)
    """
    pa = _normalise_to_unit(pts, is_gps=is_gps)
    if len(pa) < 10:
        return (0.0, "Too few points", pa)

    # Dense resample
    pa = _resample_path(pa, n=300)

    # Extract outline
    outline = _extract_outline(pa, n_angular_bins=72)

    # Smooth
    smooth = _smooth_outline(outline, window=5)

    # Score the smoothed outline (already normalised, not GPS)
    score, explain = heart_recognizability_score(smooth, is_gps=False)

    return (score, f"abstract: {explain}", smooth)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: score recognizability from a route
# ═══════════════════════════════════════════════════════════════════════════

def route_heart_recognizability(route, n_sample=100):
    """Compute heart recognizability for a GPS route.

    Returns (combined_score, explanation) where combined_score averages
    the raw route score (40%) and the abstract smooth outline score (60%),
    reflecting how humans perceive GPS art.
    """
    sampled = sample_polyline(route, n_sample)

    # Raw route score
    raw_score, raw_explain = heart_recognizability_score(sampled, is_gps=True)

    # Abstract smooth outline score
    abs_score, abs_explain, _ = abstract_heart_score(sampled, is_gps=True)

    # Combined: abstract is weighted more (humans see the outline, not the road wiggles)
    combined = round(0.4 * raw_score + 0.6 * abs_score, 1)

    explanation = (
        f"raw={raw_score} abstract={abs_score} "
        f"combined={combined}/10 | {raw_explain}"
    )

    return (combined, explanation)


def route_heart_recognizability_v2(route, routing_score=None, mode=None,
                                   n_sample=100):
    """Manual-aligned secondary heart score.

    Keeps the original HR scorer intact, but adds a second score that better
    reflects the user's manual ratings by penalising retracing/interior clutter,
    reducing template bias, and preferring the raw route over the abstracted
    outline.
    """
    sampled = sample_polyline(route, n_sample)
    if not sampled or len(sampled) < 10:
        return (0.0, "Too few points (<10)")

    raw_score, _raw_explain = heart_recognizability_score(sampled, is_gps=True)
    abs_score, _abs_explain, _ = abstract_heart_score(sampled, is_gps=True)

    components = _heart_component_scores(sampled, is_gps=True)
    retrace = _route_retrace_metrics(route)
    indent_pref = max(0.0, 1.0 - abs(components['indent'] - 0.45) / 0.45)
    route_penalty = 0.012 * max(float(routing_score or 0.0), 0.0)
    fit_bonus = 0.7 if mode == 'fit' else 0.0

    score = (
        0.35 * raw_score +
        0.10 * abs_score +
        2.8 * components['symmetry'] +
        1.6 * components['lobes'] +
        1.1 * components['cusp'] +
        0.8 * indent_pref -
        2.5 * components['deviation'] -
        1.2 * components['smoothness'] -
        6.0 * retrace['seg_repeat'] -
        2.5 * retrace['pt_repeat'] -
        route_penalty +
        fit_bonus
    )
    score = round(min(10.0, max(1.0, score)), 1)

    explanation = (
        f"raw={raw_score:.1f} abstract={abs_score:.1f} "
        f"manual_v2={score:.1f}/10 | sym={components['symmetry']:.2f} "
        f"lobes={components['lobes']:.2f} cusp={components['cusp']:.2f} "
        f"indent_pref={indent_pref:.2f} dev={components['deviation']:.2f} "
        f"smooth={components['smoothness']:.2f} "
        f"seg_repeat={retrace['seg_repeat']:.2f} "
        f"pt_repeat={retrace['pt_repeat']:.2f} "
        f"route_pen={route_penalty:.2f} fit_bonus={fit_bonus:.2f}"
    )

    return (score, explanation)
