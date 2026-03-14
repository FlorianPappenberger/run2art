"""
scoring.py — Vectorized scoring for Run2Art
=============================================
6-component bidirectional score + coarse proximity score.
All hot paths use NumPy vectorization.
"""

import numpy as np
from geometry import (
    haversine, haversine_matrix, haversine_vector,
    sample_polyline, point_to_segments_vectorized,
)


def bidirectional_score(route, ideal_pts):
    """
    6-component score (lower = better, in metres).
      1. Forward coverage   (ideal→route)     weight 0.30
      2. Reverse detour     (route→ideal)     weight 0.15
      3. Hausdorff          (worst-case)      weight 0.10
      4. Perpendicular      (to segments)     weight 0.20
      5. Turning-angle      (fidelity)        weight 0.15
      6. Length-ratio        (fidelity)        weight 0.10
    """
    if not route or len(route) < 2:
        return 1e9

    n_i = min(80, max(30, len(ideal_pts) * 3))
    n_r = min(150, max(40, len(route)))
    ideal_s = sample_polyline(ideal_pts, n_i)
    route_s = sample_polyline(route, n_r)
    if not ideal_s or not route_s:
        return 1e9

    ia = np.asarray(ideal_s, dtype=np.float64)
    ra = np.asarray(route_s, dtype=np.float64)

    # Full distance matrix (n_i × n_r)
    dist_mat = haversine_matrix(ia[:, 0], ia[:, 1], ra[:, 0], ra[:, 1])

    # 1. Forward: ideal → nearest route point
    fwd_min = dist_mat.min(axis=1)
    fwd_avg = float(fwd_min.mean())
    coverage = float((fwd_min < 100).sum()) / n_i

    # 2. Reverse: route → nearest ideal point
    rev_min = dist_mat.min(axis=0)
    rev_avg = float(rev_min.mean())

    # 3. Hausdorff distance
    hn_i = min(40, len(ideal_pts))
    hn_r = min(60, len(route))
    hs_ideal = sample_polyline(ideal_pts, hn_i)
    hs_route = sample_polyline(route, hn_r)
    hia = np.asarray(hs_ideal, dtype=np.float64)
    hra = np.asarray(hs_route, dtype=np.float64)
    hdist = haversine_matrix(hia[:, 0], hia[:, 1], hra[:, 0], hra[:, 1])
    haus = float(max(hdist.min(axis=1).max(), hdist.min(axis=0).max()))

    # 4. Perpendicular segment distance — VECTORIZED
    step = max(1, n_r // 60)
    perp_pts = ra[::step]
    if len(ia) >= 2:
        perp = float(point_to_segments_vectorized(
            perp_pts, ia[:-1], ia[1:]
        ).mean())
    else:
        perp = 0.0

    # 5. Turning-angle fidelity — VECTORIZED
    angle_penalty = _vectorized_turning_angle_penalty(route, ideal_pts)

    # 6. Length-ratio fidelity
    lr_penalty = _length_ratio_penalty(
        sample_polyline(route, min(20, len(route))),
        sample_polyline(ideal_pts, min(20, len(ideal_pts))),
    )

    # PHASE 1: Reweighted — Hausdorff + turning-angle up, coverage + detour down
    score = (haus * 0.25 + angle_penalty * 0.25 + fwd_avg * 0.20 +
             perp * 0.15 + rev_avg * 0.10 + lr_penalty * 0.05)
    if coverage < 0.75:
        score *= (2.0 - coverage)
    return score


def _vectorized_turning_angle_penalty(route, ideal_pts):
    """Turning-angle fidelity computed with vectorised NumPy ops."""
    if len(ideal_pts) < 3 or len(route) < 3:
        return 0.0

    rk = sample_polyline(route, len(ideal_pts))
    ia = np.asarray(ideal_pts, dtype=np.float64)
    ra = np.asarray(rk, dtype=np.float64)

    n = min(len(ia), len(ra))
    if n < 3:
        return 0.0

    # Vectorised turning angles for both ideal and route
    ideal_angles = _compute_turning_angles(ia[:n])
    route_angles = _compute_turning_angles(ra[:n])

    diffs = np.abs(ideal_angles - route_angles)
    return float(diffs.mean()) * 1.5


def _compute_turning_angles(pts):
    """Compute turning angles at interior points using vectorised atan2.

    Args:
        pts: (N,2) array of lat/lng points

    Returns:
        (N-2,) array of turning angles in degrees at points 1..N-2
    """
    # Vectors from each point to the next
    d1_lat = pts[1:-1, 0] - pts[:-2, 0]
    d1_lng = pts[1:-1, 1] - pts[:-2, 1]
    d2_lat = pts[2:, 0] - pts[1:-1, 0]
    d2_lng = pts[2:, 1] - pts[1:-1, 1]

    a1 = np.arctan2(d1_lat, d1_lng)
    a2 = np.arctan2(d2_lat, d2_lng)

    diff = np.degrees(a2 - a1)
    # Normalise to [-180, 180]
    diff = ((diff + 180) % 360) - 180
    return diff


def _length_ratio_penalty(route_pts, ideal_pts):
    """Compare consecutive-segment length ratios between route and ideal."""
    if len(route_pts) < 2 or len(ideal_pts) < 2:
        return 0.0

    ra = np.asarray(route_pts, dtype=np.float64)
    ia = np.asarray(ideal_pts, dtype=np.float64)

    rl = haversine_vector(ra[:-1, 0], ra[:-1, 1], ra[1:, 0], ra[1:, 1])
    il = haversine_vector(ia[:-1, 0], ia[:-1, 1], ia[1:, 0], ia[1:, 1])

    n = min(len(rl), len(il), 20)
    if n < 2:
        return 0.0

    # Subsample to n segments
    r_idx = np.linspace(0, len(rl) - 1, n).astype(int)
    i_idx = np.linspace(0, len(il) - 1, n).astype(int)
    rs = rl[r_idx]
    iss = il[i_idx]

    # Consecutive ratios
    rs_safe = np.maximum(rs[:-1], 1.0)
    is_safe = np.maximum(iss[:-1], 1.0)
    diffs = np.abs(rs[1:] / rs_safe - iss[1:] / is_safe)

    return float(diffs.mean()) * 15.0


def coarse_proximity_score(G, waypoints, kdtree_data=None):
    """Fast coarse score — proximity + coverage penalty (Phase 9 SDF-inspired)."""
    if G is None:
        return 1e9
    try:
        import osmnx as ox
        wps_a = np.asarray(waypoints, dtype=np.float64)
        if kdtree_data is not None:
            tree, _coords, _nids = kdtree_data
            dists_deg, _idx = tree.query(wps_a)
            dists = dists_deg * 111_000.0
        else:
            nids = ox.nearest_nodes(G, list(wps_a[:, 1]), list(wps_a[:, 0]))
            nlats = np.array([G.nodes[nid]['y'] for nid in nids])
            nlons = np.array([G.nodes[nid]['x'] for nid in nids])
            dists = haversine_vector(wps_a[:, 0], wps_a[:, 1], nlats, nlons)
        base = float(dists.mean() + dists.max() * 0.3)
        # Phase 9: Coverage penalty — penalise placements with road gaps
        uncovered = float((dists > 150.0).sum()) / max(len(dists), 1)
        return base * (1.0 + 2.0 * uncovered)
    except Exception:
        return 1e9
