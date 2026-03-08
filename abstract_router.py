"""
abstract_router.py — v8.1-Abstract "Symmetry-First" Router
=============================================================
Eliminates trunks by treating bilateral symmetry as a HARD constraint.
Routes are scored on mirror-error, trunk penalty, and curvature consistency
rather than pure Euclidean proximity.

AGENT 'SPATIAL'  — extract_landmarks()      → Cusp, Apex, Lobe-Arc zones
AGENT 'GRAPH'    — AbstractRouter class      → Symmetry-weighted cost + cycle search
AGENT 'VISION'   — SymmetryScorer / abstract_score() → Mirror + trunk + Fréchet
"""

import sys
import math
import heapq
import numpy as np
from scipy.spatial import cKDTree

from geometry import (
    haversine, haversine_vector, haversine_matrix,
    point_to_segment_dist, point_to_segments_vectorized,
    sample_polyline, adaptive_densify, compute_tangent_field,
    detect_sharp_vertices, normalize_to_unit_box, edge_bearing,
)
from scoring_v8 import discrete_frechet_fast, frechet_score


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 'SPATIAL' — GEOMETRIC INVARIANT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _curvature_at_points(pts):
    """Signed curvature (turning angle in degrees) at each interior point."""
    a = np.asarray(pts, dtype=np.float64)
    if len(a) < 3:
        return np.zeros(max(len(a), 0))
    d1 = a[1:-1] - a[:-2]
    d2 = a[2:] - a[1:-1]
    ang1 = np.arctan2(d1[:, 1], d1[:, 0])
    ang2 = np.arctan2(d2[:, 1], d2[:, 0])
    diff = np.degrees(ang2 - ang1)
    diff = ((diff + 180) % 360) - 180
    result = np.zeros(len(a))
    result[1:-1] = diff
    return result


def _turning_angle_chain(pts, n_sample=60):
    """Normalized turning-angle chain for a polyline.
    Returns (n_sample-2,) array of turning angles in degrees.
    """
    s = sample_polyline(pts, min(n_sample, len(pts)))
    a = np.asarray(s, dtype=np.float64)
    if len(a) < 3:
        return np.zeros(0)
    d1 = a[1:-1] - a[:-2]
    d2 = a[2:] - a[1:-1]
    ang1 = np.arctan2(d1[:, 1], d1[:, 0])
    ang2 = np.arctan2(d2[:, 1], d2[:, 0])
    diff = np.degrees(ang2 - ang1)
    return ((diff + 180) % 360) - 180


def extract_landmarks(ideal_line):
    """Identify topological landmarks: Cusp, Apex, and Lobe-Arc zones.

    Returns dict with:
      'cusp':  {'index': int, 'coord': [lat,lng], 'deflection': float}
      'apex':  {'index': int, 'coord': [lat,lng], 'deflection': float}
      'lobes': [{'start': int, 'end': int, 'side': 'left'|'right',
                 'mean_curvature': float, 'arc_center': [lat,lng]}, ...]
      'axis':  {'origin': [lat,lng], 'direction': [dlat,dlng]}
    """
    pts = np.asarray(ideal_line, dtype=np.float64)
    n = len(pts)
    if n < 6:
        return {'cusp': None, 'apex': None, 'lobes': [], 'axis': None}

    curvature = _curvature_at_points(pts)

    # --- Apex: sharpest single-point turn (bottom of heart) ---
    abs_curv = np.abs(curvature)
    apex_idx = int(np.argmax(abs_curv))

    # --- Cusp: second-sharpest turn, far from apex ---
    min_sep = max(n // 4, 3)
    abs_curv_masked = abs_curv.copy()
    lo = max(0, apex_idx - min_sep)
    hi = min(n, apex_idx + min_sep + 1)
    abs_curv_masked[lo:hi] = 0
    cusp_idx = int(np.argmax(abs_curv_masked))

    # Ensure cusp is "above" apex (higher latitude for heart)
    if pts[cusp_idx, 0] < pts[apex_idx, 0]:
        cusp_idx, apex_idx = apex_idx, cusp_idx

    cusp_info = {
        'index': cusp_idx,
        'coord': pts[cusp_idx].tolist(),
        'deflection': float(abs_curv[cusp_idx]),
    }
    apex_info = {
        'index': apex_idx,
        'coord': pts[apex_idx].tolist(),
        'deflection': float(abs_curv[apex_idx]),
    }

    # --- Symmetry axis: unit vector from cusp to apex ---
    axis_vec = pts[apex_idx] - pts[cusp_idx]
    axis_len = np.linalg.norm(axis_vec)
    if axis_len > 0:
        axis_dir = (axis_vec / axis_len).tolist()
    else:
        axis_dir = [-1.0, 0.0]
    axis_info = {
        'origin': pts[cusp_idx].tolist(),
        'direction': axis_dir,
    }

    # --- Lobe Arcs ---
    if cusp_idx < apex_idx:
        left_range = range(cusp_idx, apex_idx + 1)
        right_range = list(range(apex_idx, n)) + list(range(0, cusp_idx + 1))
    else:
        left_range = list(range(cusp_idx, n)) + list(range(0, apex_idx + 1))
        right_range = range(apex_idx, cusp_idx + 1)

    lobes = []
    for side, idx_range in [('left', left_range), ('right', right_range)]:
        idxs = list(idx_range)
        if len(idxs) < 3:
            continue
        curv_vals = np.abs(curvature[idxs])
        peak_pos = int(np.argmax(curv_vals))
        peak_val = curv_vals[peak_pos]
        if peak_val < 5.0:
            continue
        start_pos, end_pos = peak_pos, peak_pos
        while start_pos > 0 and curv_vals[start_pos - 1] > peak_val * 0.3:
            start_pos -= 1
        while end_pos < len(idxs) - 1 and curv_vals[end_pos + 1] > peak_val * 0.3:
            end_pos += 1

        arc_pts = pts[idxs[start_pos]:idxs[end_pos] + 1] if idxs[end_pos] >= idxs[start_pos] else pts[idxs[start_pos]:]
        arc_center = arc_pts.mean(axis=0).tolist() if len(arc_pts) > 0 else pts[idxs[peak_pos]].tolist()
        lobes.append({
            'start': idxs[start_pos],
            'end': idxs[end_pos],
            'side': side,
            'mean_curvature': float(curv_vals[start_pos:end_pos + 1].mean()),
            'arc_center': arc_center,
        })

    log(f"[abstract-spatial] Landmarks: cusp=idx{cusp_idx} "
        f"defl={curvature[cusp_idx]:.1f}deg, apex=idx{apex_idx} "
        f"defl={curvature[apex_idx]:.1f}deg, lobes={len(lobes)}")

    return {'cusp': cusp_info, 'apex': apex_info, 'lobes': lobes, 'axis': axis_info}


# ═══════════════════════════════════════════════════════════════════════════
#  SYMMETRY SCORER
# ═══════════════════════════════════════════════════════════════════════════

class SymmetryScorer:
    """Bilateral symmetry scorer.

    The symmetry axis runs from Cusp (top) to Apex (bottom).
    Points are classified as left/right of this axis, and the route
    is scored by comparing the left lobe with the reflected right lobe.
    """

    def __init__(self, landmarks, ideal_pts):
        self.landmarks = landmarks
        self.ideal_pts = np.asarray(ideal_pts, dtype=np.float64)
        self._setup_axis()

    def _setup_axis(self):
        """Set up the reflection axis in normalised coordinates."""
        cusp = self.landmarks.get('cusp')
        apex = self.landmarks.get('apex')
        if cusp is None or apex is None:
            self.axis_origin = self.ideal_pts.mean(axis=0)
            self.axis_dir = np.array([-1.0, 0.0])
            return

        self.axis_origin = np.array(cusp['coord'], dtype=np.float64)
        axis_end = np.array(apex['coord'], dtype=np.float64)
        v = axis_end - self.axis_origin
        vlen = np.linalg.norm(v)
        self.axis_dir = v / vlen if vlen > 0 else np.array([-1.0, 0.0])

    def _reflect_point(self, pt):
        """Reflect a point across the symmetry axis."""
        p = np.asarray(pt, dtype=np.float64)
        d = p - self.axis_origin
        proj = np.dot(d, self.axis_dir) * self.axis_dir
        perp = d - proj
        return self.axis_origin + proj - perp

    def _side_of_axis(self, pt):
        """Return sign: >0 = left, <0 = right, ~0 = on axis."""
        p = np.asarray(pt, dtype=np.float64)
        d = p - self.axis_origin
        return d[0] * self.axis_dir[1] - d[1] * self.axis_dir[0]

    def split_lobes(self, route_pts):
        """Split route into left and right lobe points."""
        pts = np.asarray(route_pts, dtype=np.float64)
        sides = np.array([self._side_of_axis(p) for p in pts])
        return pts[sides > 0], pts[sides < 0]

    def mirror_error(self, route_pts):
        """Mirror Error: average distance between left-lobe points and
        their reflections matched to right-lobe points.

        Returns distance in metres.
        """
        pts = np.asarray(route_pts, dtype=np.float64)
        if len(pts) < 6:
            return 500.0

        left, right = self.split_lobes(pts)
        if len(left) < 3 or len(right) < 3:
            return 500.0

        # Reflect left lobe across axis → match to right
        reflected_left = np.array([self._reflect_point(p) for p in left])
        right_tree = cKDTree(right)
        dists_lr, _ = right_tree.query(reflected_left)

        # Reflect right lobe → match to left
        reflected_right = np.array([self._reflect_point(p) for p in right])
        left_tree = cKDTree(left)
        dists_rl, _ = left_tree.query(reflected_right)

        # Convert from degrees to metres
        mean_deg = (dists_lr.mean() + dists_rl.mean()) / 2.0
        return float(mean_deg * 111_000.0)

    def trunk_penalty(self, route_pts):
        """Detect out-and-back spikes (trunks) and penalise.

        A trunk is a sequence of edges where the route departs and
        returns to approximately the same point, creating a spike
        with no mirrored counterpart.

        Detection: scan for window of 5 segments where cumulative
        bearing change > 150deg and total length < 300m with
        start/end within 80m.

        Returns penalty in metres (0 = no trunks).
        """
        pts = np.asarray(route_pts, dtype=np.float64)
        n = len(pts)
        if n < 5:
            return 0.0

        penalty = 0.0
        d = pts[1:] - pts[:-1]
        bearings = np.arctan2(d[:, 1], d[:, 0])
        seg_lengths = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) * 111_000.0

        window = 5
        for i in range(len(bearings) - window + 1):
            cum_seg_len = seg_lengths[i:i + window].sum()
            if cum_seg_len > 300.0:
                continue

            total_change = 0.0
            for j in range(i, i + window - 1):
                db = bearings[j + 1] - bearings[j]
                db = ((db + math.pi) % (2 * math.pi)) - math.pi
                total_change += abs(db)

            if total_change > math.radians(150):
                start_end_dist = np.linalg.norm(pts[i] - pts[i + window]) * 111_000.0
                if start_end_dist < 80.0:
                    penalty += cum_seg_len * 10.0

        return float(penalty)

    def score(self, route_pts):
        """Full symmetry-based score.

        Components (lower = better):
          Mirror error          40%  — bilateral symmetry fidelity
          Trunk penalty         25%  — out-and-back spike cost
          Turning-angle Frechet 20%  — shape grammar fidelity
          Spatial Frechet       15%  — geometric grounding

        Returns float in ~metre-like units.
        """
        pts = np.asarray(route_pts, dtype=np.float64)
        if len(pts) < 4:
            return 1e9

        m_err = self.mirror_error(route_pts)
        m_score = min(m_err, 500.0)

        t_pen = self.trunk_penalty(route_pts)
        t_score = min(t_pen, 5000.0)

        ta_fd = _turning_angle_frechet(route_pts, self.ideal_pts.tolist())
        ta_score = min(ta_fd / 180.0 * 300.0, 300.0)

        sp_fd = frechet_score(route_pts, self.ideal_pts.tolist())
        sp_score = min(sp_fd, 300.0)

        composite = (m_score * 0.40 +
                     t_score * 0.25 +
                     ta_score * 0.20 +
                     sp_score * 0.15)

        log(f"[sym-scorer] mirror={m_score:.1f}m trunk={t_score:.1f} "
            f"ta_fd={ta_score:.1f} spatial={sp_score:.1f} -> {composite:.1f}")

        return composite


def _turning_angle_frechet(route_pts, ideal_pts, n_sample=60):
    """Discrete Frechet on turning-angle chains (ignores GPS coords)."""
    r_chain = _turning_angle_chain(route_pts, n_sample)
    i_chain = _turning_angle_chain(ideal_pts, n_sample)
    if len(r_chain) < 3 or len(i_chain) < 3:
        return 180.0
    return discrete_frechet_fast(r_chain.reshape(-1, 1), i_chain.reshape(-1, 1))


def abstract_score(route, ideal_pts):
    """Module-level scoring function (called from engine.py).

    Creates a SymmetryScorer and delegates.
    """
    if not route or len(route) < 4:
        return 1e9
    landmarks = extract_landmarks(ideal_pts)
    scorer = SymmetryScorer(landmarks, ideal_pts)
    return scorer.score(route)


# ═══════════════════════════════════════════════════════════════════════════
#  PEDESTRIAN HIGHWAY MULTIPLIERS
# ═══════════════════════════════════════════════════════════════════════════

HIGHWAY_MULTIPLIERS = {
    'footway': 0.5, 'path': 0.5, 'pedestrian': 0.5,
    'track': 0.7, 'cycleway': 0.7, 'steps': 0.8,
    'living_street': 1.0, 'residential': 1.0,
    'unclassified': 1.2, 'service': 1.2,
    'tertiary': 1.5, 'tertiary_link': 1.5,
    'secondary': 2.0, 'secondary_link': 2.0,
    'primary': 3.0, 'primary_link': 3.0,
    'trunk': 5.0, 'trunk_link': 5.0,
    'motorway': 100.0, 'motorway_link': 100.0,
}


def _get_hw_mult(data):
    hw = data.get('highway', 'residential')
    if isinstance(hw, list):
        hw = hw[0]
    return HIGHWAY_MULTIPLIERS.get(hw, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _build_kdtree(G):
    """Build KDTree from graph nodes. Returns (tree, coords, node_ids) or None."""
    if G is None:
        return None
    nodes = list(G.nodes(data=True))
    if not nodes:
        return None
    coords = np.array([[d['y'], d['x']] for _, d in nodes], dtype=np.float64)
    node_ids = [n for n, _ in nodes]
    return cKDTree(coords), coords, node_ids


def _find_landmark_node(kd_data, coord):
    """Snap a landmark coordinate to the nearest graph node."""
    if kd_data is None or coord is None:
        return None
    tree, coords, nids = kd_data
    _, idx = tree.query(coord)
    return nids[idx]


# ═══════════════════════════════════════════════════════════════════════════
#  SYMMETRY-AWARE EDGE WEIGHT PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _precompute_symmetry_weights(G_sub, ideal_line, tangent_field, landmarks):
    """Symmetry-First cost:  W = L * H_w * (1 + beta*S_err + gamma*theta_dev)

    Where:
      L      = edge length
      H_w    = highway multiplier
      beta   = 0.005 per-metre symmetry penalty
      S_err  = mirror error (distance from edge midpoint reflection to
               nearest ideal-line point, in metres)
      gamma  = 1.5  heading deviation weight
      theta  = heading error in [0,1]

    Also applies:
      - Apex-aware turn handling: near apex (30m) U-turns are cheap;
        in lobes (>100m from cusp/apex) near-reversals get +500m
      - Landmark bonus: 40% cost reduction within 60m of cusp/apex
    """
    il = np.asarray(ideal_line, dtype=np.float64)
    tf = np.asarray(tangent_field, dtype=np.float64)
    n_seg = len(il) - 1

    edges = list(G_sub.edges(data=True))
    n_edges = len(edges)
    if n_edges == 0:
        return G_sub

    BETA = 0.005
    GAMMA = 1.5

    u_lats = np.empty(n_edges, dtype=np.float64)
    u_lngs = np.empty(n_edges, dtype=np.float64)
    v_lats = np.empty(n_edges, dtype=np.float64)
    v_lngs = np.empty(n_edges, dtype=np.float64)
    lengths = np.empty(n_edges, dtype=np.float64)
    hw_mults = np.empty(n_edges, dtype=np.float64)

    for k, (u, v, data) in enumerate(edges):
        u_lats[k] = G_sub.nodes[u]['y']
        u_lngs[k] = G_sub.nodes[u]['x']
        v_lats[k] = G_sub.nodes[v]['y']
        v_lngs[k] = G_sub.nodes[v]['x']
        lengths[k] = data.get('length', 50.0)
        hw_mults[k] = _get_hw_mult(data)

    mid_lats = (u_lats + v_lats) * 0.5
    mid_lngs = (u_lngs + v_lngs) * 0.5

    # -- Heading error --
    edge_bearings = np.degrees(np.arctan2(v_lngs - u_lngs, v_lats - u_lats)) % 360.0

    if n_seg > 0:
        seg_mids_lat = (il[:-1, 0] + il[1:, 0]) * 0.5
        seg_mids_lng = (il[:-1, 1] + il[1:, 1]) * 0.5
        d_to_segs = haversine_matrix(mid_lats, mid_lngs, seg_mids_lat, seg_mids_lng)
        nearest_seg = d_to_segs.argmin(axis=1)
        local_tangents = (tf[nearest_seg] + tf[np.minimum(nearest_seg + 1, len(tf) - 1)]) * 0.5
    else:
        local_tangents = np.zeros(n_edges)

    dev = np.abs(edge_bearings - local_tangents) % 360.0
    dev = np.where(dev > 180.0, 360.0 - dev, dev)
    dev = np.where(dev > 90.0, 180.0 - dev, dev)
    heading_error = dev / 90.0  # [0,1]

    # -- Symmetry error (mirror distance) --
    cusp = landmarks.get('cusp')
    apex = landmarks.get('apex')

    if cusp is not None and apex is not None:
        ax_origin = np.array(cusp['coord'], dtype=np.float64)
        ax_end = np.array(apex['coord'], dtype=np.float64)
        ax_vec = ax_end - ax_origin
        ax_len = np.linalg.norm(ax_vec)
        if ax_len > 0:
            ax_dir = ax_vec / ax_len
        else:
            ax_dir = np.array([-1.0, 0.0])

        # Reflect all midpoints across axis
        mid_pts = np.column_stack([mid_lats, mid_lngs])
        d = mid_pts - ax_origin
        proj = np.outer(d @ ax_dir, ax_dir)
        perp = d - proj
        reflected = ax_origin + proj - perp

        # Distance from each reflected point to nearest ideal-line point
        il_tree = cKDTree(il)
        refl_dists_deg, _ = il_tree.query(reflected)
        sym_error = refl_dists_deg * 111_000.0  # metres
    else:
        sym_error = np.zeros(n_edges)

    # -- Core cost --
    weights = lengths * hw_mults * (1.0 + BETA * sym_error + GAMMA * heading_error)

    # -- Apex-aware turn penalty --
    if cusp is not None and apex is not None:
        cusp_lat, cusp_lng = cusp['coord']
        apex_lat, apex_lng = apex['coord']

        d_to_cusp = haversine_vector(mid_lats, mid_lngs,
                                     np.full_like(mid_lats, cusp_lat),
                                     np.full_like(mid_lngs, cusp_lng))
        d_to_apex = haversine_vector(mid_lats, mid_lngs,
                                     np.full_like(mid_lats, apex_lat),
                                     np.full_like(mid_lngs, apex_lng))

        near_apex = d_to_apex < 30.0
        in_lobes = (d_to_apex > 100.0) & (d_to_cusp > 100.0)

        # Near apex: halve heading penalty (allow the 180deg turn)
        weights[near_apex] *= np.where(heading_error[near_apex] > 0.7,
                                        0.6, 1.0)

        # In lobes: heavy penalty for near-reversals (heading > 70deg)
        lobe_reversal = in_lobes & (heading_error > 0.78)
        weights[lobe_reversal] += 500.0

        # Landmark bonus: 40% cost reduction within 60m of cusp/apex
        near_landmark = (d_to_cusp < 60.0) | (d_to_apex < 60.0)
        weights[near_landmark] *= 0.6

    # -- Assign to graph edges --
    for k, (u, v, data) in enumerate(edges):
        keys = list(G_sub[u][v].keys())
        G_sub[u][v][keys[0]]['absw'] = float(weights[k])

    log(f"[sym-weights] {n_edges} edges, heading_err_mean={heading_error.mean():.2f}, "
        f"sym_err_mean={sym_error.mean():.0f}m")

    return G_sub


# ═══════════════════════════════════════════════════════════════════════════
#  TANGENT PRUNING
# ═══════════════════════════════════════════════════════════════════════════

def _prune_perpendicular_edges(G_sub, ideal_line, tangent_field,
                                max_deviation_deg=35.0):
    """Remove edges deviating >35deg from local tangent within 150m of ideal.

    Preserves connectivity (only prunes if resulting graph stays connected).
    """
    import networkx as nx

    il = np.asarray(ideal_line, dtype=np.float64)
    tf = np.asarray(tangent_field, dtype=np.float64)
    n_seg = len(il) - 1
    if n_seg < 1:
        return G_sub

    edges = list(G_sub.edges(data=True, keys=True))
    n_edges = len(edges)
    if n_edges < 20:
        return G_sub

    u_lats = np.array([G_sub.nodes[e[0]]['y'] for e in edges])
    u_lngs = np.array([G_sub.nodes[e[0]]['x'] for e in edges])
    v_lats = np.array([G_sub.nodes[e[1]]['y'] for e in edges])
    v_lngs = np.array([G_sub.nodes[e[1]]['x'] for e in edges])

    edge_bearings = np.degrees(np.arctan2(v_lngs - u_lngs, v_lats - u_lats)) % 360.0
    mid_lats = (u_lats + v_lats) * 0.5
    mid_lngs = (u_lngs + v_lngs) * 0.5

    midpoints = np.column_stack([mid_lats, mid_lngs])
    prox_dists = point_to_segments_vectorized(midpoints, il[:-1], il[1:])

    seg_mids_lat = (il[:-1, 0] + il[1:, 0]) * 0.5
    seg_mids_lng = (il[:-1, 1] + il[1:, 1]) * 0.5
    d_to_segs = haversine_matrix(mid_lats, mid_lngs, seg_mids_lat, seg_mids_lng)
    nearest_seg = d_to_segs.argmin(axis=1)
    local_tangents = (tf[nearest_seg] + tf[np.minimum(nearest_seg + 1, len(tf) - 1)]) * 0.5

    dev = np.abs(edge_bearings - local_tangents) % 360.0
    dev = np.where(dev > 180.0, 360.0 - dev, dev)
    dev = np.where(dev > 90.0, 180.0 - dev, dev)

    to_prune = (dev > max_deviation_deg) & (prox_dists < 150.0)
    edges_to_remove = [(e[0], e[1], e[2]) for e, prune in zip(edges, to_prune) if prune]
    if not edges_to_remove:
        return G_sub

    G_test = G_sub.copy()
    G_test.remove_edges_from(edges_to_remove)
    if G_test.number_of_edges() < 10:
        return G_sub
    if not nx.is_weakly_connected(G_test):
        largest_cc = max(nx.weakly_connected_components(G_test), key=len)
        if len(largest_cc) < G_sub.number_of_nodes() * 0.6:
            return G_sub
        G_test = G_test.subgraph(largest_cc).copy()

    log(f"[sym-prune] Removed {len(edges_to_remove)}/{n_edges} perpendicular edges")
    return G_test


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

def _route_half(G, start, end, via_node, weight_attr):
    """Route start -> via_node -> end (or start -> end if no via)."""
    import networkx as nx

    if via_node is not None and via_node != start and via_node != end:
        try:
            path_a = nx.shortest_path(G, start, via_node, weight=weight_attr)
            path_b = nx.shortest_path(G, via_node, end, weight=weight_attr)
            return path_a + path_b[1:]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    try:
        return nx.shortest_path(G, start, end, weight=weight_attr)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _mirror_search(G_sub, ideal_line, landmarks, kd_data, weight_attr='absw'):
    """Mirror-Search: route left half, penalise those edges, route right half.

    Left-path edges are penalised 50x for right-half routing,
    forcing the router to find a symmetric counterpart.
    """
    import networkx as nx

    cusp = landmarks.get('cusp')
    apex = landmarks.get('apex')
    lobes = landmarks.get('lobes', [])

    if cusp is None or apex is None or kd_data is None:
        return None

    cusp_node = _find_landmark_node(kd_data, cusp['coord'])
    apex_node = _find_landmark_node(kd_data, apex['coord'])

    if cusp_node is None or apex_node is None or cusp_node == apex_node:
        return None

    left_lobe_node = None
    right_lobe_node = None
    for lobe in lobes:
        lobe_node = _find_landmark_node(kd_data, lobe['arc_center'])
        if lobe['side'] == 'left':
            left_lobe_node = lobe_node
        else:
            right_lobe_node = lobe_node

    # Left half: cusp -> [left_lobe] -> apex
    left_path = _route_half(G_sub, cusp_node, apex_node, left_lobe_node, weight_attr)
    if left_path is None:
        return None

    # Penalise left-path edges 50x for right-half routing
    G_mod = G_sub.copy()
    for i in range(len(left_path) - 1):
        u, v = left_path[i], left_path[i + 1]
        for key in list(G_mod[u][v].keys()) if G_mod.has_edge(u, v) else []:
            G_mod[u][v][key][weight_attr] = G_mod[u][v][key].get(weight_attr, 50) * 50.0
        for key in list(G_mod[v][u].keys()) if G_mod.has_edge(v, u) else []:
            G_mod[v][u][key][weight_attr] = G_mod[v][u][key].get(weight_attr, 50) * 50.0

    # Right half: apex -> [right_lobe] -> cusp
    right_path = _route_half(G_mod, apex_node, cusp_node, right_lobe_node, weight_attr)
    if right_path is None:
        return None

    # Combine into closed loop
    cycle_nodes = left_path + right_path[1:]

    route = []
    for nid in cycle_nodes:
        pt = [G_sub.nodes[nid]['y'], G_sub.nodes[nid]['x']]
        if not route or route[-1] != pt:
            route.append(pt)

    if route and route[0] != route[-1]:
        route.append(route[0])

    return route if len(route) >= 6 else None


def _cycle_search(G_sub, ideal_line, landmarks, kd_data, weight_attr='absw'):
    """Cycle-based search using minimum_cycle_basis.

    Finds cycles in the corridor graph and selects the one with
    best symmetry vs the ideal shape.
    """
    import networkx as nx

    if kd_data is None:
        return None

    try:
        # minimum_cycle_basis requires a simple undirected graph
        G_und = nx.Graph(G_sub.to_undirected())
        cycles = nx.minimum_cycle_basis(G_und, weight=weight_attr)
    except Exception as e:
        log(f"[cycle-search] minimum_cycle_basis failed: {e}")
        return None

    if not cycles:
        return None

    scorer = SymmetryScorer(landmarks, ideal_line)

    # Only evaluate cycles with at least 10 nodes
    large_cycles = [c for c in cycles if len(c) >= 10]
    if not large_cycles:
        large_cycles = cycles[:5]

    best_score = 1e9
    best_route = None

    for cycle_nodes in large_cycles[:20]:
        route = []
        for nid in cycle_nodes:
            if nid in G_sub.nodes:
                pt = [G_sub.nodes[nid]['y'], G_sub.nodes[nid]['x']]
                if not route or route[-1] != pt:
                    route.append(pt)

        if len(route) < 6:
            continue

        if route[0] != route[-1]:
            route.append(route[0])

        cscore = scorer.score(route)
        if cscore < best_score:
            best_score = cscore
            best_route = route

    if best_route:
        log(f"[cycle-search] Best cycle: {len(best_route)} pts, score={best_score:.1f}")

    return best_route


def _segment_route(G_sub, ideal_line, landmarks, kd_data, weight_attr='absw'):
    """Fallback: waypoint-based routing with symmetry weights."""
    import networkx as nx

    if kd_data is None:
        return None

    tree, coords, nids = kd_data

    n_wps = min(25, max(8, len(ideal_line) // 2))
    wps = sample_polyline(ideal_line, n_wps)
    wps_a = np.asarray(wps, dtype=np.float64)

    _, indices = tree.query(wps_a)
    node_ids = [nids[i] for i in indices]

    deduped = [node_ids[0]]
    for i in range(1, len(node_ids)):
        if node_ids[i] != deduped[-1]:
            deduped.append(node_ids[i])

    if len(deduped) < 2:
        return None

    full = []
    for i in range(len(deduped) - 1):
        try:
            path = nx.shortest_path(G_sub, deduped[i], deduped[i + 1], weight=weight_attr)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for nid in path:
            pt = [G_sub.nodes[nid]['y'], G_sub.nodes[nid]['x']]
            if not full or full[-1] != pt:
                full.append(pt)

    if full and len(full) >= 3:
        if haversine(full[0][0], full[0][1], full[-1][0], full[-1][1]) < 50:
            try:
                path_close = nx.shortest_path(G_sub, deduped[-1], deduped[0], weight=weight_attr)
                for nid in path_close[1:]:
                    pt = [G_sub.nodes[nid]['y'], G_sub.nodes[nid]['x']]
                    if not full or full[-1] != pt:
                        full.append(pt)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    return full if len(full) >= 2 else None


# ═══════════════════════════════════════════════════════════════════════════
#  ABSTRACT ROUTER — MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class AbstractRouter:
    """Symmetry-First router: bilateral symmetry as a hard constraint.

    Routing priority:
      1. Mirror-Search (symmetric closed loops)
      2. Cycle-Search (nx.minimum_cycle_basis for symmetric loops)
      3. Segment routing (waypoint fallback)

    Best of all strategies is selected by SymmetryScorer.

    Usage:
        router = AbstractRouter(G, ideal_line)
        route = router.route()
        score = router.score(route)
    """

    WIDE_RADIUS_M = 500

    def __init__(self, G, ideal_line, config=None):
        self.G = G
        self.ideal_line = ideal_line
        self.il = np.asarray(ideal_line, dtype=np.float64)
        self.config = config or {}

        # SPATIAL: extract landmarks
        self.landmarks = extract_landmarks(ideal_line)
        self.tangent_field = compute_tangent_field(ideal_line)
        self.is_closed = self._check_closed()

        # Build wide corridor (500m)
        self.G_sub, self.kd_sub = self._build_wide_corridor()

        # Tangent pruning
        if self.G_sub is not None and self.G_sub.number_of_edges() > 0:
            self.G_sub = _prune_perpendicular_edges(
                self.G_sub, ideal_line, self.tangent_field
            )
            self.kd_sub = _build_kdtree(self.G_sub)

        # GRAPH: precompute symmetry-aware weights
        if self.G_sub is not None and self.G_sub.number_of_edges() > 0:
            _precompute_symmetry_weights(
                self.G_sub, ideal_line, self.tangent_field, self.landmarks
            )

        # VISION: symmetry scorer
        self.scorer = SymmetryScorer(self.landmarks, ideal_line)

        log(f"[AbstractRouter] landmarks={bool(self.landmarks.get('cusp'))}, "
            f"closed={self.is_closed}, "
            f"corridor={self.G_sub.number_of_nodes() if self.G_sub else 0} nodes")

    def _check_closed(self):
        if len(self.il) < 3:
            return False
        return haversine(self.il[0, 0], self.il[0, 1],
                         self.il[-1, 0], self.il[-1, 1]) < 50.0

    def _build_wide_corridor(self):
        """Build 500m corridor for abstract matching."""
        try:
            import osmnx as ox
            import networkx as nx
        except ImportError:
            return self.G, _build_kdtree(self.G)

        if self.G is None:
            return self.G, None

        il = self.il
        radius_m = self.config.get('search_radius_m', self.WIDE_RADIUS_M)
        r_deg = radius_m / 111_000.0

        try:
            from shapely.geometry import Point
            from shapely.ops import unary_union

            circles = [Point(il[i, 1], il[i, 0]).buffer(r_deg) for i in range(len(il))]
            corridor = unary_union(circles)

            nodes_gdf = ox.graph_to_gdfs(self.G, nodes=True, edges=False)
            candidates = nodes_gdf.sindex.query(corridor, predicate='intersects')
            corridor_nodes = set(nodes_gdf.index[candidates])

            if len(corridor_nodes) < 20:
                for widen in range(100, 600, 100):
                    corridor = corridor.buffer(widen / 111_000.0)
                    candidates = nodes_gdf.sindex.query(corridor, predicate='intersects')
                    corridor_nodes = set(nodes_gdf.index[candidates])
                    if len(corridor_nodes) >= 20:
                        break

            if len(corridor_nodes) < 10:
                return self.G, _build_kdtree(self.G)

            G_sub = self.G.subgraph(corridor_nodes).copy()
            if not nx.is_weakly_connected(G_sub):
                largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
                G_sub = G_sub.subgraph(largest_cc).copy()

            log(f"[abstract-corridor] {self.G.number_of_nodes()} -> "
                f"{G_sub.number_of_nodes()} nodes (r={radius_m}m)")
            return G_sub, _build_kdtree(G_sub)

        except Exception as e:
            log(f"[abstract-corridor] Shapely failed ({e}), KDTree fallback")
            return self._corridor_kdtree_fallback()

    def _corridor_kdtree_fallback(self):
        import networkx as nx
        kd = _build_kdtree(self.G)
        if kd is None:
            return self.G, None
        tree, coords, node_ids = kd

        r_deg = self.WIDE_RADIUS_M / 111_000.0
        corridor_indices = set()
        for i in range(len(self.il)):
            indices = tree.query_ball_point([self.il[i, 0], self.il[i, 1]], r_deg)
            corridor_indices.update(indices)

        corridor_nodes = {node_ids[i] for i in corridor_indices}
        if len(corridor_nodes) < 10:
            return self.G, kd

        G_sub = self.G.subgraph(corridor_nodes).copy()
        if not nx.is_weakly_connected(G_sub):
            largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
            G_sub = G_sub.subgraph(largest_cc).copy()

        return G_sub, _build_kdtree(G_sub)

    def route(self):
        """Route using symmetry-first strategy.

        Tries all strategies and picks the best by SymmetryScorer.
        """
        if self.G_sub is None:
            return None

        candidates = []

        # Strategy 1: Mirror-search (best for symmetric closed shapes)
        if self.is_closed and self.landmarks.get('cusp') and self.landmarks.get('apex'):
            mirror_route = _mirror_search(
                self.G_sub, self.ideal_line, self.landmarks,
                self.kd_sub, weight_attr='absw'
            )
            if mirror_route and len(mirror_route) >= 6:
                candidates.append(('mirror', mirror_route))

        # Strategy 2: Cycle-search (nx.minimum_cycle_basis)
        if self.is_closed:
            cycle_route = _cycle_search(
                self.G_sub, self.ideal_line, self.landmarks,
                self.kd_sub, weight_attr='absw'
            )
            if cycle_route and len(cycle_route) >= 6:
                candidates.append(('cycle', cycle_route))

        # Strategy 3: Segment routing fallback
        seg_route = _segment_route(
            self.G_sub, self.ideal_line, self.landmarks,
            self.kd_sub, weight_attr='absw'
        )
        if seg_route and len(seg_route) >= 2:
            candidates.append(('segment', seg_route))

        if not candidates:
            log("[AbstractRouter] All strategies failed")
            return None

        # Pick best by SymmetryScorer
        best_score = 1e9
        best_route = None
        best_strategy = ''
        for name, route in candidates:
            s = self.scorer.score(route)
            if s < best_score:
                best_score = s
                best_route = route
                best_strategy = name

        log(f"[AbstractRouter] Best: {best_strategy} ({len(best_route)} pts, "
            f"score={best_score:.1f}) from {len(candidates)} candidates")
        return best_route

    def score(self, route):
        """Score a route using SymmetryScorer."""
        if not route or len(route) < 4:
            return 1e9
        return self.scorer.score(route)
