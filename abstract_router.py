"""
abstract_router.py — v8.1-Abstract "Signature Matching" Router
================================================================
Treats the template as a Topological Grammar rather than a set of
coordinates.  Routes are scored on geometric invariants (turning-angle
chains, symmetry, curvature consistency) instead of Euclidean proximity.

AGENT 'SPATIAL'  — extract_landmarks()    → Cusp, Apex, Lobe-Arc zones
AGENT 'GRAPH'    — AbstractRouter class   → Signature-weighted cost + mirror search
AGENT 'VISION'   — abstract_score()       → Symmetry + curvature + turning-angle Fréchet
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
from scoring_v8 import discrete_frechet_fast


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
    """Compute the normalized turning-angle chain for a polyline.

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
    """
    pts = np.asarray(ideal_line, dtype=np.float64)
    n = len(pts)
    if n < 6:
        return {'cusp': None, 'apex': None, 'lobes': []}

    curvature = _curvature_at_points(pts)

    # --- Apex: sharpest single-point turn (bottom of heart) ---
    abs_curv = np.abs(curvature)
    apex_idx = int(np.argmax(abs_curv))

    # --- Cusp: second-sharpest turn, must be far from apex ---
    min_sep = max(n // 4, 3)
    abs_curv_masked = abs_curv.copy()
    lo = max(0, apex_idx - min_sep)
    hi = min(n, apex_idx + min_sep + 1)
    abs_curv_masked[lo:hi] = 0
    cusp_idx = int(np.argmax(abs_curv_masked))

    # Ensure cusp is "above" apex (for heart: cusp at top, apex at bottom)
    # Use vertical component (latitude)
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

    # --- Lobe Arcs: high-curvature zones on left and right ---
    # Walk from cusp → apex (left lobe) and apex → cusp (right lobe)
    # Use parametric ordering
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
        # Find the peak-curvature zone within this lobe
        peak_pos = int(np.argmax(curv_vals))
        # Expand around peak while curvature is > 50% of peak
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
        f"defl={curvature[cusp_idx]:.1f}°, apex=idx{apex_idx} "
        f"defl={curvature[apex_idx]:.1f}°, lobes={len(lobes)}")

    return {'cusp': cusp_info, 'apex': apex_info, 'lobes': lobes}


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT 'VISION' — ABSTRACT FIDELITY SCORER
# ═══════════════════════════════════════════════════════════════════════════

def _symmetry_index(route_pts):
    """Reflective symmetry across the route's vertical axis.

    Returns value in [0, 1] where 1 = perfectly symmetric.
    """
    pts = np.asarray(route_pts, dtype=np.float64)
    if len(pts) < 4:
        return 0.0

    # Normalize to unit box
    norm = normalize_to_unit_box(pts)

    # Split into top-half and bottom-half (or left/right)
    # For a heart: vertical axis of symmetry → reflect across x-center
    cx = (norm[:, 1].min() + norm[:, 1].max()) / 2.0

    # Resample to even spacing
    n_s = min(60, len(norm))
    indices = np.linspace(0, len(norm) - 1, n_s).astype(int)
    sampled = norm[indices]

    # Mirror each point across center-x
    mirrored = sampled.copy()
    mirrored[:, 1] = 2 * cx - mirrored[:, 1]

    # For each mirrored point, find nearest original point
    from scipy.spatial import cKDTree as _KD
    tree = _KD(sampled)
    dists, _ = tree.query(mirrored)

    # Symmetry = 1 - mean_distance (clamped)
    mean_d = float(dists.mean())
    return max(0.0, 1.0 - mean_d * 5.0)


def _curvature_consistency(route_pts, ideal_pts):
    """Check if the route lobes maintain consistent arc radii vs the template.

    Returns penalty in [0, 100] where 0 = perfect match.
    """
    r_curv = _curvature_at_points(sample_polyline(route_pts, 40))
    i_curv = _curvature_at_points(sample_polyline(ideal_pts, 40))

    if len(r_curv) < 5 or len(i_curv) < 5:
        return 50.0

    # Compare curvature distributions via std-ratio
    r_std = float(np.std(np.abs(r_curv[1:-1]))) + 1e-6
    i_std = float(np.std(np.abs(i_curv[1:-1]))) + 1e-6

    # Also compare mean absolute curvature
    r_mean = float(np.abs(r_curv[1:-1]).mean())
    i_mean = float(np.abs(i_curv[1:-1]).mean())

    ratio_std = max(r_std / i_std, i_std / r_std)
    ratio_mean = max(r_mean / (i_mean + 1e-6), i_mean / (r_mean + 1e-6))

    # Penalty: 0 if ratio=1, grows linearly
    return min(100.0, (ratio_std - 1.0) * 20.0 + (ratio_mean - 1.0) * 15.0)


def _turning_angle_frechet(route_pts, ideal_pts, n_sample=60):
    """Discrete Fréchet on the turning-angle chain (ignores GPS coords).

    Both chains are normalized to [-180, 180] degree sequences.
    Returns distance in degree-space.
    """
    r_chain = _turning_angle_chain(route_pts, n_sample)
    i_chain = _turning_angle_chain(ideal_pts, n_sample)

    if len(r_chain) < 3 or len(i_chain) < 3:
        return 180.0

    # Reshape as 1D "polylines" for Fréchet: (N, 1) arrays
    r_1d = r_chain.reshape(-1, 1)
    i_1d = i_chain.reshape(-1, 1)

    return discrete_frechet_fast(r_1d, i_1d)


def abstract_score(route, ideal_pts):
    """Abstract fidelity score — topology-aware.

    Components (lower = better):
      Turning-angle Fréchet   45%  — shape "grammar" fidelity
      Symmetry index          25%  — reflective symmetry (inverted: 1-sym)
      Curvature consistency   15%  — arc radius match
      Spatial Fréchet         15%  — fallback geometric fidelity

    Returns: float (lower = better, in abstract units scaled ≈ metres)
    """
    if not route or len(route) < 2:
        return 1e9

    ta_fd = _turning_angle_frechet(route, ideal_pts)
    sym = _symmetry_index(route)
    curv = _curvature_consistency(route, ideal_pts)

    # Spatial Fréchet for grounding (same as scoring_v8.frechet_score)
    from scoring_v8 import frechet_score
    spatial_fd = frechet_score(route, ideal_pts)

    # Normalize turning-angle Fréchet to metre-like scale
    # Max meaningful TA-Fréchet ≈ 180°; map to 0-300m range
    ta_score = min(ta_fd / 180.0 * 300.0, 300.0)

    # Symmetry: invert (1=perfect → 0 penalty, 0=bad → 150 penalty)
    sym_score = (1.0 - sym) * 150.0

    composite = (ta_score * 0.45 +
                 sym_score * 0.25 +
                 curv * 0.15 +
                 spatial_fd * 0.15)

    return composite


# ═══════════════════════════════════════════════════════════════════════════
#  PEDESTRIAN HIGHWAY MULTIPLIERS  (shared with core_router)
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
#  AGENT 'GRAPH' — SIGNATURE MATCHING ROUTER
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


def _precompute_signature_weights(G_sub, ideal_line, tangent_field, landmarks):
    """Signature-weighted cost: Length × (1 + HeadingError).

    Assigns 'absw' (abstract weight) to each edge.
    Also applies landmark bonuses: edges near cusp/apex get reduced weight.
    """
    il = np.asarray(ideal_line, dtype=np.float64)
    tf = np.asarray(tangent_field, dtype=np.float64)
    n_seg = len(il) - 1

    edges = list(G_sub.edges(data=True))
    n_edges = len(edges)
    if n_edges == 0:
        return G_sub

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

    # Edge bearings
    edge_bearings = np.degrees(np.arctan2(v_lngs - u_lngs, v_lats - u_lats)) % 360.0

    # Nearest ideal segment tangent
    if n_seg > 0:
        seg_mids_lat = (il[:-1, 0] + il[1:, 0]) * 0.5
        seg_mids_lng = (il[:-1, 1] + il[1:, 1]) * 0.5
        d_to_segs = haversine_matrix(mid_lats, mid_lngs, seg_mids_lat, seg_mids_lng)
        nearest_seg = d_to_segs.argmin(axis=1)
        local_tangents = (tf[nearest_seg] + tf[np.minimum(nearest_seg + 1, len(tf) - 1)]) * 0.5
    else:
        local_tangents = np.zeros(n_edges)

    # Heading error: [0, 1] range (0=aligned, 1=perpendicular)
    dev = np.abs(edge_bearings - local_tangents) % 360.0
    dev = np.where(dev > 180.0, 360.0 - dev, dev)
    dev = np.where(dev > 90.0, 180.0 - dev, dev)
    heading_error = dev / 90.0  # normalize to [0,1]

    # Signature cost: Length × HW × (1 + HeadingError)
    weights = lengths * hw_mults * (1.0 + heading_error)

    # Landmark bonus: reduce cost near cusp and apex for connectivity
    for landmark_key in ['cusp', 'apex']:
        lm = landmarks.get(landmark_key)
        if lm is None:
            continue
        lm_lat, lm_lng = lm['coord']
        d_to_lm = haversine_vector(mid_lats, mid_lngs,
                                   np.full_like(mid_lats, lm_lat),
                                   np.full_like(mid_lngs, lm_lng))
        # Within 80m of landmark: 50% cost reduction
        near_lm = d_to_lm < 80.0
        weights[near_lm] *= 0.5

    # Assign to graph edges
    for k, (u, v, data) in enumerate(edges):
        keys = list(G_sub[u][v].keys())
        G_sub[u][v][keys[0]]['absw'] = float(weights[k])

    log(f"[abstract-graph] Precomputed {n_edges} signature weights "
        f"(heading_err_mean={heading_error.mean():.2f})")

    return G_sub


def _find_landmark_node(kd_data, coord):
    """Snap a landmark coordinate to the nearest graph node."""
    if kd_data is None or coord is None:
        return None
    tree, coords, nids = kd_data
    _, idx = tree.query(coord)
    return nids[idx]


def _mirror_search(G_sub, ideal_line, landmarks, kd_data, weight_attr='absw'):
    """Mirror-Search: route left lobe, then find symmetric right lobe.

    Strategy:
    1. Route cusp → left-lobe-arc → apex (left half)
    2. Route cusp → right-lobe-arc → apex (right half)
    3. Combine into closed loop, preferring symmetric edge sequences
    """
    import networkx as nx

    il = np.asarray(ideal_line, dtype=np.float64)
    cusp = landmarks.get('cusp')
    apex = landmarks.get('apex')
    lobes = landmarks.get('lobes', [])

    if cusp is None or apex is None or kd_data is None:
        return None

    cusp_node = _find_landmark_node(kd_data, cusp['coord'])
    apex_node = _find_landmark_node(kd_data, apex['coord'])

    if cusp_node is None or apex_node is None or cusp_node == apex_node:
        return None

    # Find lobe waypoints (midpoints of each lobe arc)
    left_lobe_node = None
    right_lobe_node = None
    for lobe in lobes:
        lobe_node = _find_landmark_node(kd_data, lobe['arc_center'])
        if lobe['side'] == 'left':
            left_lobe_node = lobe_node
        else:
            right_lobe_node = lobe_node

    # Route left half: cusp → [left_lobe] → apex
    left_path = _route_half(G_sub, cusp_node, apex_node, left_lobe_node, weight_attr)
    if left_path is None:
        return None

    # Route right half: apex → [right_lobe] → cusp
    # Penalize edges used in left_path to encourage different route
    G_mod = G_sub.copy()
    for i in range(len(left_path) - 1):
        u, v = left_path[i], left_path[i + 1]
        for key in list(G_mod[u][v].keys()) if G_mod.has_edge(u, v) else []:
            G_mod[u][v][key][weight_attr] = G_mod[u][v][key].get(weight_attr, 50) * 50.0
        for key in list(G_mod[v][u].keys()) if G_mod.has_edge(v, u) else []:
            G_mod[v][u][key][weight_attr] = G_mod[v][u][key].get(weight_attr, 50) * 50.0

    right_path = _route_half(G_mod, apex_node, cusp_node, right_lobe_node, weight_attr)
    if right_path is None:
        return None

    # Combine: left_path + right_path[1:] = closed loop
    cycle_nodes = left_path + right_path[1:]

    route = []
    for nid in cycle_nodes:
        pt = [G_sub.nodes[nid]['y'], G_sub.nodes[nid]['x']]
        if not route or route[-1] != pt:
            route.append(pt)

    # Close the loop
    if route and route[0] != route[-1]:
        route.append(route[0])

    return route if len(route) >= 6 else None


def _route_half(G, start, end, via_node, weight_attr):
    """Route start → via_node → end, or start → end if via_node unavailable."""
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


def _segment_route(G_sub, ideal_line, landmarks, kd_data, weight_attr='absw'):
    """Fallback: waypoint-based routing with signature weights and zero
    U-turn penalty at cusp/apex nodes."""
    import networkx as nx

    if kd_data is None:
        return None

    tree, coords, nids = kd_data

    n_wps = min(25, max(8, len(ideal_line) // 2))
    wps = sample_polyline(ideal_line, n_wps)
    wps_a = np.asarray(wps, dtype=np.float64)

    _, indices = tree.query(wps_a)
    node_ids = [nids[i] for i in indices]

    # Deduplicate
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

    # Close loop if needed
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
    """Signature-matching router using topological grammar.

    Usage:
        router = AbstractRouter(G, ideal_line)
        route = router.route()
        score = router.score(route)
    """

    WIDE_RADIUS_M = 500  # intentionally wide search tube

    def __init__(self, G, ideal_line, config=None):
        self.G = G
        self.ideal_line = ideal_line
        self.il = np.asarray(ideal_line, dtype=np.float64)
        self.config = config or {}

        # SPATIAL: extract topological landmarks
        self.landmarks = extract_landmarks(ideal_line)
        self.tangent_field = compute_tangent_field(ideal_line)
        self.is_closed = self._check_closed()

        # Build wide corridor (500m — intentionally broad for abstract matching)
        self.G_sub, self.kd_sub = self._build_wide_corridor()

        # GRAPH: precompute signature weights
        if self.G_sub is not None and self.G_sub.number_of_edges() > 0:
            _precompute_signature_weights(
                self.G_sub, ideal_line, self.tangent_field, self.landmarks
            )

        log(f"[AbstractRouter] landmarks={bool(self.landmarks.get('cusp'))}, "
            f"closed={self.is_closed}, "
            f"corridor={self.G_sub.number_of_nodes() if self.G_sub else 0} nodes")

    def _check_closed(self):
        if len(self.il) < 3:
            return False
        return haversine(self.il[0, 0], self.il[0, 1],
                         self.il[-1, 0], self.il[-1, 1]) < 50.0

    def _build_wide_corridor(self):
        """Build a wide 500m corridor for abstract matching."""
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

            log(f"[abstract-corridor] {self.G.number_of_nodes()} → {G_sub.number_of_nodes()} nodes (r={radius_m}m)")
            return G_sub, _build_kdtree(G_sub)

        except Exception as e:
            log(f"[abstract-corridor] Shapely failed ({e}), using KDTree fallback")
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
        """Route using signature matching.

        Strategy 1: Mirror-Search (symmetric closed shapes)
        Strategy 2: Segment routing with signature weights
        """
        if self.G_sub is None:
            return None

        # Strategy 1: Mirror-search for closed shapes with landmarks
        if self.is_closed and self.landmarks.get('cusp') and self.landmarks.get('apex'):
            mirror_route = _mirror_search(
                self.G_sub, self.ideal_line, self.landmarks,
                self.kd_sub, weight_attr='absw'
            )
            if mirror_route and len(mirror_route) >= 6:
                log(f"[AbstractRouter] Mirror-search route: {len(mirror_route)} pts")
                return mirror_route

        # Strategy 2: Segment routing with signature weights
        seg_route = _segment_route(
            self.G_sub, self.ideal_line, self.landmarks,
            self.kd_sub, weight_attr='absw'
        )
        if seg_route:
            log(f"[AbstractRouter] Segment route: {len(seg_route)} pts")
            return seg_route

        return None

    def score(self, route):
        """Score using abstract fidelity (topology-aware)."""
        if not route or len(route) < 2:
            return 1e9
        return abstract_score(route, self.ideal_line)
