"""
v83_enhancements.py — Modular v8.3 Experiment Enhancements
============================================================
All v8.3 features are implemented as standalone functions that can be
toggled via a config dict. They augment (not replace) the v8.2 pipeline.

Features:
  1. Dynamic densification (road-curvature-aware)
  2. B-spline post-smoothing (snap-constrained)
  3. Multi-resolution routing (coarse→fine)
  4. Symmetry penalty for CMA-ES
  5. Forced cycle closure
  6. Overlap / edge-reuse penalty
  7. Hybrid v6 coarse+v8 fine search
  8. Configurable penalty factor scaling
"""

import math
import sys
import numpy as np
from scipy.spatial import cKDTree

from geometry import (
    haversine, haversine_vector, haversine_matrix,
    sample_polyline, adaptive_densify, shape_to_latlngs,
    point_to_segment_dist,
)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  1. DYNAMIC DENSIFICATION (road-curvature-aware)
# ═══════════════════════════════════════════════════════════════════════════

def dynamic_densify(waypoints, G, kdtree_data=None, min_spacing=10,
                    max_spacing=60, curvature_threshold=25):
    """Densify shape waypoints based on nearby road network curvature.

    At points where the road network has high turning angles (intersections,
    curves), insert more waypoints. On straight road stretches, keep sparse.

    Args:
        waypoints: list of [lat, lng]
        G: osmnx graph
        kdtree_data: (tree, coords, node_ids) tuple
        min_spacing: minimum spacing in metres at high-curvature points
        max_spacing: maximum spacing in metres on straights
        curvature_threshold: road bearing change (degrees) considered "curved"

    Returns: densified list of [lat, lng]
    """
    if not G or not waypoints or len(waypoints) < 2:
        return adaptive_densify(waypoints, base_spacing=35, curve_spacing=15)

    wps = np.asarray(waypoints, dtype=np.float64)
    n = len(wps)

    # Build KDTree if not provided
    if kdtree_data is None:
        nodes = list(G.nodes(data=True))
        coords = np.array([[nd['y'], nd['x']] for _, nd in nodes], dtype=np.float64)
        node_ids = [nid for nid, _ in nodes]
        tree = cKDTree(coords)
    else:
        tree, coords, node_ids = kdtree_data

    # For each waypoint, estimate local road curvature
    road_curvatures = np.zeros(n, dtype=np.float64)
    _, nearest_idx = tree.query(wps)

    for i in range(n):
        nid = node_ids[nearest_idx[i]]
        # Get edges around this node and compute bearing variance
        neighbors = list(G.successors(nid)) + list(G.predecessors(nid))
        if len(neighbors) < 2:
            road_curvatures[i] = 0
            continue
        bearings = []
        nlat, nlng = G.nodes[nid]['y'], G.nodes[nid]['x']
        for nb in neighbors:
            nb_lat, nb_lng = G.nodes[nb]['y'], G.nodes[nb]['x']
            b = math.degrees(math.atan2(nb_lng - nlng, nb_lat - nlat)) % 360
            bearings.append(b)
        # Curvature = range of bearings (higher = more intersection-like)
        bearings.sort()
        if len(bearings) >= 2:
            diffs = [bearings[j+1] - bearings[j] for j in range(len(bearings)-1)]
            diffs.append(360 - bearings[-1] + bearings[0])
            road_curvatures[i] = max(diffs) - min(diffs)
        else:
            road_curvatures[i] = 0

    # Now densify: spacing inversely proportional to road curvature
    result = []
    for i in range(n - 1):
        curv = (road_curvatures[i] + road_curvatures[min(i+1, n-1)]) / 2
        # Map curvature to spacing
        if curv > curvature_threshold:
            spacing = min_spacing
        else:
            t = curv / max(curvature_threshold, 1)
            spacing = max_spacing * (1 - t) + min_spacing * t

        dist = haversine(wps[i, 0], wps[i, 1], wps[i+1, 0], wps[i+1, 1])
        n_pts = max(1, int(dist / max(spacing, 1)))

        for j in range(n_pts):
            frac = j / n_pts
            lat = wps[i, 0] + frac * (wps[i+1, 0] - wps[i, 0])
            lng = wps[i, 1] + frac * (wps[i+1, 1] - wps[i, 1])
            result.append([lat, lng])

    result.append(list(wps[-1]))
    log(f"[v8.3-dynamic] Densified {n} → {len(result)} waypoints")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  2. B-SPLINE POST-SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

def spline_smooth_route(route, G, kdtree_data=None, k=3, densify_factor=2,
                        max_snap_dist_m=80):
    """Smooth a route using B-spline interpolation, then snap back to roads.

    Args:
        route: list of [lat, lng]
        G: osmnx graph
        kdtree_data: (tree, coords, node_ids) tuple
        k: spline degree (3=cubic, 4=quartic, 5=quintic)
        densify_factor: how many times more points to generate
        max_snap_dist_m: maximum distance for snapping; points beyond are kept as-is

    Returns: smoothed route list of [lat, lng]
    """
    if not route or len(route) < k + 1:
        return route

    try:
        from scipy.interpolate import make_interp_spline
    except ImportError:
        log("[v8.3-spline] scipy not available, skipping smoothing")
        return route

    ra = np.asarray(route, dtype=np.float64)
    n = len(ra)

    # Parameterize by cumulative arc length
    dists = np.sqrt(np.sum(np.diff(ra, axis=0)**2, axis=1))
    cum = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    if total < 1e-9:
        return route

    # Fit B-spline
    try:
        spline = make_interp_spline(cum, ra, k=min(k, n - 1))
    except Exception:
        return route

    # Evaluate at denser parameter values
    n_out = n * densify_factor
    t_new = np.linspace(0, total, n_out)
    smoothed = spline(t_new)

    # Snap to nearest road nodes
    if G is not None and kdtree_data is not None:
        tree, coords, node_ids = kdtree_data
        _, indices = tree.query(smoothed)
        snapped = coords[indices]

        # Check snap distances — keep original if too far
        snap_dists_deg = np.sqrt(np.sum((smoothed - snapped)**2, axis=1))
        snap_dists_m = snap_dists_deg * 111_000.0
        result = []
        for i in range(n_out):
            if snap_dists_m[i] <= max_snap_dist_m:
                result.append([float(snapped[i, 0]), float(snapped[i, 1])])
            else:
                result.append([float(smoothed[i, 0]), float(smoothed[i, 1])])
    else:
        result = [[float(smoothed[i, 0]), float(smoothed[i, 1])] for i in range(n_out)]

    # Deduplicate consecutive identical points
    deduped = [result[0]]
    for pt in result[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)

    log(f"[v8.3-spline] Smoothed {n} → {len(deduped)} points (k={k})")
    return deduped


# ═══════════════════════════════════════════════════════════════════════════
#  3. MULTI-RESOLUTION ROUTING
# ═══════════════════════════════════════════════════════════════════════════

def multi_res_route(G, ideal_line, config=None):
    """Two-stage routing: coarse on simplified graph, fine on full graph.

    Stage 1: Route on a simplified graph (merged short edges) for global shape.
    Stage 2: For each segment, refine in a fine subgraph around the coarse path.

    Args:
        G: full osmnx graph
        ideal_line: list of [lat, lng]
        config: optional dict with 'coarse_tolerance', 'fine_radius_m'

    Returns: route list of [lat, lng] or None
    """
    import networkx as nx

    config = config or {}
    fine_radius_m = config.get('fine_radius_m', 200)

    if G is None or not ideal_line or len(ideal_line) < 2:
        return None

    try:
        import osmnx as ox
    except ImportError:
        return None

    # Stage 1: Simplify graph (consolidate intersections within 30m)
    try:
        G_simple = ox.simplify_graph(G)
    except Exception:
        G_simple = G

    # Build KDTree for simplified graph
    nodes_s = list(G_simple.nodes(data=True))
    if len(nodes_s) < 10:
        return None
    coords_s = np.array([[nd['y'], nd['x']] for _, nd in nodes_s], dtype=np.float64)
    nids_s = [nid for nid, _ in nodes_s]
    tree_s = cKDTree(coords_s)

    # Route coarsely through simplified graph
    il = np.asarray(ideal_line, dtype=np.float64)
    n_wps = min(20, max(6, len(il) // 3))
    wps = sample_polyline(ideal_line, n_wps)
    wps_a = np.asarray(wps, dtype=np.float64)
    _, indices = tree_s.query(wps_a)
    coarse_nids = [nids_s[i] for i in indices]

    # Deduplicate
    deduped = [coarse_nids[0]]
    for nid in coarse_nids[1:]:
        if nid != deduped[-1]:
            deduped.append(nid)

    coarse_route = []
    for i in range(len(deduped) - 1):
        try:
            path = nx.shortest_path(G_simple, deduped[i], deduped[i+1], weight='length')
            for nid in path:
                pt = [G_simple.nodes[nid]['y'], G_simple.nodes[nid]['x']]
                if not coarse_route or coarse_route[-1] != pt:
                    coarse_route.append(pt)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if len(coarse_route) < 2:
        return None

    # Stage 2: Refine each segment in a local subgraph of the full graph
    from core_router import CoreRouter, _build_kdtree

    # Build corridor around coarse route on the FULL graph
    coarse_a = np.asarray(coarse_route, dtype=np.float64)
    nodes_full = list(G.nodes(data=True))
    coords_full = np.array([[nd['y'], nd['x']] for _, nd in nodes_full], dtype=np.float64)
    nids_full = [nid for nid, _ in nodes_full]
    tree_full = cKDTree(coords_full)

    # Find all full-graph nodes within fine_radius_m of coarse route
    fine_radius_deg = fine_radius_m / 111_000.0
    corridor_indices = set()
    for pt in coarse_a:
        idxs = tree_full.query_ball_point(pt, fine_radius_deg)
        corridor_indices.update(idxs)

    corridor_nodes = {nids_full[i] for i in corridor_indices}
    if len(corridor_nodes) < 20:
        return coarse_route  # Not enough nodes for refinement

    G_fine = G.subgraph(corridor_nodes).copy()
    if not nx.is_weakly_connected(G_fine):
        largest_cc = max(nx.weakly_connected_components(G_fine), key=len)
        G_fine = G_fine.subgraph(largest_cc).copy()

    # Route on the fine subgraph using CoreRouter
    try:
        router = CoreRouter(G_fine, ideal_line)
        fine_route = router.route()
        if fine_route and len(fine_route) >= 2:
            log(f"[v8.3-multires] Coarse {len(coarse_route)} pts → Fine {len(fine_route)} pts")
            return fine_route
    except Exception as e:
        log(f"[v8.3-multires] Fine routing failed: {e}")

    return coarse_route


# ═══════════════════════════════════════════════════════════════════════════
#  4. SYMMETRY PENALTY
# ═══════════════════════════════════════════════════════════════════════════

def symmetry_penalty(route, ideal_line=None):
    """Compute bilateral symmetry error for a route.

    Splits the route into left/right halves, mirrors one, and computes
    the Hausdorff distance between them.

    Args:
        route: list of [lat, lng]
        ideal_line: optional (used for axis estimation)

    Returns: float penalty in metres (0 = perfectly symmetric)
    """
    if not route or len(route) < 4:
        return 0.0

    ra = np.asarray(route, dtype=np.float64)

    # Find the symmetry axis: vertical line through centroid
    centroid = ra.mean(axis=0)

    # Split into left/right based on longitude relative to centroid
    # For hearts, we need to use the ideal shape's axis of symmetry
    if ideal_line and len(ideal_line) >= 4:
        ia = np.asarray(ideal_line, dtype=np.float64)
        axis_lng = ia.mean(axis=0)[1]
    else:
        axis_lng = centroid[1]

    # Split route into top-half (first half) and bottom-half (second half)
    n = len(ra)
    half = n // 2
    left_half = ra[:half]
    right_half = ra[half:]

    if len(left_half) < 2 or len(right_half) < 2:
        return 0.0

    # Mirror the right half around the axis
    right_mirrored = right_half.copy()
    right_mirrored[:, 1] = 2 * axis_lng - right_mirrored[:, 1]

    # Reverse the mirrored half (they should trace in opposite directions)
    right_mirrored = right_mirrored[::-1]

    # Sample both to same number of points
    n_sample = min(50, min(len(left_half), len(right_mirrored)))

    # Ensure ideal_line is passed for heart shapes
    if not ideal_line:
        log("[Warning] Ideal line missing for symmetry axis. Defaulting to centroid.")

    left_s = sample_polyline(left_half.tolist(), n_sample)
    right_s = sample_polyline(right_mirrored.tolist(), n_sample)

    la = np.asarray(left_s, dtype=np.float64)
    rma = np.asarray(right_s, dtype=np.float64)

    # Hausdorff-like metric between the two halves
    dists = haversine_matrix(la[:, 0], la[:, 1], rma[:, 0], rma[:, 1])
    fwd = dists.min(axis=1).mean()
    bwd = dists.min(axis=0).mean()
    return float(fwd + bwd) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  5. FORCED CYCLE CLOSURE
# ═══════════════════════════════════════════════════════════════════════════

def force_close_route(route, G, kdtree_data=None, max_gap_m=100):
    """Ensure a route forms a closed loop for closed shapes.

    If route[-1] != route[0], routes the gap using the graph.
    Only closes if the gap path is < max_gap_m metres.

    Args:
        route: list of [lat, lng]
        G: osmnx graph
        kdtree_data: (tree, coords, node_ids)
        max_gap_m: maximum gap distance to auto-close

    Returns: closed route list of [lat, lng]
    """
    import networkx as nx

    if not route or len(route) < 3:
        return route

    gap = haversine(route[-1][0], route[-1][1], route[0][0], route[0][1])
    if gap < 20:
        # Already closed (within 20m)
        if route[-1] != route[0]:
            route.append(list(route[0]))
        return route

    if gap > max_gap_m:
        log(f"[v8.3-close] Gap {gap:.0f}m > {max_gap_m}m, not closing")
        return route

    if G is None:
        # Simple: just append start point
        route.append(list(route[0]))
        return route

    # Route the gap
    if kdtree_data is None:
        return route

    tree, coords, node_ids = kdtree_data
    end_a = np.array([[route[-1][0], route[-1][1]]])
    start_a = np.array([[route[0][0], route[0][1]]])
    _, end_idx = tree.query(end_a)
    _, start_idx = tree.query(start_a)
    end_nid = node_ids[end_idx[0]]
    start_nid = node_ids[start_idx[0]]

    if end_nid == start_nid:
        route.append(list(route[0]))
        return route

    try:
        gap_path = nx.shortest_path(G, end_nid, start_nid, weight='length')
        # Check gap path length
        gap_length = 0
        for i in range(len(gap_path) - 1):
            u, v = gap_path[i], gap_path[i+1]
            gap_length += G[u][v][list(G[u][v].keys())[0]].get('length', 50)
        if gap_length > max_gap_m * 1.5:
            log(f"[v8.3-close] Gap path too long ({gap_length:.0f}m), not closing")
            return route

        for nid in gap_path[1:]:
            pt = [G.nodes[nid]['y'], G.nodes[nid]['x']]
            if pt != route[-1]:
                route.append(pt)
        log(f"[v8.3-close] Closed gap: {gap:.0f}m via {len(gap_path)} nodes")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        log(f"[v8.3-close] No path to close gap")

    return route


# ═══════════════════════════════════════════════════════════════════════════
#  6. OVERLAP / EDGE-REUSE PENALTY
# ═══════════════════════════════════════════════════════════════════════════

def apply_overlap_penalty(G_sub, previous_route, penalty_factor=3.0):
    """Increase edge weights for edges already used in a previous route.

    This discourages the router from reusing the same roads, reducing
    overlap/backtracking in the final route.

    Args:
        G_sub: corridor subgraph (modified in-place)
        previous_route: list of [lat, lng] from a prior routing attempt
        penalty_factor: cost multiplier for reused edges

    Returns: G_sub (modified)
    """
    if not previous_route or len(previous_route) < 2 or G_sub is None:
        return G_sub

    # Build set of used edges from the previous route
    nodes = list(G_sub.nodes(data=True))
    if not nodes:
        return G_sub

    coords = np.array([[nd['y'], nd['x']] for _, nd in nodes], dtype=np.float64)
    nids = [nid for nid, _ in nodes]
    tree = cKDTree(coords)

    ra = np.asarray(previous_route, dtype=np.float64)
    _, indices = tree.query(ra)
    route_nids = [nids[i] for i in indices]

    # Count edge usage
    used_edges = {}
    for i in range(len(route_nids) - 1):
        u, v = route_nids[i], route_nids[i+1]
        key = (min(u, v), max(u, v))
        used_edges[key] = used_edges.get(key, 0) + 1

    # Apply penalty to reused edges
    n_penalized = 0
    for u, v, data in G_sub.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in used_edges:
            count = used_edges[key]
            for k in G_sub[u][v]:
                old_w = G_sub[u][v][k].get('v8w', data.get('length', 50))
                G_sub[u][v][k]['v8w'] = old_w * (1 + penalty_factor * count)
            n_penalized += 1

    if n_penalized > 0:
        log(f"[v8.3-overlap] Penalized {n_penalized} reused edges (factor={penalty_factor})")

    return G_sub


# ═══════════════════════════════════════════════════════════════════════════
#  7. HYBRID V6+V8 SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def hybrid_v6_coarse_search(G, pts, center, kdtree_data=None,
                            penalty_factor=1.5, n_top=100):
    """v6-style coarse search with proximity scoring, returning top candidates
    for v8 fine routing.

    Uses v6 penalty-based edge weighting for initial screening,
    then returns candidates as (score, rotation, scale, center) tuples
    compatible with the v8 fine search pipeline.

    Args:
        G: osmnx graph
        pts: normalized shape points
        center: [lat, lng]
        kdtree_data: (tree, coords, node_ids)
        penalty_factor: v6-style penalty multiplier
        n_top: number of top candidates to return
    """
    from scoring import coarse_proximity_score
    from geometry import densify

    # Wider rotation search than v8 default
    rotations = list(range(0, 360, 10))  # 36 steps
    scales = [0.007, 0.009, 0.011, 0.014, 0.018, 0.022, 0.027, 0.033]  # 8 scales
    offsets = [(0, 0)]
    # Add some offsets
    for km in [0.5, 1.0, 1.5]:
        deg_lat = km * 0.009
        deg_lng = km * 0.012
        offsets.extend([(deg_lat, 0), (-deg_lat, 0), (0, deg_lng), (0, -deg_lng)])

    results = []
    for dlat, dlng in offsets:
        c = [center[0] + dlat, center[1] + dlng]
        for sc in scales:
            for rot in rotations:
                wps = shape_to_latlngs(pts, c, sc, rot)
                d = densify(wps, spacing_m=150)
                score = coarse_proximity_score(G, d, kdtree_data=kdtree_data)
                # Apply v6 penalty factor to emphasize proximity
                results.append((score * penalty_factor, rot, sc, c))

    results.sort(key=lambda x: x[0])
    log(f"[v8.3-hybrid] v6 coarse: {len(results)} combos, top score={results[0][0]:.1f}")
    return results[:n_top]


# ═══════════════════════════════════════════════════════════════════════════
#  8a. HEART INDENT DETECTION & ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════

def find_heart_indent(ideal_pts):
    """Detect the top indent (V-dip between lobes) in a heart-shaped ideal line.

    The indent is the southernmost point on the top quarter of the shape
    that lies near the shape's vertical center line — the classic dip
    between the two lobes.

    Returns: (lat, lng, deflection_deg) tuple suitable as an apex point,
             or None if not found.
    """
    ia = np.asarray(ideal_pts, dtype=np.float64)
    if len(ia) < 10:
        return None

    centroid = ia.mean(axis=0)
    lat_range = ia[:, 0].max() - ia[:, 0].min()
    lng_range = ia[:, 1].max() - ia[:, 1].min()

    # Top quarter: points with lat > centroid + 0.125 * lat_range (northern half)
    north_threshold = centroid[0] + lat_range * 0.125
    # Near center line: within 10% of lng range from centroid
    center_band = lng_range * 0.10

    candidates = []
    for i in range(len(ia)):
        if ia[i, 0] > north_threshold:
            # In the northern portion — check if near center longitude
            if abs(ia[i, 1] - centroid[1]) < center_band:
                candidates.append((i, ia[i, 0], ia[i, 1]))

    if not candidates:
        return None

    # The indent is the SOUTHERNMOST (lowest lat) of these center-top points
    indent = min(candidates, key=lambda c: c[1])
    log(f"[v8.3-indent] Detected indent at lat={indent[1]:.4f}, lng={indent[2]:.4f}")
    return (indent[1], indent[2], 90.0)  # treat as 90° deflection apex


def indent_proximity_penalty(route, indent_pt, threshold_m=80):
    """Penalize routes that don't come close enough to the heart's top indent.

    Args:
        route: list of [lat, lng]
        indent_pt: (lat, lng) of the indent
        threshold_m: distance beyond which penalty kicks in

    Returns: penalty in metres (0 if route visits the indent)
    """
    if not route or indent_pt is None:
        return 0.0

    ra = np.asarray(route, dtype=np.float64)
    dists = haversine_vector(ra[:, 0], ra[:, 1], indent_pt[0], indent_pt[1])
    min_dist = float(dists.min())

    if min_dist <= threshold_m:
        return 0.0
    return (min_dist - threshold_m) * 3.0


# ═══════════════════════════════════════════════════════════════════════════
#  8b. ENHANCED SCORING (v8.3 composite with symmetry + indent)
# ═══════════════════════════════════════════════════════════════════════════

def score_v83(route, ideal_pts, symmetry_weight=0.0, v6_proximity_weight=0.0,
              indent_weight=0.0, indent_pt=None,
              G=None, kdtree_data=None):
    """v8.3 composite score: v8 metrics + optional symmetry + v6 proximity + indent.

    Args:
        route: list of [lat, lng]
        ideal_pts: list of [lat, lng]
        symmetry_weight: 0.0–0.5, how much to weight bilateral symmetry
        v6_proximity_weight: 0.0–0.3, how much to blend v6 proximity scoring
        G: graph (for v6 proximity)
        kdtree_data: for v6 proximity

    Returns: float score (lower = better)
    """
    from scoring_v8 import score_v8

    base_score = score_v8(route, ideal_pts)

    if symmetry_weight > 0:
        sym = symmetry_penalty(route, ideal_pts)
        base_score = base_score * (1.0 - symmetry_weight) + sym * symmetry_weight

    if v6_proximity_weight > 0 and G is not None:
        from scoring import coarse_proximity_score
        from geometry import densify
        # Approximate proximity of route to road network
        n_sample = min(40, len(route))
        route_s = sample_polyline(route, n_sample)
        prox = coarse_proximity_score(G, route_s, kdtree_data=kdtree_data)
        base_score = base_score * (1.0 - v6_proximity_weight) + prox * v6_proximity_weight

    if indent_weight > 0 and indent_pt is not None:
        indent_pen = indent_proximity_penalty(route, indent_pt)
        base_score += indent_weight * indent_pen

    return base_score


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN: v8.3 Enhanced fit_and_score
# ═══════════════════════════════════════════════════════════════════════════

def fit_and_score_v83(G, pts, rot, scale, center, config=None,
                      kdtree_data=None):
    """v8.3/v8.4 enhanced routing pipeline with configurable features.

    Config keys (v8.3):
        dynamic_densify: bool — use road-curvature-aware densification
        spline_k: int or None — B-spline degree for post-smoothing (3/4/5)
        multi_res: bool — use multi-resolution routing
        symmetry_weight: float — symmetry penalty weight (0.0–0.5)
        penalty_factor: float — v6-style penalty scaling (default 1.0)
        force_close: bool — force loop closure
        overlap_penalty: float — edge reuse penalty factor (0 = off)
        v6_proximity_weight: float — blend v6 proximity into scoring
        indent_enforce: bool — enforce heart top indent via apex injection
        indent_weight: float — indent proximity penalty weight (0.0–1.0)

    Config keys (v8.4 — Perceptual & Road-Aware):
        density_auto_scale: bool — auto-shift center to high-density area
        use_road_hierarchy: bool — apply pedestrian-friendly road bonus
        use_skeleton_score: bool + skeleton_weight: float
        use_fgw: bool + fgw_weight: float
        use_perceptual_loss: bool + perceptual_weight: float
        use_ph_topology: bool + ph_weight: float
        v84_blend: float — weight of v84 total in final score (default 0.4)

    Returns: (score, route) tuple   — also (score, route, v84_detail)
             if any v8.4 flag is active
    """
    from core_router import CoreRouter, _build_kdtree
    from scoring_v8 import score_v8, frechet_normalized

    config = config or {}
    enable_dynamic = config.get('dynamic_densify', False)
    spline_k = config.get('spline_k', None)
    enable_multires = config.get('multi_res', False)
    sym_weight = config.get('symmetry_weight', 0.0)
    penalty_factor = config.get('penalty_factor', 1.0)
    enable_close = config.get('force_close', False)
    overlap_pen = config.get('overlap_penalty', 0.0)
    v6_prox_weight = config.get('v6_proximity_weight', 0.0)
    refine_iterations = config.get('refine_iterations', 0)
    enable_indent = config.get('indent_enforce', False)
    indent_weight = config.get('indent_weight', 0.0)

    # ── v8.4: detect if any new flag is on ──
    v84_flags = ['density_auto_scale', 'use_road_hierarchy',
                 'use_skeleton_score', 'use_fgw',
                 'use_perceptual_loss', 'use_ph_topology']
    has_v84 = any(config.get(f, False) for f in v84_flags)

    # ── v8.4 Step 0: density auto-scale (shift center) ──
    if config.get('density_auto_scale', False) and G:
        from v84_perceptual import maybe_auto_scale_center
        center = maybe_auto_scale_center(center, pts, G, config)

    # Step 1: Shape to geographic coords
    wps = shape_to_latlngs(pts, center, scale, rot)

    # Step 2: Densify (standard or dynamic)
    if enable_dynamic and G:
        ideal = dynamic_densify(wps, G, kdtree_data=kdtree_data)
    else:
        ideal = adaptive_densify(wps, base_spacing=35, curve_spacing=15)

    if not G:
        from routing import route_osrm
        dense = adaptive_densify(wps, base_spacing=40, curve_spacing=18)
        route = route_osrm(dense)
        if not route:
            return (1e9, None)
        return (score_v8(route, ideal), route)

    # Step 2b: Detect heart indent for apex injection
    indent_pt = None
    if enable_indent:
        indent_info = find_heart_indent(ideal)
        if indent_info:
            indent_pt = (indent_info[0], indent_info[1])

    # Step 3: Route
    route = None
    try:
        if enable_multires:
            route = multi_res_route(G, ideal, config)

        if not route or len(route) < 2:
            # Standard CoreRouter pipeline
            router_config = {}
            if penalty_factor != 1.0:
                router_config['beta'] = 0.0003 * penalty_factor

            # Indent enforcement: lower apex threshold + inject indent as apex
            if enable_indent:
                router_config['apex_threshold'] = 40
                router_config['apex_radius_m'] = 50.0
                if indent_pt:
                    router_config['extra_apex_points'] = [
                        (indent_pt[0], indent_pt[1], 90.0)
                    ]

            # v8.4: road-hierarchy flag passed into router config
            if config.get('use_road_hierarchy', False):
                router_config['use_road_hierarchy'] = True

            router = CoreRouter(G, ideal, config=router_config)

            # ── v8.4: apply road-hierarchy bonus post-weight ──
            if config.get('use_road_hierarchy', False) and router.G_sub:
                from v84_perceptual import apply_road_hierarchy_bonus
                apply_road_hierarchy_bonus(router.G_sub)

            route = router.route()
    except Exception as e:
        log(f"[v8.3] Routing failed: {e}")

    if not route or len(route) < 2:
        return (1e9, None)

    # Step 4: Post-processing
    kd = kdtree_data or _build_kdtree(G)

    if enable_close:
        route = force_close_route(route, G, kdtree_data=kd)

    if spline_k is not None and spline_k >= 2:
        route = spline_smooth_route(route, G, kdtree_data=kd, k=spline_k)

    # Step 5: v8.3 Score
    v83_score = score_v83(route, ideal,
                          symmetry_weight=sym_weight,
                          v6_proximity_weight=v6_prox_weight,
                          indent_weight=indent_weight,
                          indent_pt=indent_pt,
                          G=G, kdtree_data=kd)

    # Fréchet gate
    fd_norm = frechet_normalized(route, ideal)
    if fd_norm > 0.12:
        v83_score *= 2.0
        log(f"[v8.3] Fréchet gate: fd_norm={fd_norm:.3f} → score doubled to {v83_score:.1f}")

    # ── Step 6: v8.4 Perceptual scores ──
    if has_v84:
        from v84_perceptual import score_v84
        v84_detail = score_v84(route, ideal, config, G=G, kdtree_data=kd)
        v84_total = v84_detail['v84_total']
        blend = config.get('v84_blend', 0.4)
        score = v83_score * (1.0 - blend) + v84_total * blend
        log(f"[v8.4] v83={v83_score:.1f} v84={v84_total:.1f} "
            f"blend={blend:.2f} → {score:.1f}")
        return (score, route)

    return (v83_score, route)
