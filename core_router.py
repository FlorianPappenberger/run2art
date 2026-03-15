"""
core_router.py — Soft-Constraint CoreRouter for Run2Art v8.0
=============================================================
Eliminates 'trunks' by replacing hard waypoint routing with a continuous
edge-cost field that penalizes deviation from the template center-line,
heading misalignment, and illegal U-turns.

Key innovations:
  - Vectorized edge-weight precomputation (NumPy — no per-call Python)
  - Heading penalty: edges deviating >20° from local tangent are penalized
  - Apex sensitivity: U-turns only allowed within 15m of sharp vertices
  - Bidirectional A* with shape-following heuristic
  - Adaptive wide-tube corridor (narrow at curves, wide on straights)
"""

import sys
import math
import heapq
import numpy as np
from scipy.spatial import cKDTree

from geometry import (
    haversine, haversine_vector, haversine_matrix,
    point_to_segment_dist, point_to_segments_vectorized,
    sample_polyline, adaptive_densify, identify_anchor_points,
    compute_tangent_field, detect_sharp_vertices,
    normalize_to_unit_box, edge_bearing, bearing_deviation,
)
from scoring_v8 import score_v8, frechet_score


def log(msg):
    print(msg, file=sys.stderr, flush=True)


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
#  HEADING PENALTY
# ═══════════════════════════════════════════════════════════════════════════

def heading_penalty_vectorized(edge_bearings, local_tangents):
    """Vectorized heading penalty for arrays of edge bearings vs local tangents.

    Args:
        edge_bearings: (E,) array of compass bearings
        local_tangents: (E,) array of local shape tangent bearings

    Returns:
        (E,) array of penalty values in metres
    """
    dev = np.abs(edge_bearings - local_tangents) % 360.0
    dev = np.where(dev > 180.0, 360.0 - dev, dev)
    dev = np.where(dev > 90.0, 180.0 - dev, dev)
    # dev is now in [0, 90]: 0 = aligned, 90 = perpendicular

    penalty = np.zeros_like(dev)
    # 0-20°: no penalty (aligned)
    mask_mid = (dev >= 20.0) & (dev < 45.0)
    penalty[mask_mid] = (dev[mask_mid] - 20.0) / 25.0 * 100.0

    mask_high = (dev >= 45.0) & (dev < 70.0)
    penalty[mask_high] = 100.0 + (dev[mask_high] - 45.0) ** 2 * 0.40

    mask_vhigh = (dev >= 70.0) & (dev < 80.0)
    penalty[mask_vhigh] = 350.0

    mask_perp = dev >= 80.0
    penalty[mask_perp] = 500.0

    return penalty


# ═══════════════════════════════════════════════════════════════════════════
#  U-TURN PENALTY
# ═══════════════════════════════════════════════════════════════════════════

def compute_uturn_mask(G_sub, edge_midpoints, apex_points, apex_radius_m=15.0):
    """Compute which edges are near sharp vertices (where U-turns are allowed).

    Args:
        G_sub: corridor subgraph (unused but available for future)
        edge_midpoints: (E, 2) array of [lat, lng]
        apex_points: list of (lat, lng, deflection_deg)
        apex_radius_m: radius around apex where U-turns are free

    Returns:
        (E,) boolean array — True if edge is near an apex (U-turn ok)
    """
    if not apex_points:
        return np.zeros(len(edge_midpoints), dtype=bool)

    apex_coords = np.array([[a[0], a[1]] for a in apex_points], dtype=np.float64)
    mids = np.asarray(edge_midpoints, dtype=np.float64)

    # Distance from each edge midpoint to each apex
    dists = haversine_matrix(mids[:, 0], mids[:, 1],
                             apex_coords[:, 0], apex_coords[:, 1])
    # True if within apex_radius_m of ANY apex
    return dists.min(axis=1) <= apex_radius_m


# ═══════════════════════════════════════════════════════════════════════════
#  ADAPTIVE WIDE-TUBE CORRIDOR
# ═══════════════════════════════════════════════════════════════════════════

def build_adaptive_corridor(G, ideal_line, base_radius_m=200,
                            min_radius_m=150, max_radius_m=500):
    """Build corridor with variable-width tube: narrow at curves, wide on straights.

    Returns (G_sub, kd_sub) or (G, kd_full) if pruning fails.
    """
    try:
        import osmnx as ox
        import networkx as nx
    except ImportError:
        return G, _build_kdtree(G)

    if G is None:
        return G, None

    il = np.asarray(ideal_line, dtype=np.float64)
    if len(il) < 2:
        return G, _build_kdtree(G)

    # Compute per-point curvature for adaptive radius
    radii = _adaptive_radii(il, base_radius_m, min_radius_m, max_radius_m)

    try:
        from shapely.geometry import Point, LineString
        from shapely.ops import unary_union

        # Build variable-width corridor as union of circles
        circles = []
        for i in range(len(il)):
            r_deg = radii[i] / 111_000.0
            p = Point(il[i, 1], il[i, 0])  # (lng, lat)
            circles.append(p.buffer(r_deg))

        corridor_shape = unary_union(circles)

        nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
        candidates = nodes_gdf.sindex.query(corridor_shape, predicate='intersects')
        corridor_nodes = set(nodes_gdf.index[candidates])

        if len(corridor_nodes) < 20:
            # Iterative widening: 100m steps up to 800m
            for widen_m in range(100, 900, 100):
                corridor_shape = corridor_shape.buffer(widen_m / 111_000.0)
                candidates = nodes_gdf.sindex.query(corridor_shape, predicate='intersects')
                corridor_nodes = set(nodes_gdf.index[candidates])
                if len(corridor_nodes) >= 20:
                    break

        if len(corridor_nodes) < 10:
            log(f"[v8-corridor] Too few nodes ({len(corridor_nodes)}), using full graph")
            return G, _build_kdtree(G)

        G_sub = G.subgraph(corridor_nodes).copy()

        if not nx.is_weakly_connected(G_sub):
            largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
            if len(largest_cc) < len(corridor_nodes) * 0.5:
                # Too fragmented — widen
                wider = corridor_shape.buffer(base_radius_m / 111_000.0)
                candidates = nodes_gdf.sindex.query(wider, predicate='intersects')
                corridor_nodes = set(nodes_gdf.index[candidates])
                G_sub = G.subgraph(corridor_nodes).copy()
                largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
            G_sub = G_sub.subgraph(largest_cc).copy()

        log(f"[v8-corridor] Adaptive: {G.number_of_nodes()} -> {G_sub.number_of_nodes()} nodes")
        return G_sub, _build_kdtree(G_sub)

    except Exception as e:
        log(f"[v8-corridor] Shapely corridor failed ({e}), using KDTree fallback")
        return _corridor_kdtree_adaptive(G, il, radii)


def _adaptive_radii(il, base_m, min_m, max_m):
    """Compute per-point tube radius based on local curvature."""
    n = len(il)
    radii = np.full(n, base_m, dtype=np.float64)

    if n < 3:
        return radii

    # Compute curvature at interior points
    d1_lat = il[1:-1, 0] - il[:-2, 0]
    d1_lng = il[1:-1, 1] - il[:-2, 1]
    d2_lat = il[2:, 0] - il[1:-1, 0]
    d2_lng = il[2:, 1] - il[1:-1, 1]

    a1 = np.arctan2(d1_lng, d1_lat)
    a2 = np.arctan2(d2_lng, d2_lat)
    deflection = np.abs(((np.degrees(a2 - a1) + 180) % 360) - 180)

    # High curvature (>60°) → narrow tube; low curvature → wide tube
    # Linear interpolation: 0° → max_m, 180° → min_m
    t = np.clip(deflection / 180.0, 0.0, 1.0)
    radii[1:-1] = max_m * (1.0 - t) + min_m * t

    # Endpoints inherit from neighbors
    radii[0] = radii[1]
    radii[-1] = radii[-2]

    return radii


def _corridor_kdtree_adaptive(G, il, radii):
    """KDTree fallback with adaptive per-point radii."""
    import networkx as nx
    kd_full = _build_kdtree(G)
    if kd_full is None:
        return G, None
    tree, coords, node_ids = kd_full

    corridor_indices = set()
    for i in range(len(il)):
        r_deg = radii[i] / 111_000.0
        indices = tree.query_ball_point([il[i, 0], il[i, 1]], r_deg)
        corridor_indices.update(indices)

    # Midpoints
    for i in range(len(il) - 1):
        mid = [(il[i, 0] + il[i + 1, 0]) / 2, (il[i, 1] + il[i + 1, 1]) / 2]
        r_deg = (radii[i] + radii[i + 1]) / 2.0 / 111_000.0
        indices = tree.query_ball_point(mid, r_deg)
        corridor_indices.update(indices)

    corridor_nodes = {node_ids[i] for i in corridor_indices}
    if len(corridor_nodes) < 10:
        return G, kd_full

    G_sub = G.subgraph(corridor_nodes).copy()
    if nx.is_weakly_connected(G_sub):
        return G_sub, _build_kdtree(G_sub)

    largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
    G_sub = G_sub.subgraph(largest_cc).copy()
    if G_sub.number_of_nodes() < 10:
        return G, kd_full

    return G_sub, _build_kdtree(G_sub)


def _build_kdtree(G):
    if G is None:
        return None
    nodes = list(G.nodes(data=True))
    coords = np.array([[n[1]['y'], n[1]['x']] for n in nodes], dtype=np.float64)
    node_ids = [n[0] for n in nodes]
    return (cKDTree(coords), coords, node_ids)


# ═══════════════════════════════════════════════════════════════════════════
#  VECTORIZED EDGE-WEIGHT PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def prune_perpendicular_edges(G_sub, ideal_line, tangent_field, max_deviation_deg=35.0):
    """Remove edges whose heading deviates >max_deviation_deg from the local tangent.

    Preserves edges in low-density areas (>150m from ideal) for connectivity.
    Only prunes if the resulting graph remains weakly connected with >20 nodes.
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

    # Compute edge bearings and nearest tangents
    u_lats = np.array([G_sub.nodes[e[0]]['y'] for e in edges])
    u_lngs = np.array([G_sub.nodes[e[0]]['x'] for e in edges])
    v_lats = np.array([G_sub.nodes[e[1]]['y'] for e in edges])
    v_lngs = np.array([G_sub.nodes[e[1]]['x'] for e in edges])

    edge_bearings = np.degrees(np.arctan2(v_lngs - u_lngs, v_lats - u_lats)) % 360.0
    mid_lats = (u_lats + v_lats) * 0.5
    mid_lngs = (u_lngs + v_lngs) * 0.5

    # Distance from each midpoint to nearest ideal segment
    midpoints = np.column_stack([mid_lats, mid_lngs])
    prox_dists = point_to_segments_vectorized(midpoints, il[:-1], il[1:])

    # Find nearest tangent
    seg_mids_lat = (il[:-1, 0] + il[1:, 0]) * 0.5
    seg_mids_lng = (il[:-1, 1] + il[1:, 1]) * 0.5
    d_to_segs = haversine_matrix(mid_lats, mid_lngs, seg_mids_lat, seg_mids_lng)
    nearest_seg = d_to_segs.argmin(axis=1)
    local_tangents = (tf[nearest_seg] + tf[np.minimum(nearest_seg + 1, len(tf) - 1)]) * 0.5

    # Compute deviation (bidirectional)
    dev = np.abs(edge_bearings - local_tangents) % 360.0
    dev = np.where(dev > 180.0, 360.0 - dev, dev)
    dev = np.where(dev > 90.0, 180.0 - dev, dev)

    # Mark edges to prune: high deviation AND close to ideal line
    to_prune = (dev > max_deviation_deg) & (prox_dists < 150.0)

    edges_to_remove = [(e[0], e[1], e[2]) for e, prune in zip(edges, to_prune) if prune]
    if not edges_to_remove:
        return G_sub

    # Check that pruning doesn't break connectivity
    G_test = G_sub.copy()
    G_test.remove_edges_from(edges_to_remove)
    if G_test.number_of_edges() < 10:
        return G_sub
    if not nx.is_weakly_connected(G_test):
        # Only keep the largest component, but check it's big enough
        largest_cc = max(nx.weakly_connected_components(G_test), key=len)
        if len(largest_cc) < G_sub.number_of_nodes() * 0.6:
            return G_sub  # Too much damage, skip pruning
        G_test = G_test.subgraph(largest_cc).copy()

    n_pruned = len(edges_to_remove)
    log(f"[v8.1-prune] Removed {n_pruned}/{n_edges} perpendicular edges (>{max_deviation_deg}°)")
    return G_test


def precompute_edge_weights(G_sub, ideal_line, tangent_field, apex_points,
                            w_head=8.0, w_uturn=1.0, beta=0.0003,
                            tube_radius_m=300, apex_radius_m=15.0):
    """v8.1: Multiplicative penalty field with additive heading/uturn.

    Cost = L × H × (1 + β·d²) + w_head·heading + w_uturn·uturn

    Assigns attribute 'v8w' to each edge in G_sub.
    Returns: G_sub (modified in-place with 'v8w' attributes)
    """
    il = np.asarray(ideal_line, dtype=np.float64)
    tf = np.asarray(tangent_field, dtype=np.float64)
    n_seg = len(il) - 1

    edges = list(G_sub.edges(data=True))
    n_edges = len(edges)
    if n_edges == 0:
        return G_sub

    # Extract edge endpoints and data
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

    # Edge midpoints
    mid_lats = (u_lats + v_lats) * 0.5
    mid_lngs = (u_lngs + v_lngs) * 0.5
    midpoints = np.column_stack([mid_lats, mid_lngs])

    # ── C_network: length × highway_multiplier ──
    c_network = lengths * hw_mults

    # ── C_proximity: perpendicular distance to ideal line ──
    if n_seg > 0:
        c_proximity = point_to_segments_vectorized(
            midpoints, il[:-1], il[1:]
        )
    else:
        c_proximity = np.zeros(n_edges)

    # ── C_heading: heading deviation penalty ──
    # Edge bearings
    edge_bearings = np.degrees(np.arctan2(v_lngs - u_lngs, v_lats - u_lats)) % 360.0

    # Find nearest ideal segment for each edge midpoint
    if n_seg > 0:
        seg_mids_lat = (il[:-1, 0] + il[1:, 0]) * 0.5
        seg_mids_lng = (il[:-1, 1] + il[1:, 1]) * 0.5
        # Distance from each edge midpoint to each segment midpoint
        d_to_segs = haversine_matrix(mid_lats, mid_lngs, seg_mids_lat, seg_mids_lng)
        nearest_seg = d_to_segs.argmin(axis=1)  # (E,) index of nearest segment
        # Tangent at nearest segment = average of tangent at endpoints
        local_tangents = (tf[nearest_seg] + tf[np.minimum(nearest_seg + 1, len(tf) - 1)]) * 0.5
    else:
        local_tangents = np.zeros(n_edges)

    c_heading = heading_penalty_vectorized(edge_bearings, local_tangents)

    # ── Curvature-adaptive beta (Phase 2) ──
    # Higher beta at sharp curves forces routes closer to ideal at cusps/clefts
    if n_seg > 1:
        bearing_changes = np.abs(np.diff(tf))
        bearing_changes = np.minimum(bearing_changes, 360.0 - bearing_changes)
        seg_kappa = np.clip(bearing_changes / 90.0, 0.0, 2.0)
        edge_kappa = seg_kappa[nearest_seg]
        local_beta = beta * (1.0 + 2.0 * edge_kappa)
    else:
        local_beta = beta

    # ── C_uturn: penalty for edges that reverse direction near non-apex areas ──
    near_apex = compute_uturn_mask(G_sub, midpoints, apex_points, apex_radius_m=apex_radius_m)
    reverse_dev = np.abs(edge_bearings - local_tangents) % 360.0
    reverse_dev = np.where(reverse_dev > 180.0, 360.0 - reverse_dev, reverse_dev)
    is_reversal = reverse_dev > 150.0  # nearly opposite direction
    c_uturn = np.where(is_reversal & ~near_apex, 5000.0, 0.0)

    # ── v8.1 Multiplicative penalty field ──
    # C = L × H × (1 + β_local·d²) + heading + uturn
    base_cost = lengths * hw_mults * (1.0 + local_beta * c_proximity ** 2)
    weights = base_cost + w_head * c_heading + w_uturn * c_uturn

    # Assign to graph edges
    for k, (u, v, data) in enumerate(edges):
        G_sub[u][v][list(G_sub[u][v].keys())[0]]['v8w'] = float(weights[k])

    log(f"[v8.1-weights] Precomputed {n_edges} edge weights "
        f"(prox={c_proximity.mean():.0f}m, head={c_heading.mean():.0f}m, "
        f"uturn={int(c_uturn.sum() > 0)} reversals, beta={beta})")

    return G_sub


# ═══════════════════════════════════════════════════════════════════════════
#  BIDIRECTIONAL A*
# ═══════════════════════════════════════════════════════════════════════════

def bidirectional_astar(G, source, target, weight_attr='v8w',
                        ideal_line=None, shape_bonus=0.2):
    """Bidirectional A* with shape-following heuristic.

    Searches from both source and target simultaneously.
    Shape bonus: nodes within 50m of ideal_line get reduced heuristic.

    Returns: list of node IDs, or None
    """
    if source == target:
        return [source]

    if source not in G or target not in G:
        return None

    # Build shape KDTree for bonus (if ideal_line provided)
    shape_tree = None
    if ideal_line is not None and len(ideal_line) >= 2:
        il = np.asarray(ideal_line, dtype=np.float64)
        shape_tree = cKDTree(il)

    def h_forward(n):
        lat, lon = G.nodes[n]['y'], G.nodes[n]['x']
        lat2, lon2 = G.nodes[target]['y'], G.nodes[target]['x']
        d = haversine(lat, lon, lat2, lon2)
        # Shape bonus: reduce heuristic for nodes near ideal line
        if shape_tree is not None:
            dist_deg, _ = shape_tree.query([lat, lon])
            dist_m = dist_deg * 111_000.0
            if dist_m < 50.0:
                d *= (1.0 - shape_bonus)
        return d * 0.8  # admissibility factor

    def h_backward(n):
        lat, lon = G.nodes[n]['y'], G.nodes[n]['x']
        lat2, lon2 = G.nodes[source]['y'], G.nodes[source]['x']
        d = haversine(lat, lon, lat2, lon2)
        if shape_tree is not None:
            dist_deg, _ = shape_tree.query([lat, lon])
            dist_m = dist_deg * 111_000.0
            if dist_m < 50.0:
                d *= (1.0 - shape_bonus)
        return d * 0.8

    # Forward search data
    f_open = [(h_forward(source), 0.0, source)]  # (f, g, node)
    f_g = {source: 0.0}
    f_parent = {source: None}
    f_closed = set()

    # Backward search data
    b_open = [(h_backward(target), 0.0, target)]
    b_g = {target: 0.0}
    b_parent = {target: None}
    b_closed = set()

    best_cost = float('inf')
    best_meeting = None
    max_iterations = min(G.number_of_nodes() * 2, 50000)

    for _ in range(max_iterations):
        # Expand forward
        if f_open:
            f_f, f_gc, f_node = heapq.heappop(f_open)
            if f_gc > f_g.get(f_node, float('inf')):
                pass  # stale entry
            elif f_node not in f_closed:
                f_closed.add(f_node)
                # Check meeting
                if f_node in b_g:
                    cost = f_g[f_node] + b_g[f_node]
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = f_node

                for _, neighbor, data in G.edges(f_node, data=True):
                    if neighbor in f_closed:
                        continue
                    w = data.get(weight_attr, data.get('length', 50.0))
                    new_g = f_g[f_node] + w
                    if new_g < f_g.get(neighbor, float('inf')):
                        f_g[neighbor] = new_g
                        f_parent[neighbor] = f_node
                        heapq.heappush(f_open, (new_g + h_forward(neighbor),
                                                 new_g, neighbor))

        # Expand backward
        if b_open:
            b_f, b_gc, b_node = heapq.heappop(b_open)
            if b_gc > b_g.get(b_node, float('inf')):
                pass
            elif b_node not in b_closed:
                b_closed.add(b_node)
                if b_node in f_g:
                    cost = f_g[b_node] + b_g[b_node]
                    if cost < best_cost:
                        best_cost = cost
                        best_meeting = b_node

                # Backward: traverse predecessors (incoming edges)
                for pred, _, data in G.in_edges(b_node, data=True):
                    if pred in b_closed:
                        continue
                    w = data.get(weight_attr, data.get('length', 50.0))
                    new_g = b_g[b_node] + w
                    if new_g < b_g.get(pred, float('inf')):
                        b_g[pred] = new_g
                        b_parent[pred] = b_node
                        heapq.heappush(b_open, (new_g + h_backward(pred),
                                                 new_g, pred))

        # Termination check
        f_min = f_open[0][0] if f_open else float('inf')
        b_min = b_open[0][0] if b_open else float('inf')
        if f_min + b_min >= best_cost:
            break
        if not f_open and not b_open:
            break

    if best_meeting is None:
        return None

    # Reconstruct path
    path_fwd = []
    n = best_meeting
    while n is not None:
        path_fwd.append(n)
        n = f_parent.get(n)
    path_fwd.reverse()

    path_bwd = []
    n = b_parent.get(best_meeting)
    while n is not None:
        path_bwd.append(n)
        n = b_parent.get(n)

    return path_fwd + path_bwd


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE-DISJOINT CYCLE FINDER (for closed shapes)
# ═══════════════════════════════════════════════════════════════════════════

def find_shape_cycle(G_sub, ideal_line, kdtree_data=None, weight_attr='v8w'):
    """Find a closed loop in the corridor that best approximates the shape.

    Strategy: Edge-disjoint path pairs
    1. Find start_node (nearest to ideal[0])
    2. Find opposite_node (nearest to ideal[n//2])
    3. Route start→opposite via best path (clockwise half)
    4. Remove those edges, route start→opposite again (counter-clockwise half)
    5. Concatenate: path_cw + reverse(path_ccw) = closed loop

    Falls back to single-path loop if edge-disjoint fails.
    """
    import networkx as nx

    il = np.asarray(ideal_line, dtype=np.float64)
    n = len(il)
    if n < 4:
        return None

    # Find start and opposite nodes
    if kdtree_data is not None:
        tree, coords, nids = kdtree_data
        _, start_idx = tree.query([il[0, 0], il[0, 1]])
        _, opp_idx = tree.query([il[n // 2, 0], il[n // 2, 1]])
        start_node = nids[start_idx]
        opp_node = nids[opp_idx]
    else:
        return None

    if start_node == opp_node:
        return None

    # Path 1: start → opposite (will use the "best" edges)
    try:
        path1 = nx.shortest_path(G_sub, start_node, opp_node, weight=weight_attr)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    # Path 2: start → opposite on a modified graph (penalize path1 edges)
    G_mod = G_sub.copy()
    for i in range(len(path1) - 1):
        u, v = path1[i], path1[i + 1]
        if G_mod.has_edge(u, v):
            for key in G_mod[u][v]:
                G_mod[u][v][key][weight_attr] = G_mod[u][v][key].get(weight_attr, 50.0) * 100.0
        if G_mod.has_edge(v, u):
            for key in G_mod[v][u]:
                G_mod[v][u][key][weight_attr] = G_mod[v][u][key].get(weight_attr, 50.0) * 100.0

    try:
        path2 = nx.shortest_path(G_mod, start_node, opp_node, weight=weight_attr)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    # Check that paths are sufficiently different (not using same edges)
    edges1 = set(zip(path1[:-1], path1[1:]))
    edges2 = set(zip(path2[:-1], path2[1:]))
    overlap = len(edges1 & edges2) + len(edges1 & {(v, u) for u, v in edges2})
    if overlap > len(edges1) * 0.3:
        # Too much overlap — paths aren't truly disjoint
        return None

    # Combine: path1 + reverse(path2) = closed loop
    cycle_nodes = path1 + list(reversed(path2[1:-1]))

    # Convert to coordinates
    route = []
    for nid in cycle_nodes:
        pt = [G_sub.nodes[nid]['y'], G_sub.nodes[nid]['x']]
        if not route or route[-1] != pt:
            route.append(pt)

    # Close the loop
    if route and route[0] != route[-1]:
        route.append(route[0])

    return route if len(route) >= 4 else None


# ═══════════════════════════════════════════════════════════════════════════
#  CORE ROUTER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class CoreRouter:
    """Soft-constraint router that eliminates trunks.

    Usage:
        router = CoreRouter(G, ideal_line)
        route = router.route()
        score = router.score(route)
    """

    def __init__(self, G, ideal_line, config=None):
        self.G = G
        self.ideal_line = ideal_line
        self.il = np.asarray(ideal_line, dtype=np.float64)
        self.config = config or {}

        # Precompute shape properties
        self.tangent_field = compute_tangent_field(ideal_line)
        self.apex_points = detect_sharp_vertices(
            ideal_line,
            angle_threshold=self.config.get('apex_threshold', 120)
        )
        # Allow injection of additional apex points (e.g. heart indent)
        extra = self.config.get('extra_apex_points', [])
        if extra:
            self.apex_points.extend(extra)
        self.is_closed = self._check_closed()

        # Build adaptive corridor
        self.G_sub, self.kd_sub = build_adaptive_corridor(
            G, ideal_line,
            base_radius_m=self.config.get('base_radius_m', 250),
            min_radius_m=self.config.get('min_radius_m', 150),
            max_radius_m=self.config.get('max_radius_m', 500),
        )

        # v8.1: Tangent-prune perpendicular edges before weighting
        if self.G_sub is not None and self.G_sub.number_of_edges() > 0:
            self.G_sub = prune_perpendicular_edges(
                self.G_sub, ideal_line, self.tangent_field,
                max_deviation_deg=self.config.get('max_tangent_dev', 35.0),
            )
            self.kd_sub = _build_kdtree(self.G_sub)

        # Precompute edge weights (v8.1 multiplicative formula)
        if self.G_sub is not None and self.G_sub.number_of_edges() > 0:
            precompute_edge_weights(
                self.G_sub, ideal_line, self.tangent_field, self.apex_points,
                w_head=self.config.get('w_head', 8.0),
                w_uturn=self.config.get('w_uturn', 1.0),
                beta=self.config.get('beta', 0.0003),
                tube_radius_m=self.config.get('tube_radius_m', 300),
                apex_radius_m=self.config.get('apex_radius_m', 15.0),
            )

        log(f"[CoreRouter] apexes={len(self.apex_points)}, "
            f"closed={self.is_closed}, "
            f"corridor={self.G_sub.number_of_nodes() if self.G_sub else 0} nodes")

        # Phase 8: Elastic deformation - pull ideal line towards roads for routing
        self.routing_line = self._elastic_deform()

    def _elastic_deform(self, alpha=0.15, iterations=2):
        """Gently deform ideal line towards nearby road nodes for routing.

        Returns deformed line as list of [lat, lng]. Original self.ideal_line
        is preserved for scoring.
        """
        if self.kd_sub is None or self.G_sub is None or self.G_sub.number_of_nodes() < 5:
            return self.ideal_line
        il = self.il.copy()
        tree, coords, _ = self.kd_sub
        n = len(il)
        for it in range(iterations):
            decay = alpha / (it + 1)
            _, indices = tree.query(il)
            nearest = coords[indices]
            # Only deform interior points (preserve start/end for closed shapes)
            for i in range(1, n - 1):
                il[i] += decay * (nearest[i] - il[i])
        return il.tolist()

    def _check_closed(self):
        """Check if shape is closed (first ≈ last point)."""
        if len(self.il) < 3:
            return False
        return haversine(self.il[0, 0], self.il[0, 1],
                         self.il[-1, 0], self.il[-1, 1]) < 50.0

    def route(self):
        """Route the shape using soft-constraint bidirectional A*.

        For closed shapes: tries cycle-based routing first.
        For open shapes: routes through anchor waypoints.

        Returns: list of [lat, lng] or None
        """
        if self.G_sub is None:
            return None

        # Strategy 1: Cycle-based routing (closed shapes only)
        if self.is_closed:
            cycle_route = find_shape_cycle(
                self.G_sub, self.ideal_line,
                kdtree_data=self.kd_sub, weight_attr='v8w'
            )
            if cycle_route and len(cycle_route) >= 4:
                # Validate: is the cycle actually close to the shape?
                fd = frechet_score(cycle_route, self.ideal_line)
                if fd < 200:  # reasonable threshold
                    log(f"[CoreRouter] Cycle route: Frechet={fd:.1f}m")
                    return cycle_route

        # Strategy 2: Soft-constraint segment routing
        route = self._route_segments()

        # Phase 10: Try routing from alternative start positions (beam-like)
        if self.is_closed and route:
            best_route = route
            best_fd = frechet_score(route, self.ideal_line)
            n_pts = len(self.il)
            for shift in [n_pts // 3, 2 * n_pts // 3]:
                shifted_il = self.il[shift:].tolist() + self.il[:shift].tolist()
                self_backup_il = self.ideal_line
                self.ideal_line = shifted_il
                self.il = np.asarray(shifted_il, dtype=np.float64)
                alt = self._route_segments()
                self.ideal_line = self_backup_il
                self.il = np.asarray(self_backup_il, dtype=np.float64)
                if alt:
                    alt_fd = frechet_score(alt, self_backup_il)
                    if alt_fd < best_fd:
                        best_fd = alt_fd
                        best_route = alt
            route = best_route

        if route:
            return route

        # Strategy 3: Fallback to anchor-point routing with v8 weights
        return self._route_anchors()

    def _route_segments(self):
        """v8.1: Route with trunk-killer skip logic.

        Evaluates each waypoint's detour cost vs skip cost.
        Drops waypoints where detour/direct ratio > SKIP_THRESHOLD.
        Caps skipping at 30% of waypoints.
        """
        if self.kd_sub is None:
            return None

        import networkx as nx
        tree, coords, nids = self.kd_sub
        SKIP_THRESHOLD = 3.0

        # Sample ideal line for waypoints (not too dense)
        # Phase 8: Use elastically deformed routing line for waypoint placement
        routing_src = self.routing_line if hasattr(self, 'routing_line') else self.ideal_line
        n_wps = min(25, max(8, len(self.il) // 2))
        wps = sample_polyline(routing_src, n_wps)
        wps_a = np.asarray(wps, dtype=np.float64)

        # Snap to nearest graph nodes
        _, indices = tree.query(wps_a)
        node_ids = [nids[i] for i in indices]
        wp_coords = [wps[i] for i in range(len(wps))]

        # Deduplicate consecutive
        deduped_nids = [node_ids[0]]
        deduped_coords = [wp_coords[0]]
        for i in range(1, len(node_ids)):
            if node_ids[i] != deduped_nids[-1]:
                deduped_nids.append(node_ids[i])
                deduped_coords.append(wp_coords[i])

        if len(deduped_nids) < 2:
            return None

        # ── Trunk-Killer: evaluate skip cost for each intermediate waypoint ──
        max_skips = max(1, int(len(deduped_nids) * 0.3))
        skip_set = set()

        for i in range(1, len(deduped_nids) - 1):
            if len(skip_set) >= max_skips:
                break
            prev_idx = i - 1
            # Find previous non-skipped
            while prev_idx in skip_set and prev_idx > 0:
                prev_idx -= 1
            next_idx = i + 1

            # Direct distance from prev to next (geographic)
            direct_dist = haversine(
                deduped_coords[prev_idx][0], deduped_coords[prev_idx][1],
                deduped_coords[next_idx][0], deduped_coords[next_idx][1]
            )
            if direct_dist < 10:
                continue

            # Detour distance via this waypoint
            try:
                path_a = nx.shortest_path(self.G_sub, deduped_nids[prev_idx],
                                          deduped_nids[i], weight='v8w')
                path_b = nx.shortest_path(self.G_sub, deduped_nids[i],
                                          deduped_nids[next_idx], weight='v8w')
                cost_via = sum(
                    self.G_sub[path_a[j]][path_a[j+1]][list(self.G_sub[path_a[j]][path_a[j+1]].keys())[0]].get('v8w', 50)
                    for j in range(len(path_a) - 1)
                ) + sum(
                    self.G_sub[path_b[j]][path_b[j+1]][list(self.G_sub[path_b[j]][path_b[j+1]].keys())[0]].get('v8w', 50)
                    for j in range(len(path_b) - 1)
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                skip_set.add(i)
                continue

            try:
                path_direct = nx.shortest_path(self.G_sub, deduped_nids[prev_idx],
                                               deduped_nids[next_idx], weight='v8w')
                cost_direct = sum(
                    self.G_sub[path_direct[j]][path_direct[j+1]][list(self.G_sub[path_direct[j]][path_direct[j+1]].keys())[0]].get('v8w', 50)
                    for j in range(len(path_direct) - 1)
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            ratio = cost_via / max(cost_direct, 1.0)
            if ratio > SKIP_THRESHOLD:
                skip_set.add(i)
                log(f"[v8.1-skip] Waypoint {i}: ratio={ratio:.1f} > {SKIP_THRESHOLD} — SKIPPED")

        # Build final waypoint list (without skipped)
        final_nids = [deduped_nids[i] for i in range(len(deduped_nids)) if i not in skip_set]
        if len(final_nids) < 2:
            final_nids = deduped_nids  # fallback to all

        if skip_set:
            log(f"[v8.1-skip] Skipped {len(skip_set)}/{len(deduped_nids)} waypoints")

        # Route segment by segment using bidirectional A*
        full = []
        for i in range(len(final_nids) - 1):
            path = bidirectional_astar(
                self.G_sub, final_nids[i], final_nids[i + 1],
                weight_attr='v8w', ideal_line=self.ideal_line
            )
            if path is None:
                try:
                    path = nx.shortest_path(self.G_sub, final_nids[i], final_nids[i + 1],
                                            weight='v8w')
                except Exception:
                    continue

            for nid in path:
                pt = [self.G_sub.nodes[nid]['y'], self.G_sub.nodes[nid]['x']]
                if not full or full[-1] != pt:
                    full.append(pt)

        # Close loop if shape is closed
        if self.is_closed and full and len(full) >= 3:
            if haversine(full[0][0], full[0][1], full[-1][0], full[-1][1]) > 30:
                # Route back to start
                start_node = final_nids[-1]
                end_node = final_nids[0]
                path = bidirectional_astar(
                    self.G_sub, start_node, end_node,
                    weight_attr='v8w', ideal_line=self.ideal_line
                )
                if path:
                    for nid in path[1:]:
                        pt = [self.G_sub.nodes[nid]['y'], self.G_sub.nodes[nid]['x']]
                        if not full or full[-1] != pt:
                            full.append(pt)

        return full if len(full) >= 2 else None

    def _route_anchors(self):
        """Fallback: anchor-point routing with v8 weights."""
        if self.kd_sub is None:
            return None

        import networkx as nx
        tree, coords, nids = self.kd_sub

        anchors = identify_anchor_points(self.ideal_line, angle_threshold=25)
        while len(anchors) < min(8, len(self.ideal_line)):
            new_thresh = 25 - (8 - len(anchors)) * 3
            if new_thresh < 5:
                break
            anchors = identify_anchor_points(self.ideal_line, angle_threshold=max(5, new_thresh))

        anchor_wps = [self.ideal_line[i] for i in anchors]
        wps_a = np.asarray(anchor_wps, dtype=np.float64)
        _, indices = tree.query(wps_a)
        node_ids = [nids[i] for i in indices]

        full = []
        for i in range(len(node_ids) - 1):
            if node_ids[i] == node_ids[i + 1]:
                continue
            try:
                path = nx.shortest_path(self.G_sub, node_ids[i], node_ids[i + 1],
                                        weight='v8w')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for nid in path:
                pt = [self.G_sub.nodes[nid]['y'], self.G_sub.nodes[nid]['x']]
                if not full or full[-1] != pt:
                    full.append(pt)

        return full if len(full) >= 2 else None

    def score(self, route):
        """Score a route using the v8 Fréchet-primary scorer."""
        if not route or len(route) < 2:
            return 1e9
        return score_v8(route, self.ideal_line)
