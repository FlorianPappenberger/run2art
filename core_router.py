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
    penalty[mask_high] = 100.0 + (dev[mask_high] - 45.0) ** 2 * 0.32

    mask_perp = dev >= 70.0
    penalty[mask_perp] = 300.0

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
            # Widen everything
            corridor_shape = corridor_shape.buffer(max_radius_m / 111_000.0)
            candidates = nodes_gdf.sindex.query(corridor_shape, predicate='intersects')
            corridor_nodes = set(nodes_gdf.index[candidates])

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

def precompute_edge_weights(G_sub, ideal_line, tangent_field, apex_points,
                            w_net=1.0, w_prox=12.0, w_head=8.0,
                            w_uturn=1.0, w_bio=2.0,
                            tube_radius_m=300):
    """Precompute all edge weights in one vectorized pass.

    Assigns attribute 'v8w' to each edge in G_sub.
    A* / Dijkstra can then use weight='v8w' (string attribute) —
    no Python callback per-edge.

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

    # ── C_uturn: penalty for edges that reverse direction near non-apex areas ──
    near_apex = compute_uturn_mask(G_sub, midpoints, apex_points, apex_radius_m=15.0)
    # Detect near-180° turns by checking if this edge doubles back
    # We approximate: edges longer than 5m that point opposite to local tangent
    reverse_dev = np.abs(edge_bearings - local_tangents) % 360.0
    reverse_dev = np.where(reverse_dev > 180.0, 360.0 - reverse_dev, reverse_dev)
    is_reversal = reverse_dev > 150.0  # nearly opposite direction
    c_uturn = np.where(is_reversal & ~near_apex, 2000.0, 0.0)

    # ── C_biomech: surface preference gradient ──
    d_norm = np.clip(c_proximity / max(tube_radius_m, 1.0), 0.0, 1.0)
    c_biomech = hw_mults * (1.0 + 2.0 * d_norm ** 2) * 10.0

    # ── Composite weight ──
    weights = (w_net * c_network +
               w_prox * c_proximity +
               w_head * c_heading +
               w_uturn * c_uturn +
               w_bio * c_biomech)

    # Assign to graph edges
    for k, (u, v, data) in enumerate(edges):
        G_sub[u][v][list(G_sub[u][v].keys())[0]]['v8w'] = float(weights[k])

    log(f"[v8-weights] Precomputed {n_edges} edge weights "
        f"(prox={c_proximity.mean():.0f}m, head={c_heading.mean():.0f}m, "
        f"uturn={int(c_uturn.sum() > 0)} reversals)")

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
        self.is_closed = self._check_closed()

        # Build adaptive corridor
        self.G_sub, self.kd_sub = build_adaptive_corridor(
            G, ideal_line,
            base_radius_m=self.config.get('base_radius_m', 250),
            min_radius_m=self.config.get('min_radius_m', 150),
            max_radius_m=self.config.get('max_radius_m', 500),
        )

        # Precompute edge weights
        if self.G_sub is not None and self.G_sub.number_of_edges() > 0:
            precompute_edge_weights(
                self.G_sub, ideal_line, self.tangent_field, self.apex_points,
                w_net=self.config.get('w_net', 1.0),
                w_prox=self.config.get('w_prox', 12.0),
                w_head=self.config.get('w_head', 8.0),
                w_uturn=self.config.get('w_uturn', 1.0),
                w_bio=self.config.get('w_bio', 2.0),
                tube_radius_m=self.config.get('tube_radius_m', 300),
            )

        log(f"[CoreRouter] apexes={len(self.apex_points)}, "
            f"closed={self.is_closed}, "
            f"corridor={self.G_sub.number_of_nodes() if self.G_sub else 0} nodes")

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
        if route:
            return route

        # Strategy 3: Fallback to anchor-point routing with v8 weights
        return self._route_anchors()

    def _route_segments(self):
        """Route by connecting sampled waypoints using bidirectional A*."""
        if self.kd_sub is None:
            return None

        tree, coords, nids = self.kd_sub

        # Sample ideal line for waypoints (not too dense)
        n_wps = min(25, max(8, len(self.il) // 2))
        wps = sample_polyline(self.ideal_line, n_wps)
        wps_a = np.asarray(wps, dtype=np.float64)

        # Snap to nearest graph nodes
        _, indices = tree.query(wps_a)
        node_ids = [nids[i] for i in indices]

        # Deduplicate consecutive
        deduped = [node_ids[0]]
        for nid in node_ids[1:]:
            if nid != deduped[-1]:
                deduped.append(nid)

        if len(deduped) < 2:
            return None

        # Route segment by segment using bidirectional A*
        full = []
        for i in range(len(deduped) - 1):
            path = bidirectional_astar(
                self.G_sub, deduped[i], deduped[i + 1],
                weight_attr='v8w', ideal_line=self.ideal_line
            )
            if path is None:
                # Fallback to networkx shortest path
                try:
                    import networkx as nx
                    path = nx.shortest_path(self.G_sub, deduped[i], deduped[i + 1],
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
                start_node = deduped[0]
                end_node = deduped[-1]
                path = bidirectional_astar(
                    self.G_sub, end_node, start_node,
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
