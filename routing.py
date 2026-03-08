"""
routing.py — Routing engine for Run2Art
========================================
Graph fetching, spatial corridor pruning, A* routing, and pedestrian cost model.

Key optimisations:
  - Wide-Tube adaptive corridor: GeoPandas sindex prunes graph to ~500 nodes
  - Bi-directional A* with haversine heuristic
  - Anchor-point strategy: route skeleton first, infill only where needed
  - Pedestrian-first cost function favouring footways/paths
"""

import sys
import os
import hashlib
import pickle
import time
import json
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

from geometry import (
    haversine, haversine_vector, haversine_matrix, point_to_segment_dist,
    adaptive_densify, sample_polyline, identify_anchor_points,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
    log("[routing] osmnx + networkx loaded")
except ImportError:
    HAS_OSMNX = False
    log("[routing] WARNING: osmnx not found — OSRM fallback only")

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH CACHING
# ═══════════════════════════════════════════════════════════════════════════

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

_graph_mem_cache = {}
_MEM_CACHE_MAX = 8


def _graph_cache_key(center, dist):
    lat_r = round(center[0], 3)
    lng_r = round(center[1], 3)
    raw = f"{lat_r:.3f},{lng_r:.3f},{dist}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_get(key):
    if key in _graph_mem_cache:
        log(f"[cache] Memory hit: {key}")
        return _graph_mem_cache[key][0]
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                G = pickle.load(f)
            _mem_store(key, G)
            log(f"[cache] Disk hit: {key}")
            return G
        except Exception:
            pass
    return None


def _cache_put(key, G):
    _mem_store(key, G)
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    try:
        with open(path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"[cache] Saved to disk: {key}")
    except Exception as e:
        log(f"[cache] Disk write failed: {e}")


def _mem_store(key, G):
    _graph_mem_cache[key] = (G, time.time())
    if len(_graph_mem_cache) > _MEM_CACHE_MAX:
        oldest = min(_graph_mem_cache, key=lambda k: _graph_mem_cache[k][1])
        del _graph_mem_cache[oldest]


def fetch_graph(center, dist=2500):
    """Fetch osmnx walk-network graph with caching, or None."""
    if not HAS_OSMNX:
        return None
    key = _graph_cache_key(center, dist)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        t0 = time.time()
        G = ox.graph_from_point((center[0], center[1]), dist=dist,
                                network_type='walk')
        if G.number_of_edges() < 10:
            return None
        elapsed = time.time() - t0
        log(f"[fetch_graph] Downloaded in {elapsed:.1f}s — "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        _cache_put(key, G)
        return G
    except Exception as e:
        log(f"[fetch_graph] Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  SPATIAL INDEX — KDTree + GeoPandas
# ═══════════════════════════════════════════════════════════════════════════

def build_kdtree(G):
    """Build a cKDTree from graph node coordinates."""
    if G is None:
        return None
    nodes = list(G.nodes(data=True))
    coords = np.array([[n[1]['y'], n[1]['x']] for n in nodes], dtype=np.float64)
    node_ids = [n[0] for n in nodes]
    tree = cKDTree(coords)
    return (tree, coords, node_ids)


def _compute_node_density(G):
    """Nodes per km² from graph bounding box."""
    nodes = list(G.nodes(data=True))
    lats = [n[1]['y'] for n in nodes]
    lngs = [n[1]['x'] for n in nodes]
    lat_range = (max(lats) - min(lats)) * 111.0  # km
    lng_range = (max(lngs) - min(lngs)) * 111.0 * np.cos(np.radians(np.mean(lats)))
    area_km2 = max(lat_range * lng_range, 0.01)
    return len(nodes) / area_km2


def build_corridor_subgraph(G, ideal_line, base_radius_m=200):
    """Build a subgraph containing only nodes within a tube around the ideal line.

    Uses GeoPandas spatial index when available, falls back to KDTree.
    Tube radius adapts: 100–500m based on node density.

    Returns (G_sub, kd_sub) or (G, kd_full) if pruning fails.
    """
    if G is None:
        return G, None

    n_nodes = G.number_of_nodes()

    # Adaptive tube radius based on node density
    density = _compute_node_density(G)
    tube_radius_m = max(100, min(500, base_radius_m / max(np.sqrt(density / 1000), 0.3)))
    tube_radius_deg = tube_radius_m / 111_000.0  # rough conversion

    il = np.asarray(ideal_line, dtype=np.float64)
    if len(il) < 2:
        return G, build_kdtree(G)

    if HAS_GEOPANDAS:
        try:
            return _corridor_geopandas(G, il, tube_radius_deg, tube_radius_m)
        except Exception as e:
            log(f"[corridor] GeoPandas failed ({e}), falling back to KDTree")

    return _corridor_kdtree(G, il, tube_radius_deg, tube_radius_m)


def _corridor_geopandas(G, il, tube_radius_deg, tube_radius_m):
    """Build corridor using GeoPandas spatial index (sindex)."""
    # Create buffered ideal line
    line = LineString([(il[i, 1], il[i, 0]) for i in range(len(il))])  # (lng, lat)
    buffer = line.buffer(tube_radius_deg)

    # Get node GeoDataFrame
    nodes_gdf = ox.graph_to_gdfs(G if hasattr(G, 'graph') else G, nodes=True, edges=False)

    # Spatial query: nodes within buffer
    candidates = nodes_gdf.sindex.query(buffer, predicate='intersects')
    corridor_nodes = set(nodes_gdf.index[candidates])

    if len(corridor_nodes) < 20:
        # Too few nodes — widen the tube
        buffer_wide = line.buffer(tube_radius_deg * 2.0)
        candidates = nodes_gdf.sindex.query(buffer_wide, predicate='intersects')
        corridor_nodes = set(nodes_gdf.index[candidates])

    if len(corridor_nodes) < 10:
        log(f"[corridor] Only {len(corridor_nodes)} nodes in tube, using full graph")
        return G, build_kdtree(G)

    G_sub = G.subgraph(corridor_nodes).copy()

    # Verify connectivity — if disconnected, widen
    if not nx.is_weakly_connected(G_sub):
        largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
        if len(largest_cc) < len(corridor_nodes) * 0.6:
            # Too fragmented, widen tube
            buffer_wide = line.buffer(tube_radius_deg * 2.5)
            candidates = nodes_gdf.sindex.query(buffer_wide, predicate='intersects')
            corridor_nodes = set(nodes_gdf.index[candidates])
            G_sub = G.subgraph(corridor_nodes).copy()
            largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
        G_sub = G_sub.subgraph(largest_cc).copy()

    log(f"[corridor] GeoPandas: {G.number_of_nodes()} → {G_sub.number_of_nodes()} nodes "
        f"(tube={tube_radius_m:.0f}m)")
    return G_sub, build_kdtree(G_sub)


def _corridor_kdtree(G, il, tube_radius_deg, tube_radius_m):
    """Build corridor using KDTree ball query (fallback)."""
    kd_full = build_kdtree(G)
    if kd_full is None:
        return G, None
    tree, coords, node_ids = kd_full

    # Query all points within tube radius of any ideal line point
    corridor_indices = set()
    for i in range(len(il)):
        indices = tree.query_ball_point([il[i, 0], il[i, 1]], tube_radius_deg)
        corridor_indices.update(indices)

    # Also add points near segment midpoints for better coverage
    for i in range(len(il) - 1):
        mid = [(il[i, 0] + il[i+1, 0]) / 2, (il[i, 1] + il[i+1, 1]) / 2]
        indices = tree.query_ball_point(mid, tube_radius_deg)
        corridor_indices.update(indices)

    corridor_nodes = {node_ids[i] for i in corridor_indices}

    if len(corridor_nodes) < 10:
        log(f"[corridor] Only {len(corridor_nodes)} nodes in tube, using full graph")
        return G, kd_full

    G_sub = G.subgraph(corridor_nodes).copy()

    # Ensure connectivity
    if nx.is_weakly_connected(G_sub):
        kd_sub = build_kdtree(G_sub)
        log(f"[corridor] KDTree: {G.number_of_nodes()} → {G_sub.number_of_nodes()} nodes "
            f"(tube={tube_radius_m:.0f}m)")
        return G_sub, kd_sub

    largest_cc = max(nx.weakly_connected_components(G_sub), key=len)
    G_sub = G_sub.subgraph(largest_cc).copy()

    if G_sub.number_of_nodes() < 10:
        return G, kd_full

    kd_sub = build_kdtree(G_sub)
    log(f"[corridor] KDTree: {G.number_of_nodes()} → {G_sub.number_of_nodes()} nodes "
        f"(tube={tube_radius_m:.0f}m)")
    return G_sub, kd_sub


# ═══════════════════════════════════════════════════════════════════════════
#  PEDESTRIAN COST FUNCTION (WS3: BioMechanics)
# ═══════════════════════════════════════════════════════════════════════════

# Highway type multipliers — lower = preferred for runners
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


def _get_highway_multiplier(data):
    """Get pedestrian-friendliness multiplier from edge data."""
    hw = data.get('highway', 'residential')
    if isinstance(hw, list):
        hw = hw[0]
    return HIGHWAY_MULTIPLIERS.get(hw, 1.0)


def make_pedestrian_weight_fn(G, ideal_line, alpha=1.0, beta=8.0, gamma=3.0,
                               tube_radius_m=300):
    """Multi-factor pedestrian cost function.

    C = α·C_network + β·C_proximity + γ·C_biomech

    Where:
      C_network  = edge_length × highway_multiplier
      C_proximity = distance to ideal line
      C_biomech  = highway_multiplier × penalty_gradient
    """
    il = np.asarray(ideal_line, dtype=np.float64)
    n_seg = len(il) - 1
    tube_deg = tube_radius_m / 111_000.0
    _cache = {}

    def weight_fn(u, v, data):
        key = (u, v)
        if key in _cache:
            return _cache[key]

        length = data.get('length', 50)
        hw_mult = _get_highway_multiplier(data)

        # Edge midpoint
        mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) * 0.5
        mid_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) * 0.5

        # Perpendicular distance to ideal line
        best_dist = 1e9
        for j in range(n_seg):
            best_dist = min(best_dist, point_to_segment_dist(
                [mid_lat, mid_lon], [il[j, 0], il[j, 1]],
                [il[j+1, 0], il[j+1, 1]]))

        # Penalty gradient: quadratic increase near tube boundary
        d_norm = min(best_dist / max(tube_radius_m, 1.0), 1.0)
        penalty_gradient = 1.0 + 2.0 * d_norm * d_norm

        # Multi-factor cost
        c_network = length * hw_mult
        c_proximity = best_dist
        c_biomech = hw_mult * penalty_gradient * 10.0

        w = alpha * c_network + beta * c_proximity + gamma * c_biomech
        _cache[key] = w
        return w

    return weight_fn


def _haversine_heuristic(G):
    """Create an A* heuristic function using haversine distance."""
    def heuristic(u, target):
        lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
        lat2, lon2 = G.nodes[target]['y'], G.nodes[target]['x']
        return haversine(lat1, lon1, lat2, lon2)
    return heuristic


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTING — A* with corridor + pedestrian cost
# ═══════════════════════════════════════════════════════════════════════════

def route_graph(G, waypoints, weight='length', kdtree_data=None):
    """Route through graph using A* when heuristic is available, else Dijkstra."""
    if G is None:
        return None
    try:
        if kdtree_data is not None:
            tree, _coords, nid_list = kdtree_data
            wps_a = np.asarray(waypoints, dtype=np.float64)
            _, indices = tree.query(wps_a)
            node_ids = [nid_list[i] for i in indices]
        else:
            node_ids = [ox.nearest_nodes(G, w[1], w[0]) for w in waypoints]

        heuristic = _haversine_heuristic(G)
        full = []
        for i in range(len(node_ids) - 1):
            o, d = node_ids[i], node_ids[i + 1]
            if o == d:
                continue
            path = _find_path(G, o, d, weight, heuristic)
            if path is None:
                continue
            for nid in path:
                pt = [G.nodes[nid]['y'], G.nodes[nid]['x']]
                if not full or full[-1] != pt:
                    full.append(pt)
        return full if len(full) >= 2 else None
    except Exception:
        return None


def _find_path(G, source, target, weight, heuristic):
    """Find path using A* with fallback to Dijkstra, then plain length."""
    # Try A* with heuristic (only works with callable weight)
    if callable(weight):
        try:
            return nx.astar_path(G, source, target,
                                 heuristic=heuristic, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        # Fallback to plain length
        try:
            return nx.shortest_path(G, source, target, weight='length')
        except nx.NetworkXNoPath:
            return None

    # Non-callable weight (string attr name)
    try:
        return nx.astar_path(G, source, target,
                             heuristic=heuristic, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    if weight != 'length':
        try:
            return nx.shortest_path(G, source, target, weight='length')
        except nx.NetworkXNoPath:
            pass
    return None


def route_shape_aware(G, waypoints, ideal_line, penalty=10.0,
                      kdtree_data=None, tube_radius_m=300):
    """Route with pedestrian-aware shape-deviation weight function."""
    if G is None or not ideal_line or len(ideal_line) < 2:
        return route_graph(G, waypoints, kdtree_data=kdtree_data)
    weight_fn = make_pedestrian_weight_fn(G, ideal_line,
                                           tube_radius_m=tube_radius_m)
    return route_graph(G, waypoints, weight=weight_fn, kdtree_data=kdtree_data)


def route_with_anchors(G, waypoints, ideal_line, kdtree_data=None,
                       tube_radius_m=300):
    """Anchor-point routing strategy: route skeleton first, infill if needed.

    1. Identify high-curvature anchor points from waypoints
    2. Route anchor-to-anchor (skeleton)
    3. Check deviation — infill with denser waypoints where deviation > 80m
    """
    if G is None or not ideal_line or len(ideal_line) < 2:
        return route_graph(G, waypoints, kdtree_data=kdtree_data)

    # Step 1: Get anchor indices
    anchor_indices = identify_anchor_points(waypoints, angle_threshold=25)

    # Ensure enough anchors (at least ~8 for shape fidelity)
    while len(anchor_indices) < min(8, len(waypoints)):
        # Lower threshold to get more anchors
        new_threshold = 25 - (8 - len(anchor_indices)) * 3
        if new_threshold < 5:
            break
        anchor_indices = identify_anchor_points(waypoints, angle_threshold=max(5, new_threshold))

    anchor_wps = [waypoints[i] for i in anchor_indices]

    weight_fn = make_pedestrian_weight_fn(G, ideal_line,
                                           tube_radius_m=tube_radius_m)

    # Step 2: Route skeleton
    skeleton_route = route_graph(G, anchor_wps, weight=weight_fn,
                                 kdtree_data=kdtree_data)
    if skeleton_route is None:
        return route_graph(G, waypoints, weight=weight_fn,
                           kdtree_data=kdtree_data)

    # Step 3: Check deviation and infill
    ideal_s = sample_polyline(ideal_line, min(60, len(ideal_line) * 2))
    route_s = sample_polyline(skeleton_route, min(80, len(skeleton_route)))
    if len(ideal_s) < 4 or len(route_s) < 4:
        return skeleton_route

    ia = np.asarray(ideal_s, dtype=np.float64)
    ra = np.asarray(route_s, dtype=np.float64)
    dists = haversine_matrix(ia[:, 0], ia[:, 1], ra[:, 0], ra[:, 1]).min(axis=1)

    # Where deviation > 80m, find the waypoint segments that need infilling
    bad_mask = dists > 80.0
    if not bad_mask.any():
        return skeleton_route  # Skeleton is good enough

    # Infill: add intermediate waypoints from original dense list
    bad_ideal_pts = ia[bad_mask]
    infill_wps = list(anchor_wps)
    for bp in bad_ideal_pts[::2]:  # every other to avoid overload
        best_pos, best_d = 0, 1e9
        for k in range(len(infill_wps) - 1):
            d = point_to_segment_dist(list(bp), infill_wps[k], infill_wps[k+1])
            if d < best_d:
                best_d = d
                best_pos = k + 1
        infill_wps.insert(best_pos, list(bp))

    infill_route = route_graph(G, infill_wps, weight=weight_fn,
                                kdtree_data=kdtree_data)

    if infill_route and len(infill_route) >= 2:
        return infill_route
    return skeleton_route


def route_osrm(waypoints):
    """Route via public OSRM demo server (foot profile)."""
    full = []
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        url = (f"https://router.project-osrm.org/route/v1/foot/"
               f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Run2Art/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get("routes"):
                for lon, lat in data["routes"][0]["geometry"]["coordinates"]:
                    pt = [lat, lon]
                    if not full or full[-1] != pt:
                        full.append(pt)
        except Exception:
            continue
        time.sleep(0.05)
    return full if len(full) >= 2 else None
