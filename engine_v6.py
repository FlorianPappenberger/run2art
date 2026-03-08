"""
engine_v6.py — Run2Art Geospatial Engine v6.0 (Shape-Aware Refactor)
=====================================================================
Major improvements over v5.1:
  - Segment-constrained routing with adaptive "tube" search spaces
  - Multi-objective edge weighting with turning-angle penalties
  - Curvature-based adaptive densification (proportional to local radius)
  - 6-component cost integration directly into Dijkstra (not just post-scoring)

Based on:
  - Waschk & Krüger (2018) — Shape-aware routing for GPS art
  - Li & Fu (2026) — Multi-objective route optimization
  - dsleo/stravart (2024) — Bidirectional scoring metrics

Modes: "fit", "optimize", "best_shape"
"""

import sys
import json
import math
import os
import hashlib
import pickle
import urllib.request
import time

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    """Log to stderr (visible in server console, not in stdout JSON)."""
    print(msg, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
    log("[engine_v6] osmnx + networkx loaded — shape-aware routing enabled")
except ImportError:
    HAS_OSMNX = False
    log("[engine_v6] WARNING: osmnx not found — OSRM fallback only")


# ---------------------------------------------------------------------------
# Graph caching (disk + in-memory LRU)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

_graph_mem_cache = {}  # key → (graph, timestamp)
_MEM_CACHE_MAX = 8     # keep at most 8 graphs in memory


def _graph_cache_key(center, dist):
    """Deterministic cache key for a (center, dist) pair, rounded to ~200m grid."""
    lat_r = round(center[0], 3)   # ~111m resolution
    lng_r = round(center[1], 3)   # ~80m at mid-latitudes
    raw = f"{lat_r:.3f},{lng_r:.3f},{dist}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_get(key):
    """Try memory cache, then disk cache. Returns graph or None."""
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
    """Store graph in memory + disk."""
    _mem_store(key, G)
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    try:
        with open(path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"[cache] Saved to disk: {key}")
    except Exception as e:
        log(f"[cache] Disk write failed: {e}")


def _mem_store(key, G):
    """Store in memory LRU, evict oldest if full."""
    _graph_mem_cache[key] = (G, time.time())
    if len(_graph_mem_cache) > _MEM_CACHE_MAX:
        oldest = min(_graph_mem_cache, key=lambda k: _graph_mem_cache[k][1])
        del _graph_mem_cache[oldest]


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """Distance in metres between two geographic points (scalar)."""
    R = 6_371_000
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) *
         math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(min(1, math.sqrt(a)))


def haversine_matrix(lats1, lons1, lats2, lons2):
    """Vectorised pairwise haversine. Returns (M, N) distance matrix in metres."""
    R = 6_371_000.0
    p = np.pi / 180.0
    lat1 = np.asarray(lats1, dtype=np.float64)[:, None] * p
    lon1 = np.asarray(lons1, dtype=np.float64)[:, None] * p
    lat2 = np.asarray(lats2, dtype=np.float64)[None, :] * p
    lon2 = np.asarray(lons2, dtype=np.float64)[None, :] * p
    dlat = (lat2 - lat1) / 2.0
    dlon = (lon2 - lon1) / 2.0
    a = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
    return 2.0 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def haversine_vector(lats1, lons1, lats2, lons2):
    """Vectorised element-wise haversine. All arrays same length → 1-D result."""
    R = 6_371_000.0
    p = np.pi / 180.0
    lat1 = np.asarray(lats1, dtype=np.float64) * p
    lon1 = np.asarray(lons1, dtype=np.float64) * p
    lat2 = np.asarray(lats2, dtype=np.float64) * p
    lon2 = np.asarray(lons2, dtype=np.float64) * p
    dlat = (lat2 - lat1) / 2.0
    dlon = (lon2 - lon1) / 2.0
    a = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
    return 2.0 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def rotate_shape(pts, angle_deg):
    """Rotate normalised [0,1] points around (0.5, 0.5)."""
    cx, cy = 0.5, 0.5
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [[cx + (x - cx) * cos_a - (y - cy) * sin_a,
             cy + (x - cx) * sin_a + (y - cy) * cos_a] for x, y in pts]


def shape_to_latlngs(pts, center, scale_deg, rotation_deg=0):
    """Convert normalised shape → geographic lat/lng coordinates."""
    if rotation_deg:
        pts = rotate_shape(pts, rotation_deg)
    cx, cy = 0.5, 0.5
    return [[center[0] - (y - cy) * scale_deg,
             center[1] + (x - cx) * scale_deg * 1.4] for x, y in pts]


def sample_polyline(pts, n):
    """Evenly sample *n* points along a lat/lng polyline (NumPy-accelerated)."""
    if len(pts) < 2:
        return list(pts)
    pts_a = np.asarray(pts, dtype=np.float64)
    seg_dists = haversine_vector(pts_a[:-1, 0], pts_a[:-1, 1],
                                 pts_a[1:, 0], pts_a[1:, 1])
    cum = np.empty(len(pts_a), dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(seg_dists, out=cum[1:])
    total = cum[-1]
    if total < 1:
        return [list(p) for p in pts]
    targets = np.linspace(0.0, total, n)
    seg_idx = np.searchsorted(cum, targets, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, len(pts_a) - 2)
    seg_len = cum[seg_idx + 1] - cum[seg_idx]
    safe_len = np.where(seg_len < 1e-9, 1.0, seg_len)
    t = np.where(seg_len < 1e-9, 0.0, (targets - cum[seg_idx]) / safe_len)
    lats = pts_a[seg_idx, 0] + t * (pts_a[seg_idx + 1, 0] - pts_a[seg_idx, 0])
    lngs = pts_a[seg_idx, 1] + t * (pts_a[seg_idx + 1, 1] - pts_a[seg_idx, 1])
    return [[lats[i], lngs[i]] for i in range(n)]


def turning_angle(a, b, c):
    """Signed turning angle in degrees at point b for path a→b→c."""
    d1 = math.atan2(b[0] - a[0], b[1] - a[1])
    d2 = math.atan2(c[0] - b[0], c[1] - b[1])
    diff = math.degrees(d2 - d1)
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def compute_curvature(a, b, c):
    """Compute unsigned curvature at point b (in inverse metres).
    Returns 1/radius approximation based on turning angle and distance."""
    angle = abs(turning_angle(a, b, c))
    if angle < 1.0:
        return 0.0
    dist = (haversine(a[0], a[1], b[0], b[1]) + 
            haversine(b[0], b[1], c[0], c[1])) / 2.0
    if dist < 1.0:
        return 0.0
    # Curvature ≈ Δθ / arc_length (in radians/meter)
    return math.radians(angle) / dist


def point_to_segment_dist(p, a, b):
    """Perpendicular distance (m) from point p to segment a→b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg_sq = dx * dx + dy * dy
    if seg_sq < 1e-14:
        return haversine(p[0], p[1], a[0], a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg_sq))
    proj = [a[0] + t * dx, a[1] + t * dy]
    return haversine(p[0], p[1], proj[0], proj[1])


def min_dist_to_polyline(p, polyline):
    """Minimum distance (m) from point p to any segment of a polyline."""
    return min(point_to_segment_dist(p, polyline[j], polyline[j + 1])
               for j in range(len(polyline) - 1))


def bearing(a, b):
    """Compute bearing in degrees from point a to point b."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


# ═══════════════════════════════════════════════════════════════════════════
#  CURVATURE-BASED ADAPTIVE DENSIFICATION (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_densify_v6(waypoints, k0=2.5, min_spacing=20, max_spacing=120):
    """
    Curvature-based adaptive densification.
    
    Formula: spacing(i) = max(min_spacing, min(max_spacing, k0 / (|κ_i| + ε)))
    
    Where κ_i is local curvature at waypoint i (computed from 3-point turning angle).
    
    Parameters:
        k0: Calibration constant (higher = denser on curves)
        min_spacing: Minimum spacing in meters (for tight curves)
        max_spacing: Maximum spacing in meters (for straight segments)
    
    Returns: Densified waypoint list with curvature-adaptive spacing.
    """
    if len(waypoints) < 3:
        return _uniform_densify(waypoints, spacing_m=60)
    
    n = len(waypoints)
    dense = [waypoints[0]]
    
    for i in range(n - 1):
        a = waypoints[max(0, i - 1)]
        b = waypoints[i]
        c = waypoints[min(n - 1, i + 1)]
        d = waypoints[min(n - 1, i + 2)]
        
        # Compute local curvature at b and c
        curv_b = compute_curvature(a, b, c)
        curv_c = compute_curvature(b, c, d)
        avg_curv = (curv_b + curv_c) / 2.0
        
        # Spacing inversely proportional to curvature
        epsilon = 0.01
        spacing = k0 / (avg_curv + epsilon)
        spacing = max(min_spacing, min(max_spacing, spacing))
        
        # Insert points
        dist = haversine(b[0], b[1], waypoints[i + 1][0], waypoints[i + 1][1])
        n_seg = max(1, int(round(dist / spacing)))
        
        for j in range(1, n_seg + 1):
            t = j / n_seg
            dense.append([b[0] + t * (waypoints[i + 1][0] - b[0]),
                          b[1] + t * (waypoints[i + 1][1] - b[1])])
    
    return dense


def _uniform_densify(waypoints, spacing_m=60):
    """Fallback uniform densification."""
    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        dist = haversine(a[0], a[1], b[0], b[1])
        n_seg = max(1, int(round(dist / spacing_m)))
        for j in range(1, n_seg + 1):
            t = j / n_seg
            dense.append([a[0] + t * (b[0] - a[0]),
                          a[1] + t * (b[1] - a[1])])
    return dense


# ═══════════════════════════════════════════════════════════════════════════
#  SCORING (NumPy-vectorised — unchanged from v5.1)
# ═══════════════════════════════════════════════════════════════════════════

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

    # 4. Perpendicular segment distance
    step = max(1, n_r // 60)
    perp_pts = route_s[::step]
    perp = (sum(min_dist_to_polyline(rp, ideal_s) for rp in perp_pts)
            / len(perp_pts)) if len(ideal_s) >= 2 else 0.0

    # 5. Turning-angle fidelity
    angle_penalty = 0.0
    if len(ideal_pts) >= 3 and len(route) >= 3:
        rk = sample_polyline(route, len(ideal_pts))
        diffs = [abs(turning_angle(ideal_pts[k-1], ideal_pts[k], ideal_pts[k+1]) -
                     turning_angle(rk[k-1], rk[k], rk[k+1]))
                 for k in range(1, min(len(ideal_pts), len(rk)) - 1)]
        if diffs:
            angle_penalty = (sum(diffs) / len(diffs)) * 1.5

    # 6. Length-ratio fidelity
    lr_penalty = _length_ratio_penalty(
        sample_polyline(route, min(20, len(route))),
        sample_polyline(ideal_pts, min(20, len(ideal_pts))),
    )

    score = (fwd_avg * 0.30 + rev_avg * 0.15 + haus * 0.10 +
             perp * 0.20 + angle_penalty * 0.15 + lr_penalty * 0.10)
    if coverage < 0.75:
        score *= (2.0 - coverage)
    return score


def _length_ratio_penalty(route_pts, ideal_pts):
    """Compare consecutive-segment length ratios between route and ideal."""
    def seg_lens(pts):
        return [haversine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                for i in range(len(pts) - 1)]
    rl, il = seg_lens(route_pts), seg_lens(ideal_pts)
    n = min(len(rl), len(il), 20)
    if n < 2:
        return 0.0
    def pick(lst, cnt):
        if len(lst) <= cnt:
            return lst
        s = len(lst) / cnt
        return [lst[int(i * s)] for i in range(cnt)]
    rs, iss = pick(rl, n), pick(il, n)
    diffs = [abs(rs[k+1] / max(rs[k], 1.0) - iss[k+1] / max(iss[k], 1.0))
             for k in range(len(rs) - 1)]
    return (sum(diffs) / len(diffs)) * 15.0 if diffs else 0.0


def coarse_proximity_score(G, waypoints, kdtree_data=None):
    """Fast coarse score — proximity of waypoints to nearest road nodes.
    Uses KDTree when available for O(log n) lookup."""
    if G is None:
        return 1e9
    try:
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
        return float(dists.mean() + dists.max() * 0.3)
    except Exception:
        return 1e9


# ═══════════════════════════════════════════════════════════════════════════
#  SEGMENT-CONSTRAINED ROUTING (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def build_segment_map(ideal_line, n_segments=None):
    """
    Build a mapping from ideal-line points to segment indices.
    
    Returns: (segments, lookup_tree)
        segments: List of [(lat1, lng1), (lat2, lng2), seg_idx, bearing, length]
        lookup_tree: cKDTree for fast segment lookup
    """
    if len(ideal_line) < 2:
        return [], None
    
    if n_segments is None:
        n_segments = len(ideal_line) - 1
    
    segments = []
    segment_centers = []
    
    for i in range(len(ideal_line) - 1):
        a, b = ideal_line[i], ideal_line[i + 1]
        mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
        bear = bearing(a, b)
        length = haversine(a[0], a[1], b[0], b[1])
        segments.append((a, b, i, bear, length))
        segment_centers.append(mid)
    
    if not segment_centers:
        return segments, None
    
    tree = cKDTree(np.array(segment_centers, dtype=np.float64))
    return segments, tree


def find_nearest_segment(point, lookup_tree, segments):
    """Find the nearest ideal-line segment to a given point."""
    if lookup_tree is None or not segments:
        return None
    dist, idx = lookup_tree.query([point[0], point[1]])
    if idx >= len(segments):
        return None
    return segments[idx]


def make_multi_objective_weight_fn(G, ideal_line, segments, lookup_tree,
                                   alpha=15.0, beta=0.8, tube_radius=80.0):
    """
    Create a multi-objective edge weight function for Dijkstra routing.
    
    Cost = length + α·D_perp + β·Δθ + γ·T(tube)
    
    Where:
        - D_perp: Perpendicular distance to nearest ideal segment
        - Δθ: Angular deviation from expected bearing
        - T(tube): Hard constraint (∞ penalty if outside tube)
    
    Parameters:
        alpha: Shape deviation penalty weight (default: 15.0)
        beta: Turning penalty weight (default: 0.8)
        tube_radius: Maximum allowed distance from ideal (meters)
    """
    il = np.asarray(ideal_line, dtype=np.float64)
    n_seg = len(il) - 1
    _cache = {}

    def weight_fn(u, v, data):
        key = (u, v)
        if key in _cache:
            return _cache[key]
        
        length = data.get('length', 50)
        mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) * 0.5
        mid_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) * 0.5
        edge_bear = bearing([G.nodes[u]['y'], G.nodes[u]['x']],
                           [G.nodes[v]['y'], G.nodes[v]['x']])
        
        # Find perpendicular distance to nearest ideal segment
        best_perp = 1e9
        best_seg = None
        for j in range(n_seg):
            d = point_to_segment_dist([mid_lat, mid_lon],
                                     [il[j, 0], il[j, 1]],
                                     [il[j+1, 0], il[j+1, 1]])
            if d < best_perp:
                best_perp = d
                best_seg = j
        
        # Tube constraint: hard reject if too far from ideal
        if best_perp > tube_radius:
            w = 1e9  # Effectively infinite cost
            _cache[key] = w
            return w
        
        # Angular deviation penalty
        if best_seg is not None:
            ideal_bear = bearing([il[best_seg, 0], il[best_seg, 1]],
                                [il[best_seg + 1, 0], il[best_seg + 1, 1]])
            angle_diff = abs(edge_bear - ideal_bear)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            # Convert to cost (0° = 0m, 90° = 100m, 180° = 200m)
            angle_cost = (angle_diff / 180.0) * 200.0 * beta
        else:
            angle_cost = 0.0
        
        # Total cost
        w = length + alpha * best_perp + angle_cost
        _cache[key] = w
        return w

    return weight_fn


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTING (Enhanced with segment constraints)
# ═══════════════════════════════════════════════════════════════════════════

def build_kdtree(G):
    """Build a cKDTree from graph node coordinates for fast nearest-node lookup."""
    if G is None:
        return None
    nodes = list(G.nodes(data=True))
    coords = np.array([[n[1]['y'], n[1]['x']] for n in nodes], dtype=np.float64)
    node_ids = [n[0] for n in nodes]
    tree = cKDTree(coords)
    return (tree, coords, node_ids)


def route_graph(G, waypoints, weight='length', kdtree_data=None):
    """Route through osmnx graph. Uses KDTree for nearest-node when available."""
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
        
        full = []
        for i in range(len(node_ids) - 1):
            o, d = node_ids[i], node_ids[i + 1]
            if o == d:
                continue
            try:
                path = nx.shortest_path(G, o, d, weight=weight)
            except nx.NetworkXNoPath:
                # Fallback to unweighted if no path found with custom weight
                if weight != 'length' and not callable(weight):
                    try:
                        path = nx.shortest_path(G, o, d, weight='length')
                    except nx.NetworkXNoPath:
                        continue
                elif callable(weight):
                    try:
                        path = nx.shortest_path(G, o, d, weight='length')
                    except nx.NetworkXNoPath:
                        continue
                else:
                    continue
            
            for nid in path:
                pt = [G.nodes[nid]['y'], G.nodes[nid]['x']]
                if not full or full[-1] != pt:
                    full.append(pt)
        
        return full if len(full) >= 2 else None
    except Exception:
        return None


def route_segment_constrained(G, waypoints, ideal_line, kdtree_data=None,
                              alpha=15.0, beta=0.8, adaptive_tube=True):
    """
    Route with segment-constrained multi-objective weighting.
    
    Parameters:
        alpha: Shape deviation penalty (default: 15.0)
        beta: Turning penalty (default: 0.8)
        adaptive_tube: Use curvature-based tube radius (default: True)
    """
    if G is None or not ideal_line or len(ideal_line) < 2:
        return route_graph(G, waypoints, kdtree_data=kdtree_data)
    
    # Build segment map for fast lookup
    segments, seg_tree = build_segment_map(ideal_line)
    
    # Compute adaptive tube radius based on average spacing
    if adaptive_tube and len(ideal_line) >= 2:
        spacings = [haversine(ideal_line[i][0], ideal_line[i][1],
                             ideal_line[i+1][0], ideal_line[i+1][1])
                   for i in range(len(ideal_line) - 1)]
        avg_spacing = sum(spacings) / len(spacings)
        tube_radius = max(80.0, 1.5 * avg_spacing)
    else:
        tube_radius = 100.0
    
    log(f"[route_segment_constrained] Tube radius: {tube_radius:.0f}m, "
        f"α={alpha}, β={beta}")
    
    # Create multi-objective weight function
    weight_fn = make_multi_objective_weight_fn(
        G, ideal_line, segments, seg_tree,
        alpha=alpha, beta=beta, tube_radius=tube_radius
    )
    
    return route_graph(G, waypoints, weight=weight_fn, kdtree_data=kdtree_data)


def route_osrm(waypoints):
    """Route via public OSRM demo server (foot profile)."""
    full = []
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        url = (f"https://router.project-osrm.org/route/v1/foot/"
               f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Run2Art/6.0"})
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
#  ROUTE + SCORE HELPER (Enhanced with v6 features)
# ═══════════════════════════════════════════════════════════════════════════

def fit_and_score_v6(G, pts, rot, scale, center, kdtree_data=None,
                     refine=False, use_constraints=True):
    """
    v6 enhanced fit-and-score with curvature-based densification and
    segment-constrained routing.
    
    Parameters:
        use_constraints: Use segment-constrained routing (default: True)
        refine: Apply iterative refinement pass (default: False)
    """
    wps = shape_to_latlngs(pts, center, scale, rot)
    
    # v6: Curvature-based adaptive densification
    dense = adaptive_densify_v6(wps, k0=2.5, min_spacing=20, max_spacing=120)
    ideal = adaptive_densify_v6(wps, k0=3.0, min_spacing=15, max_spacing=80)
    
    route = None
    if G:
        if use_constraints:
            # v6: Segment-constrained routing with multi-objective weights
            route = route_segment_constrained(G, dense, ideal,
                                             kdtree_data=kdtree_data,
                                             alpha=15.0, beta=0.8,
                                             adaptive_tube=True)
        else:
            # Fallback to standard routing
            route = route_graph(G, dense, kdtree_data=kdtree_data)
        
        if not route:
            route = route_graph(G, dense, kdtree_data=kdtree_data)
    
    if not route:
        route = route_osrm(dense)
    
    if not route:
        return (1e9, None)
    
    # Refinement pass with segment constraints
    if refine and G and route and len(ideal) >= 4:
        route = _refine_route_v6(G, route, ideal, dense, kdtree_data)
    
    return (bidirectional_score(route, ideal), route)


def _refine_route_v6(G, route, ideal, original_wps, kdtree_data):
    """
    v6 refinement: Insert corrective waypoints at high-deviation locations
    and re-route with tighter constraints.
    """
    ideal_s = sample_polyline(ideal, min(60, len(ideal) * 2))
    route_s = sample_polyline(route, min(80, len(route)))
    if len(ideal_s) < 4 or len(route_s) < 4:
        return route
    
    # Identify high-deviation points
    ia = np.asarray(ideal_s, dtype=np.float64)
    ra = np.asarray(route_s, dtype=np.float64)
    dists = haversine_matrix(ia[:, 0], ia[:, 1], ra[:, 0], ra[:, 1]).min(axis=1)
    
    # More aggressive threshold for v6
    threshold = 60.0
    bad_indices = np.where(dists > threshold)[0]
    if len(bad_indices) == 0:
        return route
    
    # Build corrective waypoint list
    corrective = [ideal_s[i] for i in bad_indices[::2]]
    if not corrective:
        return route
    
    # Merge into original waypoints
    merged = list(original_wps)
    for cp in corrective:
        best_pos, best_d = 0, 1e9
        for k in range(len(merged) - 1):
            d = point_to_segment_dist(cp, merged[k], merged[k + 1])
            if d < best_d:
                best_d = d
                best_pos = k + 1
        merged.insert(best_pos, cp)
    
    # Re-route with tighter constraints
    new_route = route_segment_constrained(G, merged, ideal,
                                         kdtree_data=kdtree_data,
                                         alpha=18.0, beta=1.0,
                                         adaptive_tube=True)
    
    if new_route:
        new_score = bidirectional_score(new_route, ideal)
        old_score = bidirectional_score(route, ideal)
        if new_score < old_score:
            log(f"[refine_v6] Improved: {old_score:.1f}m → {new_score:.1f}m")
            return new_route
    
    return route


def make_result(route, score, rot, scale, center, idx=None, name=""):
    """Build a standard result dict."""
    length_m = 0.0
    if route and len(route) >= 2:
        for i in range(len(route) - 1):
            length_m += haversine(route[i][0], route[i][1],
                                  route[i+1][0], route[i+1][1])
    r = {
        "route": route,
        "score": round(score, 1),
        "rotation": round(rot, 1),
        "scale": round(scale, 5),
        "center": [round(center[0], 6), round(center[1], 6)],
        "route_length_m": round(length_m, 0),
    }
    if idx is not None:
        r["shape_index"] = idx
    if name:
        r["shape_name"] = name
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  GRID GENERATION (Unchanged)
# ═══════════════════════════════════════════════════════════════════════════

DEG_PER_KM_LAT = 0.009
DEG_PER_KM_LNG = 0.012


def make_offsets(km_range, steps):
    """Generate (dlat, dlng) grid covering ±km_range."""
    vals = [0.0]
    for s in range(1, steps):
        v = (s / max(steps - 1, 1)) * km_range
        vals.extend([v, -v])
    return [(dlat * DEG_PER_KM_LAT, dlng * DEG_PER_KM_LNG)
            for dlat in vals for dlng in vals]


def coarse_grid_search(G, pts, center, rotations, scales, offsets,
                       densify_spacing=200, kdtree_data=None):
    """Coarse scan: proximity scoring only (no routing). Returns sorted list."""
    results = []
    for dlat, dlng in offsets:
        c = [center[0] + dlat, center[1] + dlng]
        for sc in scales:
            for rot in rotations:
                wps = shape_to_latlngs(pts, c, sc, rot)
                d = _uniform_densify(wps, spacing_m=densify_spacing)
                results.append((coarse_proximity_score(G, d,
                                kdtree_data=kdtree_data), rot, sc, c))
    results.sort(key=lambda x: x[0])
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MODE: FIT (Enhanced with v6 features)
# ═══════════════════════════════════════════════════════════════════════════

def mode_fit(payload):
    """
    v6 Smart Quick Fit with segment-constrained routing.
    """
    shapes = payload.get("shapes", [])
    idx = payload.get("shape_index", 0)
    center = payload.get("center_point", [51.505, -0.09])

    if idx < 0 or idx >= len(shapes):
        return {"error": "Invalid shape index"}

    pts = shapes[idx]["pts"]
    name = shapes[idx].get("name", "")

    G = fetch_graph(center, dist=2500) if HAS_OSMNX else None
    kd = build_kdtree(G) if G else None
    if G:
        log(f"[fit_v6] Graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges")

    # Step 1: Coarse scan
    rotations = list(range(0, 360, 45))
    scales = [0.010, 0.014, 0.018, 0.023, 0.030]
    offsets = make_offsets(km_range=1.0, steps=2)

    if G:
        coarse = coarse_grid_search(G, pts, center, rotations, scales,
                                    offsets, densify_spacing=150,
                                    kdtree_data=kd)
        top = coarse[:4]
    else:
        top = [(0, r, s, center)
               for r in [0, 90, 180, 270] for s in [0.009, 0.012, 0.016]]

    # Step 2: Fine routing with v6 enhancements
    best_score, best = 1e9, None
    n_routed = 0

    for _, rot, sc, c in top:
        for dr in [0, -15, 15]:
            for sf in [1.0, 0.90, 1.10]:
                for dlat, dlng in [(0, 0), (0.001, 0), (-0.001, 0),
                                   (0, 0.0015), (0, -0.0015)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score_v6(G, pts, r2, s2, c2,
                                                    kdtree_data=kd,
                                                    use_constraints=True)
                    n_routed += 1
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2, idx, name)
                        log(f"[fit_v6] Best: rot={r2:.0f}° scale={s2:.4f} "
                            f"score={score:.1f}m")

    # Refinement pass
    if best and best_score < 1e9:
        r2, s2, c2 = best['rotation'], best['scale'], best['center']
        ref_score, ref_route = fit_and_score_v6(G, pts, r2, s2, c2,
                                               kdtree_data=kd, refine=True,
                                               use_constraints=True)
        if ref_score < best_score:
            best = make_result(ref_route, ref_score, r2, s2, c2, idx, name)
            best_score = ref_score
            log(f"[fit_v6] Refined: score={ref_score:.1f}m")

    log(f"[fit_v6] Done — {n_routed} routings, best={best_score:.1f}m")
    return best or {"error": "Could not trace shape on road network"}


# ═══════════════════════════════════════════════════════════════════════════
#  MODE: OPTIMIZE (Enhanced with v6 features)
# ═══════════════════════════════════════════════════════════════════════════

def mode_optimize(payload):
    """
    v6 Full optimization with segment-constrained routing.
    """
    shapes = payload.get("shapes", [])
    idx = payload.get("shape_index", 0)
    center = payload.get("center_point", [51.505, -0.09])

    if idx < 0 or idx >= len(shapes):
        return {"error": "Invalid shape index"}

    pts = shapes[idx]["pts"]
    name = shapes[idx].get("name", "")

    G = fetch_graph(center, dist=4000)
    if G is None:
        return _osrm_optimize(pts, center, idx, name)

    kd = build_kdtree(G)

    # Step 1: Coarse — larger scales
    rotations = list(range(0, 360, 15))
    scales = [0.010, 0.014, 0.018, 0.022, 0.027, 0.033, 0.040]
    offsets = make_offsets(km_range=2.0, steps=3)
    coarse = coarse_grid_search(G, pts, center, rotations, scales,
                                offsets, densify_spacing=200,
                                kdtree_data=kd)

    # Step 2: Fine search with v6 routing
    best = _fine_search_v6(G, pts, coarse[:10], n_fine=8, kdtree_data=kd)
    if best is None:
        return {"error": "Could not fit shape. Try a different area."}

    # Refinement pass
    r2, s2, c2 = best['rotation'], best['scale'], best['center']
    ref_score, ref_route = fit_and_score_v6(G, pts, r2, s2, c2,
                                           kdtree_data=kd, refine=True,
                                           use_constraints=True)
    if ref_route and ref_score < best.get('score', 1e9):
        best = make_result(ref_route, ref_score, r2, s2, c2)
        log(f"[optimize_v6] Refined: score={ref_score:.1f}m")

    best["shape_index"] = idx
    best["shape_name"] = name
    return best


def _fine_search_v6(G, pts, candidates, n_fine=8, kdtree_data=None):
    """v6 fine search with segment constraints."""
    seen, kept = set(), []
    for _, rot, sc, c in candidates:
        key = (round(rot, 0), round(sc, 4), round(c[0], 4), round(c[1], 4))
        if key not in seen:
            seen.add(key)
            kept.append((rot, sc, c))
        if len(kept) >= n_fine:
            break

    best_score, best = 1e9, None
    for rot, sc, c in kept:
        for dr in [0, -7, 7, -15, 15]:
            for sf in [1.0, 0.88, 1.12]:
                for dlat, dlng in [(0, 0), (0.003, 0), (-0.003, 0),
                                   (0, 0.004), (0, -0.004)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score_v6(G, pts, r2, s2, c2,
                                                    kdtree_data=kdtree_data,
                                                    use_constraints=True)
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2)
    return best


def _osrm_optimize(pts, center, idx, name):
    """OSRM-only optimisation fallback."""
    best_score, best = 1e9, None
    for sc in [0.008, 0.012, 0.018]:
        for rot in range(0, 360, 30):
            score, route = fit_and_score_v6(None, pts, rot, sc, center,
                                           use_constraints=False)
            if score < best_score:
                best_score = score
                best = make_result(route, score, rot, sc, center, idx, name)
    return best or {"error": "Could not fit shape via OSRM."}


# ═══════════════════════════════════════════════════════════════════════════
#  MODE: BEST_SHAPE (Preserved from v5.1)
# ═══════════════════════════════════════════════════════════════════════════

def _resample_normalised(pts, n=36):
    """Resample normalised shape to n evenly-spaced perimeter points."""
    if len(pts) < 2:
        return [(0.5, 0.5)] * n
    cum = [0.0]
    for i in range(1, len(pts)):
        dx, dy = pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]
        cum.append(cum[-1] + math.sqrt(dx*dx + dy*dy))
    total = cum[-1]
    if total < 1e-9:
        return [tuple(pts[0])] * n
    step, result, seg = total / n, [], 0
    for i in range(n):
        target = i * step
        while seg < len(cum) - 2 and cum[seg + 1] < target:
            seg += 1
        sl = cum[seg + 1] - cum[seg]
        if sl < 1e-12:
            result.append((pts[seg][0], pts[seg][1]))
        else:
            t = (target - cum[seg]) / sl
            result.append((pts[seg][0] + t * (pts[seg+1][0] - pts[seg][0]),
                           pts[seg][1] + t * (pts[seg+1][1] - pts[seg][1])))
    return result


def _shape_distance(pts_a, pts_b, n=36):
    """Rotation-invariant RMS distance between two normalised shapes."""
    a = _resample_normalised(pts_a, n)
    b = _resample_normalised(pts_b, n)
    ax, ay = sum(p[0] for p in a) / n, sum(p[1] for p in a) / n
    a = [(p[0] - ax, p[1] - ay) for p in a]
    bx, by = sum(p[0] for p in b) / n, sum(p[1] for p in b) / n
    b = [(p[0] - bx, p[1] - by) for p in b]

    def rms(sa, sb, shift):
        return math.sqrt(sum((sa[i][0] - sb[(i+shift)%n][0])**2 +
                             (sa[i][1] - sb[(i+shift)%n][1])**2
                             for i in range(n)) / n)

    b_rev = list(reversed(b))
    return min(min(rms(a, b, s), rms(a, b_rev, s)) for s in range(n))


def _similarity_map(shapes, threshold=0.15):
    """Build shape_index → [similar indices] dict."""
    ns = len(shapes)
    sim = {i: [] for i in range(ns)}
    for i in range(ns):
        for j in range(i + 1, ns):
            if _shape_distance(shapes[i]["pts"], shapes[j]["pts"]) < threshold:
                sim[i].append(j)
                sim[j].append(i)
    return sim


def mode_best_shape(payload):
    """v6 best_shape mode (uses v6 routing)."""
    shapes = payload.get("shapes", [])
    center = payload.get("center_point", [51.505, -0.09])
    if not shapes:
        return {"error": "No shapes provided"}

    G = fetch_graph(center, dist=4000)
    if G is None:
        return {"error": "Could not fetch road network (osmnx required)."}

    kd = build_kdtree(G)
    sim_map = _similarity_map(shapes, threshold=0.15)

    # Coarse scan
    rotations = list(range(0, 360, 30))
    scales = [0.007, 0.012, 0.018]
    offsets = make_offsets(km_range=1.5, steps=2)

    all_coarse = []
    for si, shape in enumerate(shapes):
        p = shape["pts"]
        for dlat, dlng in offsets:
            c = [center[0] + dlat, center[1] + dlng]
            for sc in scales:
                for rot in rotations:
                    wps = shape_to_latlngs(p, c, sc, rot)
                    d = _uniform_densify(wps, spacing_m=250)
                    all_coarse.append((coarse_proximity_score(G, d,
                                       kdtree_data=kd),
                                       rot, sc, c, si))
    all_coarse.sort(key=lambda x: x[0])

    # Similarity boost
    top10 = all_coarse[:10]
    bonus, seen = [], set()
    for score, rot, sc, c, si in top10[:5]:
        for sim_si in sim_map.get(si, []):
            key = (sim_si, rot, round(sc, 5), round(c[0], 4), round(c[1], 4))
            if key not in seen:
                seen.add(key)
                bonus.append((score * 1.1, rot, sc, c, sim_si))

    candidates = sorted(list(top10) + bonus, key=lambda x: x[0])
    final, seen_p = [], set()
    for item in candidates:
        key = (item[4], item[1], round(item[2], 5))
        if key not in seen_p:
            seen_p.add(key)
            final.append(item)
        if len(final) >= 18:
            break

    # Fine evaluation with v6 routing
    best_score, best = 1e9, None
    for _, rot, sc, c, si in final:
        p = shapes[si]["pts"]
        for dr in [0, -10, 10]:
            for sf in [1.0, 0.9, 1.1]:
                for dlat, dlng in [(0, 0), (0.002, 0), (-0.002, 0),
                                   (0, 0.003), (0, -0.003)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score_v6(G, p, r2, s2, c2,
                                                    kdtree_data=kd,
                                                    use_constraints=True)
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2,
                                           si, shapes[si].get("name", ""))

    return best or {"error": "Insufficient road density for GPS art here."}


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

HANDLERS = {"fit": mode_fit, "optimize": mode_optimize,
            "best_shape": mode_best_shape}


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input"}))
        return

    mode = payload.get("mode", "fit")
    handler = HANDLERS.get(mode)
    if not handler:
        print(json.dumps({"error": f"Unknown mode: {mode}"}))
        return

    result = handler(payload)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
