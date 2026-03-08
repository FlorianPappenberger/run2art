"""
engine.py ΓÇö Run2Art Geospatial Engine v5.1 (Optimised)
======================================================
Transforms geometric shapes into GPS art running routes on real road networks.

Algorithms:
  - Shape-aware routing: edge weights penalise deviation from ideal shape
    (Waschk & Kr├╝ger, SIGGRAPH Asia 2018 / CVM 2019)
  - Adaptive densification: tighter spacing on curves, looser on straights
  - 6-component scoring: coverage, detour, Hausdorff, perpendicular,
    turning-angle, length-ratio (Li & Fu 2026; dsleo/stravart 2024)
  - Shape-similarity clustering for best-shape search
  - Two-step coarseΓåÆfine optimisation (Balduz 2017 inspired)

Performance optimisations (v5.1):
  - NumPy-vectorised haversine, sample_polyline, bidirectional_score
  - scipy cKDTree for O(log n) nearest-node lookup
  - Lazy shape-weight function (computes only visited edges)
  - Early termination in fine search loops

Modes:
  "fit"         Smart quick fit ΓÇö light coarse scan + shape-aware routing
  "optimize"    Full two-step coarseΓåÆfine for a single shape
  "best_shape"  Two-step coarseΓåÆfine across ALL shapes with similarity boost

I/O: stdin JSON ΓåÆ stdout JSON
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
    log("[engine] osmnx + networkx loaded ΓÇö shape-aware routing enabled")
except ImportError:
    HAS_OSMNX = False
    log("[engine] WARNING: osmnx not found ΓÇö OSRM fallback only")


# ---------------------------------------------------------------------------
# Graph caching (disk + in-memory LRU)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

_graph_mem_cache = {}  # key ΓåÆ (graph, timestamp)
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


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  GEOMETRY HELPERS
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

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
    """Vectorised element-wise haversine. All arrays same length ΓåÆ 1-D result."""
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
    """Convert normalised shape ΓåÆ geographic lat/lng coordinates."""
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
    """Signed turning angle in degrees at point b for path aΓåÆbΓåÆc."""
    d1 = math.atan2(b[0] - a[0], b[1] - a[1])
    d2 = math.atan2(c[0] - b[0], c[1] - b[1])
    diff = math.degrees(d2 - d1)
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def point_to_segment_dist(p, a, b):
    """Perpendicular distance (m) from point p to segment aΓåÆb."""
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


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  DENSIFICATION
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def densify(waypoints, spacing_m=120):
    """Insert points so consecutive waypoints are Γëñ spacing_m apart."""
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


def adaptive_densify(waypoints, base_spacing=120, curve_spacing=50):
    """Adaptive densification: tighter on curves, looser on straights."""
    if len(waypoints) < 3:
        return densify(waypoints, spacing_m=base_spacing)
    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        curvature = 0.0
        if i > 0:
            curvature = max(curvature, abs(turning_angle(waypoints[i-1], a, b)))
        if i + 2 < len(waypoints):
            curvature = max(curvature, abs(turning_angle(a, b, waypoints[i+2])))
        if curvature > 60:
            spacing = curve_spacing
        elif curvature > 30:
            spacing = (base_spacing + curve_spacing) / 2
        else:
            spacing = base_spacing
        dist = haversine(a[0], a[1], b[0], b[1])
        n_seg = max(1, int(round(dist / spacing)))
        for j in range(1, n_seg + 1):
            t = j / n_seg
            dense.append([a[0] + t * (b[0] - a[0]),
                          a[1] + t * (b[1] - a[1])])
    return dense


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  SCORING (NumPy-vectorised)
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def bidirectional_score(route, ideal_pts):
    """
    6-component score (lower = better, in metres).
      1. Forward coverage   (idealΓåÆroute)     weight 0.30
      2. Reverse detour     (routeΓåÆideal)     weight 0.15
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

    # Full distance matrix (n_i ├ù n_r)
    dist_mat = haversine_matrix(ia[:, 0], ia[:, 1], ra[:, 0], ra[:, 1])

    # 1. Forward: ideal ΓåÆ nearest route point
    fwd_min = dist_mat.min(axis=1)
    fwd_avg = float(fwd_min.mean())
    coverage = float((fwd_min < 100).sum()) / n_i

    # 2. Reverse: route ΓåÆ nearest ideal point
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
    """Fast coarse score ΓÇö proximity of waypoints to nearest road nodes.
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


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  ROUTING
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def _set_shape_weights(G, ideal_line, penalty=3.0, attr='shape_weight'):
    """Set shape-aware edge weights on graph (vectorised)."""
    edges = list(G.edges(keys=True, data=True))
    n_edges = len(edges)
    if n_edges == 0:
        return attr
    mid_lats = np.empty(n_edges, dtype=np.float64)
    mid_lons = np.empty(n_edges, dtype=np.float64)
    lengths = np.empty(n_edges, dtype=np.float64)
    for i, (u, v, _k, data) in enumerate(edges):
        mid_lats[i] = (G.nodes[u]['y'] + G.nodes[v]['y']) * 0.5
        mid_lons[i] = (G.nodes[u]['x'] + G.nodes[v]['x']) * 0.5
        lengths[i] = data.get('length', 50)
    il = np.asarray(ideal_line, dtype=np.float64)
    n_seg = len(il) - 1
    min_dists = np.full(n_edges, 1e9, dtype=np.float64)
    for j in range(n_seg):
        ax, ay = il[j, 0], il[j, 1]
        bx, by = il[j+1, 0], il[j+1, 1]
        dx, dy = bx - ax, by - ay
        seg_sq = dx * dx + dy * dy
        if seg_sq < 1e-14:
            d = haversine_vector(mid_lats, mid_lons,
                                 np.full(n_edges, ax), np.full(n_edges, ay))
        else:
            t = ((mid_lats - ax) * dx + (mid_lons - ay) * dy) / seg_sq
            t = np.clip(t, 0.0, 1.0)
            proj_lat = ax + t * dx
            proj_lon = ay + t * dy
            d = haversine_vector(mid_lats, mid_lons, proj_lat, proj_lon)
        np.minimum(min_dists, d, out=min_dists)
    weights = lengths + penalty * min_dists
    for i, (u, v, _k, data) in enumerate(edges):
        data[attr] = weights[i]
    return attr


def _make_shape_weight_fn(G, ideal_line, penalty=3.0):
    """Lazy weight function ΓÇö computes shape penalty only for edges visited by Dijkstra."""
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
        best = 1e9
        for j in range(n_seg):
            best = min(best, point_to_segment_dist(
                [mid_lat, mid_lon], [il[j, 0], il[j, 1]],
                [il[j+1, 0], il[j+1, 1]]))
        w = length + penalty * best
        _cache[key] = w
        return w

    return weight_fn


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


def route_shape_aware(G, waypoints, ideal_line, kdtree_data=None):
    """Route with shape-deviation lazy weight function."""
    if G is None or not ideal_line or len(ideal_line) < 2:
        return route_graph(G, waypoints, kdtree_data=kdtree_data)
    weight_fn = _make_shape_weight_fn(G, ideal_line)
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
        log(f"[fetch_graph] Downloaded in {elapsed:.1f}s ΓÇö "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        _cache_put(key, G)
        return G
    except Exception as e:
        log(f"[fetch_graph] Error: {e}")
        return None


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  ROUTE + SCORE HELPER
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def fit_and_score(G, pts, rot, scale, center,
                  dense_spacing=80, curve_spacing=35, kdtree_data=None):
    """Densify ΓåÆ route (shape-aware if possible) ΓåÆ score."""
    wps = shape_to_latlngs(pts, center, scale, rot)
    dense = adaptive_densify(wps, base_spacing=dense_spacing,
                             curve_spacing=curve_spacing)
    ideal = adaptive_densify(wps, base_spacing=60, curve_spacing=25)

    route = None
    if G:
        route = route_shape_aware(G, dense, ideal, kdtree_data=kdtree_data)
        if not route:
            route = route_graph(G, dense, kdtree_data=kdtree_data)
    if not route:
        route = route_osrm(dense)
    if not route:
        return (1e9, None)

    return (bidirectional_score(route, ideal), route)


def make_result(route, score, rot, scale, center, idx=None, name=""):
    """Build a standard result dict."""
    r = {
        "route": route,
        "score": round(score, 1),
        "rotation": round(rot, 1),
        "scale": round(scale, 5),
        "center": [round(center[0], 6), round(center[1], 6)],
    }
    if idx is not None:
        r["shape_index"] = idx
    if name:
        r["shape_name"] = name
    return r


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  GRID GENERATION
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

DEG_PER_KM_LAT = 0.009
DEG_PER_KM_LNG = 0.012


def make_offsets(km_range, steps):
    """Generate (dlat, dlng) grid covering ┬▒km_range."""
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
                d = densify(wps, spacing_m=densify_spacing)
                results.append((coarse_proximity_score(G, d,
                                kdtree_data=kdtree_data), rot, sc, c))
    results.sort(key=lambda x: x[0])
    return results


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  MODE: FIT (Smart Quick Fit)
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def mode_fit(payload):
    """
    Smart Quick Fit ΓÇö light coarse scan + shape-aware routing.
      Step 1: 360 proximity combos (8 rotations ├ù 5 scales ├ù 9 offsets)
      Step 2: Route top-4 with ┬▒15┬░ / ┬▒10% fine variations
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
        log(f"[fit] Graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges")

    # Step 1: Coarse scan
    rotations = list(range(0, 360, 45))                # 8
    scales = [0.006, 0.009, 0.012, 0.016, 0.021]      # 5
    offsets = make_offsets(km_range=1.0, steps=2)       # 9

    if G:
        coarse = coarse_grid_search(G, pts, center, rotations, scales,
                                    offsets, densify_spacing=150,
                                    kdtree_data=kd)
        top = coarse[:4]
    else:
        top = [(0, r, s, center)
               for r in [0, 90, 180, 270] for s in [0.009, 0.012, 0.016]]

    # Step 2: Fine routing with early termination
    best_score, best = 1e9, None
    n_routed = 0

    for _, rot, sc, c in top:
        for dr in [0, -15, 15]:
            for sf in [1.0, 0.90, 1.10]:
                for dlat, dlng in [(0, 0), (0.001, 0), (-0.001, 0),
                                   (0, 0.0015), (0, -0.0015)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score(G, pts, r2, s2, c2,
                                                 kdtree_data=kd)
                    n_routed += 1
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2, idx, name)
                        log(f"[fit] Best: rot={r2:.0f}┬░ scale={s2:.4f} "
                            f"score={score:.1f}m")

    log(f"[fit] Done ΓÇö {n_routed} routings, best={best_score:.1f}m")
    return best or {"error": "Could not trace shape on road network"}


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  MODE: OPTIMIZE (Full two-step)
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def mode_optimize(payload):
    """
    Full optimisation for a single shape.
      Step 1: ~4200 coarse combos (proximity only)
      Step 2: Top-8 ├ù fine variations with shape-aware routing
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

    # Step 1: Coarse
    rotations = list(range(0, 360, 15))
    scales = [0.005, 0.008, 0.011, 0.014, 0.017, 0.020, 0.025]
    offsets = make_offsets(km_range=2.0, steps=3)
    coarse = coarse_grid_search(G, pts, center, rotations, scales,
                                offsets, densify_spacing=200,
                                kdtree_data=kd)

    # Step 2: Fine search
    best = _fine_search(G, pts, coarse[:10], n_fine=8, kdtree_data=kd)
    if best is None:
        return {"error": "Could not fit shape. Try a different area."}
    best["shape_index"] = idx
    best["shape_name"] = name
    return best


def _fine_search(G, pts, candidates, n_fine=8, kdtree_data=None):
    """Route top coarse candidates with fine variations + early termination."""
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
        candidate_improved = False
        for dr in [0, -7, 7, -15, 15]:
            for sf in [1.0, 0.88, 1.12]:
                for dlat, dlng in [(0, 0), (0.003, 0), (-0.003, 0),
                                   (0, 0.004), (0, -0.004)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score(G, pts, r2, s2, c2,
                                                 kdtree_data=kdtree_data)
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2)
                        candidate_improved = True
        # Early termination: skip remaining candidates if they keep failing
        if not candidate_improved and best is not None:
            pass  # continue to next candidate anyway
    return best


def _osrm_optimize(pts, center, idx, name):
    """OSRM-only optimisation fallback."""
    best_score, best = 1e9, None
    for sc in [0.008, 0.012, 0.018]:
        for rot in range(0, 360, 30):
            score, route = fit_and_score(None, pts, rot, sc, center,
                                         dense_spacing=150, curve_spacing=60)
            if score < best_score:
                best_score = score
                best = make_result(route, score, rot, sc, center, idx, name)
    return best or {"error": "Could not fit shape via OSRM."}


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  MODE: BEST_SHAPE (all shapes + similarity clustering)
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

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
    """Build shape_index ΓåÆ [similar indices] dict."""
    ns = len(shapes)
    sim = {i: [] for i in range(ns)}
    for i in range(ns):
        for j in range(i + 1, ns):
            if _shape_distance(shapes[i]["pts"], shapes[j]["pts"]) < threshold:
                sim[i].append(j)
                sim[j].append(i)
    return sim


def mode_best_shape(payload):
    """
    Two-step optimisation across ALL shapes with similarity-boosted selection.
      Step 1: 324 coarse combos per shape
      Step 1.5: Similarity boost for geometrically similar shapes
      Step 2: Fine routing for top-18 candidates
    """
    shapes = payload.get("shapes", [])
    center = payload.get("center_point", [51.505, -0.09])
    if not shapes:
        return {"error": "No shapes provided"}

    G = fetch_graph(center, dist=4000)
    if G is None:
        return {"error": "Could not fetch road network (osmnx required)."}

    kd = build_kdtree(G)
    sim_map = _similarity_map(shapes, threshold=0.15)

    # Step 1: Coarse for each shape
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
                    d = densify(wps, spacing_m=250)
                    all_coarse.append((coarse_proximity_score(G, d,
                                       kdtree_data=kd),
                                       rot, sc, c, si))
    all_coarse.sort(key=lambda x: x[0])

    # Step 1.5: Similarity boost
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

    # Step 2: Fine evaluation
    best_score, best = 1e9, None
    for _, rot, sc, c, si in final:
        p = shapes[si]["pts"]
        for dr in [0, -10, 10]:
            for sf in [1.0, 0.9, 1.1]:
                for dlat, dlng in [(0, 0), (0.002, 0), (-0.002, 0),
                                   (0, 0.003), (0, -0.003)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score(G, p, r2, s2, c2,
                                                 kdtree_data=kd)
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2,
                                           si, shapes[si].get("name", ""))

    return best or {"error": "Insufficient road density for GPS art here."}


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
#  MAIN
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

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
