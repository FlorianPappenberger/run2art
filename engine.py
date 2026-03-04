"""
engine.py — Run2Art Geospatial Engine v5
========================================
Transforms geometric shapes into GPS art running routes on real road networks.

Algorithms:
  - Shape-aware routing: edge weights penalise deviation from ideal shape
    (Waschk & Krüger, SIGGRAPH Asia 2018 / CVM 2019)
  - Adaptive densification: tighter spacing on curves, looser on straights
  - 6-component scoring: coverage, detour, Hausdorff, perpendicular,
    turning-angle, length-ratio (Li & Fu 2026; dsleo/stravart 2024)
  - Shape-similarity clustering for best-shape search
  - Two-step coarse→fine optimisation (Balduz 2017 inspired)

Modes:
  "fit"         Smart quick fit — light coarse scan + shape-aware routing
  "optimize"    Full two-step coarse→fine for a single shape
  "best_shape"  Two-step coarse→fine across ALL shapes with similarity boost

I/O: stdin JSON → stdout JSON
"""

import sys
import json
import math
import os
import hashlib
import pickle
import urllib.request
import time

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
    log("[engine] osmnx + networkx loaded — shape-aware routing enabled")
except ImportError:
    HAS_OSMNX = False
    log("[engine] WARNING: osmnx not found — OSRM fallback only")


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
    # Memory
    if key in _graph_mem_cache:
        log(f"[cache] Memory hit: {key}")
        return _graph_mem_cache[key][0]
    # Disk
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
    """Distance in metres between two geographic points."""
    R = 6_371_000
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) *
         math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(min(1, math.sqrt(a)))


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
    """Evenly sample *n* points along a lat/lng polyline."""
    if len(pts) < 2:
        return list(pts)
    dists = [0.0]
    for i in range(1, len(pts)):
        dists.append(dists[-1] + haversine(pts[i-1][0], pts[i-1][1],
                                           pts[i][0], pts[i][1]))
    total = dists[-1]
    if total < 1:
        return list(pts)
    step = total / max(n - 1, 1)
    sampled, seg = [], 0
    for i in range(n):
        target = i * step
        while seg < len(dists) - 2 and dists[seg + 1] < target:
            seg += 1
        seg_len = dists[seg + 1] - dists[seg]
        if seg_len < 1e-9:
            sampled.append(list(pts[seg]))
        else:
            t = (target - dists[seg]) / seg_len
            sampled.append([pts[seg][0] + t * (pts[seg+1][0] - pts[seg][0]),
                            pts[seg][1] + t * (pts[seg+1][1] - pts[seg][1])])
    return sampled


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


# ═══════════════════════════════════════════════════════════════════════════
#  DENSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def densify(waypoints, spacing_m=120):
    """Insert points so consecutive waypoints are ≤ spacing_m apart."""
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


# ═══════════════════════════════════════════════════════════════════════════
#  SCORING
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

    # 1. Forward: ideal → nearest route point
    fwd_total, covered = 0.0, 0
    for pt in ideal_s:
        d = min(haversine(pt[0], pt[1], rp[0], rp[1]) for rp in route_s)
        fwd_total += d
        if d < 100:
            covered += 1
    fwd_avg = fwd_total / len(ideal_s)
    coverage = covered / len(ideal_s)

    # 2. Reverse: route → nearest ideal point
    rev_avg = sum(min(haversine(rp[0], rp[1], pt[0], pt[1])
                      for pt in ideal_s) for rp in route_s) / len(route_s)

    # 3. Hausdorff distance (sampled)
    hs_i = sample_polyline(ideal_pts, min(40, len(ideal_pts)))
    hs_r = sample_polyline(route, min(60, len(route)))
    haus = max(
        max(min(haversine(a[0], a[1], b[0], b[1]) for b in hs_r) for a in hs_i),
        max(min(haversine(a[0], a[1], b[0], b[1]) for b in hs_i) for a in hs_r),
    )

    # 4. Perpendicular segment distance
    step = max(1, len(route_s) // 60)
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


def coarse_proximity_score(G, waypoints):
    """Fast coarse score — proximity of waypoints to nearest road nodes."""
    if G is None:
        return 1e9
    try:
        nids = ox.nearest_nodes(G, [w[1] for w in waypoints],
                                   [w[0] for w in waypoints])
    except Exception:
        return 1e9
    dists = [haversine(waypoints[i][0], waypoints[i][1],
                       G.nodes[nid]['y'], G.nodes[nid]['x'])
             for i, nid in enumerate(nids)]
    return (sum(dists) / len(dists) + max(dists) * 0.3) if dists else 1e9


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════════════════════════════════

def _set_shape_weights(G, ideal_line, penalty=3.0, attr='shape_weight'):
    """Set shape-aware edge weights on graph (Waschk & Krüger 2018)."""
    for u, v, _k, data in G.edges(keys=True, data=True):
        length = data.get('length', 50)
        mid = [(G.nodes[u]['y'] + G.nodes[v]['y']) / 2,
               (G.nodes[u]['x'] + G.nodes[v]['x']) / 2]
        data[attr] = length + penalty * min_dist_to_polyline(mid, ideal_line)
    return attr


def route_graph(G, waypoints, weight='length'):
    """Route through osmnx graph using given edge weight attribute."""
    if G is None:
        return None
    try:
        node_ids = [ox.nearest_nodes(G, w[1], w[0]) for w in waypoints]
        full = []
        for i in range(len(node_ids) - 1):
            o, d = node_ids[i], node_ids[i + 1]
            if o == d:
                continue
            try:
                path = nx.shortest_path(G, o, d, weight=weight)
            except nx.NetworkXNoPath:
                if weight != 'length':
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


def route_shape_aware(G, waypoints, ideal_line):
    """Route with shape-deviation weighted edges."""
    if G is None or not ideal_line or len(ideal_line) < 2:
        return route_graph(G, waypoints)
    attr = _set_shape_weights(G, ideal_line)
    return route_graph(G, waypoints, weight=attr)


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
        log(f"[fetch_graph] Downloaded in {elapsed:.1f}s — "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        _cache_put(key, G)
        return G
    except Exception as e:
        log(f"[fetch_graph] Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTE + SCORE HELPER
# ═══════════════════════════════════════════════════════════════════════════

def fit_and_score(G, pts, rot, scale, center,
                  dense_spacing=80, curve_spacing=35):
    """
    Densify → route (shape-aware if possible) → score.
    Returns (score, route) or (1e9, None).
    """
    wps = shape_to_latlngs(pts, center, scale, rot)
    dense = adaptive_densify(wps, base_spacing=dense_spacing,
                             curve_spacing=curve_spacing)
    ideal = adaptive_densify(wps, base_spacing=60, curve_spacing=25)

    route = None
    if G:
        route = route_shape_aware(G, dense, ideal)
        if not route:
            route = route_graph(G, dense)
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


# ═══════════════════════════════════════════════════════════════════════════
#  GRID GENERATION
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
                       densify_spacing=200):
    """Coarse scan: proximity scoring only (no routing). Returns sorted list."""
    results = []
    for dlat, dlng in offsets:
        c = [center[0] + dlat, center[1] + dlng]
        for sc in scales:
            for rot in rotations:
                wps = shape_to_latlngs(pts, c, sc, rot)
                d = densify(wps, spacing_m=densify_spacing)
                results.append((coarse_proximity_score(G, d), rot, sc, c))
    results.sort(key=lambda x: x[0])
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MODE: FIT (Smart Quick Fit)
# ═══════════════════════════════════════════════════════════════════════════

def mode_fit(payload):
    """
    Smart Quick Fit — light coarse scan + shape-aware routing.
      Step 1: 360 proximity combos (8 rotations × 5 scales × 9 offsets)
      Step 2: Route top-4 with ±15° / ±10% fine variations
    """
    shapes = payload.get("shapes", [])
    idx = payload.get("shape_index", 0)
    center = payload.get("center_point", [51.505, -0.09])

    if idx < 0 or idx >= len(shapes):
        return {"error": "Invalid shape index"}

    pts = shapes[idx]["pts"]
    name = shapes[idx].get("name", "")

    G = fetch_graph(center, dist=2500) if HAS_OSMNX else None
    if G:
        log(f"[fit] Graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges")

    # Step 1: Coarse scan
    rotations = list(range(0, 360, 45))                # 8
    scales = [0.006, 0.009, 0.012, 0.016, 0.021]      # 5
    offsets = make_offsets(km_range=1.0, steps=2)       # 9

    if G:
        coarse = coarse_grid_search(G, pts, center, rotations, scales,
                                    offsets, densify_spacing=150)
        top = coarse[:4]
    else:
        top = [(0, r, s, center)
               for r in [0, 90, 180, 270] for s in [0.009, 0.012, 0.016]]

    # Step 2: Fine routing
    best_score, best = 1e9, None
    n_routed = 0

    for _, rot, sc, c in top:
        for dr in [0, -15, 15]:
            for sf in [1.0, 0.90, 1.10]:
                for dlat, dlng in [(0, 0), (0.001, 0), (-0.001, 0),
                                   (0, 0.0015), (0, -0.0015)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score(G, pts, r2, s2, c2)
                    n_routed += 1
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2, idx, name)
                        log(f"[fit] Best: rot={r2:.0f}° scale={s2:.4f} "
                            f"score={score:.1f}m")

    log(f"[fit] Done — {n_routed} routings, best={best_score:.1f}m")
    return best or {"error": "Could not trace shape on road network"}


# ═══════════════════════════════════════════════════════════════════════════
#  MODE: OPTIMIZE (Full two-step)
# ═══════════════════════════════════════════════════════════════════════════

def mode_optimize(payload):
    """
    Full optimisation for a single shape.
      Step 1: ~4200 coarse combos (proximity only)
      Step 2: Top-8 × fine variations with shape-aware routing
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

    # Step 1: Coarse
    rotations = list(range(0, 360, 15))
    scales = [0.005, 0.008, 0.011, 0.014, 0.017, 0.020, 0.025]
    offsets = make_offsets(km_range=2.0, steps=3)
    coarse = coarse_grid_search(G, pts, center, rotations, scales,
                                offsets, densify_spacing=200)

    # Step 2: Fine search
    best = _fine_search(G, pts, coarse[:10], n_fine=8)
    if best is None:
        return {"error": "Could not fit shape. Try a different area."}
    best["shape_index"] = idx
    best["shape_name"] = name
    return best


def _fine_search(G, pts, candidates, n_fine=8):
    """Route top coarse candidates with fine variations."""
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
                    score, route = fit_and_score(G, pts, r2, s2, c2)
                    if score < best_score:
                        best_score = score
                        best = make_result(route, score, r2, s2, c2)
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


# ═══════════════════════════════════════════════════════════════════════════
#  MODE: BEST_SHAPE (all shapes + similarity clustering)
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
                    all_coarse.append((coarse_proximity_score(G, d),
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
                    score, route = fit_and_score(G, p, r2, s2, c2)
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
        sys.exit(1)

    mode = payload.get("mode", "fit")
    log(f"[engine] mode={mode}")
    result = HANDLERS.get(mode, mode_fit)(payload)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
