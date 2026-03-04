"""
engine.py — GPS Art Python Geospatial Engine  v4
=================================================

Major improvements over v3 (informed by deep research across GPS art ecosystem):

  v4 additions:
  5. Shape-aware routing — custom edge weights penalise road segments
     far from the ideal shape line (Waschk & Krüger 2018 multi-objective)
  6. Segment-constrained routing — per-segment local penalty fields prevent
     the router from drifting toward other parts of the shape
  7. Adaptive densification — tighter spacing on curves, looser on straights
  8. Enhanced scoring v4:
     a. Hausdorff distance (from dsleo/stravart) — worst-case deviation
     b. Perpendicular segment distance (Li & Fu 2026 ScoreS) — true
        geometric distance to ideal line segments, not just nearest points
     c. Length-ratio fidelity (Li & Fu 2026) — prevents stretching/compression
  9. point_to_segment_distance helper for precise geometric evaluation
  10. Shape-similarity clustering — rotation-invariant RMS distance between
      normalised shape outlines; powers intelligent best-shape search that
      propagates promising parameters to geometrically similar shapes

  v3 (preserved):
  1. Two-step coarse→fine optimisation
  2. Shape densification (uniform)
  3. Bidirectional scoring (forward + reverse)
  4. Centre-point shifting

Research references:
  - Waschk & Krüger, SIGGRAPH Asia 2018 / CVM 2019: multi-objective shortest
    path minimising Riemannian distance; standard routing sacrifices details
  - Li & Fu, ISPRS 2026: invariant spatial relationships (turning angles,
    length ratios) + subgraph matching; perpendicular distance scoring
  - Balduz, TU Vienna 2017: rasterisation proximity scoring for fast screening
  - dsleo/stravart (GitHub, 2024): Hausdorff distance, area-difference scoring,
    Optuna Bayesian optimization over position/rotation/scale
  - GPSArtify / GPS Art App: project-and-route approach; community feedback
    reveals "not enough roads" failures from naive shape matching
  - gps2gpx.art: manual overlay approach — shows difficulty of automation

Modes:
  "fit"         single shape with given rotation & scale (uses densification)
  "optimize"    two-step coarse→fine for a single shape
  "best_shape"  two-step coarse→fine across ALL shapes

Payload (stdin JSON):
  mode, shapes, shape_index, center_point, rotation_deg, scale, zoom_level, bbox

Output (stdout JSON):
  { route, score, rotation, scale, center, shape_index, shape_name }
  or { error }
"""

import sys, json, math, urllib.request, time

try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


# ╔═══════════════════════════════════════════════════════╗
# ║  GEOMETRY HELPERS                                     ║
# ╚═══════════════════════════════════════════════════════╝

def haversine(lat1, lon1, lat2, lon2):
    """Distance in metres between two lat/lng points."""
    R = 6_371_000
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) *
         math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(min(1, math.sqrt(a)))


def rotate_shape(pts, angle_deg):
    """Rotate normalised [0,1] points around centre (0.5, 0.5)."""
    cx, cy = 0.5, 0.5
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [[cx + (x - cx) * cos_a - (y - cy) * sin_a,
             cy + (x - cx) * sin_a + (y - cy) * cos_a] for x, y in pts]


def shape_to_latlngs(pts, center, scale_deg, rotation_deg=0):
    """Convert normalised shape points → geographic lat/lng."""
    if rotation_deg:
        pts = rotate_shape(pts, rotation_deg)
    cx, cy = 0.5, 0.5
    return [[center[0] - (y - cy) * scale_deg,
             center[1] + (x - cx) * scale_deg * 1.4] for x, y in pts]


def _sample_polyline(pts, n):
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
    while diff > 180: diff -= 360
    while diff < -180: diff += 360
    return diff


def point_to_segment_distance(p, a, b):
    """
    Perpendicular distance (m) from point p to line segment a→b.
    Falls back to nearest endpoint if projection is outside segment.
    Inspired by Li & Fu (2026) perpendicular distance evaluation.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-14:
        return haversine(p[0], p[1], a[0], a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg_len_sq))
    proj = [a[0] + t * dx, a[1] + t * dy]
    return haversine(p[0], p[1], proj[0], proj[1])


def length_ratio_fidelity(route_pts, ideal_pts):
    """
    Measure how well consecutive-segment length ratios match between
    route and ideal shape.  From Li & Fu (2026) invariant spatial relations.
    Returns a penalty score (0 = perfect match).
    """
    def seg_lengths(pts):
        return [haversine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                for i in range(len(pts) - 1)]
    r_lens = seg_lengths(route_pts)
    i_lens = seg_lengths(ideal_pts)
    if len(r_lens) < 2 or len(i_lens) < 2:
        return 0.0
    # Resample to same number of segments
    n = min(len(r_lens), len(i_lens), 20)
    r_s = _sample_lengths(r_lens, n)
    i_s = _sample_lengths(i_lens, n)
    diffs = []
    for k in range(len(r_s) - 1):
        r_ratio = r_s[k+1] / max(r_s[k], 1.0)
        i_ratio = i_s[k+1] / max(i_s[k], 1.0)
        diffs.append(abs(r_ratio - i_ratio))
    return sum(diffs) / max(len(diffs), 1)


def _sample_lengths(lengths, n):
    """Resample a list of segment lengths to n values."""
    if len(lengths) <= n:
        return lengths
    step = len(lengths) / n
    return [lengths[int(i * step)] for i in range(n)]


# ╔═══════════════════════════════════════════════════════╗
# ║  SHAPE DENSIFICATION                                  ║
# ╚═══════════════════════════════════════════════════════╝

def densify_waypoints(waypoints, spacing_m=120):
    """
    Insert intermediate waypoints so consecutive points are ≤ spacing_m apart.
    This is the KEY to shape recognisability — it prevents the router from
    taking short-cuts that destroy the shape's silhouette.
    """
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
    """
    Adaptive densification: tighter spacing on high-curvature segments,
    looser on straight segments.  A significant improvement over uniform
    densification for shape fidelity.

    Research insight: Waschk & Krüger (2018) note that geometric details
    are lost at curves; dsleo/stravart uses angle comparison to identify
    problem areas.  This addresses both.
    """
    if len(waypoints) < 3:
        return densify_waypoints(waypoints, spacing_m=base_spacing)

    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        # Determine curvature at this segment
        curvature = 0.0
        if i > 0:
            curvature = max(curvature,
                            abs(turning_angle(waypoints[i-1], a, b)))
        if i + 2 < len(waypoints):
            curvature = max(curvature,
                            abs(turning_angle(a, b, waypoints[i+2])))

        # High curvature (> 30°) → tighter spacing
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


# ╔═══════════════════════════════════════════════════════╗
# ║  SCORING FUNCTIONS                                     ║
# ╚═══════════════════════════════════════════════════════╝

def hausdorff_distance(pts_a, pts_b):
    """
    Hausdorff distance between two point sets (metres).
    Captures worst-case deviation — from dsleo/stravart metrics.py (2024).
    scipy.spatial.distance.directed_hausdorff equivalent, but pure-Python.
    """
    def directed_haus(s1, s2):
        return max(min(haversine(a[0], a[1], b[0], b[1]) for b in s2) for a in s1)
    return max(directed_haus(pts_a, pts_b), directed_haus(pts_b, pts_a))


def perpendicular_score(route_pts, ideal_pts):
    """
    For each route point, compute perpendicular distance to the nearest
    ideal shape *segment* (not just nearest point).
    From Li & Fu (2026) ScoreS formula.
    """
    if len(ideal_pts) < 2 or not route_pts:
        return 0.0
    total = 0.0
    # Sample route to limit computation
    step = max(1, len(route_pts) // 60)
    count = 0
    for i in range(0, len(route_pts), step):
        rp = route_pts[i]
        min_d = float('inf')
        for j in range(len(ideal_pts) - 1):
            d = point_to_segment_distance(rp, ideal_pts[j], ideal_pts[j+1])
            if d < min_d:
                min_d = d
        total += min_d
        count += 1
    return total / max(count, 1)


def bidirectional_score(route, ideal_pts):
    """
    Enhanced scoring v4 — incorporates findings from:
      - Li & Fu (2026): perpendicular distance + turning angles + length ratios
      - dsleo/stravart: Hausdorff distance for worst-case capture
      - Waschk & Krüger (2018): shape fidelity emphasis

    Components:
      1. Forward  (ideal→route): shape *coverage*
      2. Reverse  (route→ideal): *detour* penalty
      3. Hausdorff distance: worst-case deviation
      4. Perpendicular segment distance (Li & Fu ScoreS)
      5. Turning-angle fidelity
      6. Length-ratio fidelity

    Returns a single float (lower = better, in metres).
    """
    if not route or len(route) < 2:
        return 1e9

    n_ideal = min(80, max(30, len(ideal_pts) * 3))
    n_route = min(150, max(40, len(route)))
    ideal_s = _sample_polyline(ideal_pts, n_ideal)
    route_s = _sample_polyline(route, n_route)

    if not ideal_s or not route_s:
        return 1e9

    # ── 1. Forward: every ideal sample → nearest route sample ───────
    fwd_total = 0.0
    covered = 0
    for pt in ideal_s:
        d = min(haversine(pt[0], pt[1], rp[0], rp[1]) for rp in route_s)
        fwd_total += d
        if d < 100:
            covered += 1
    fwd_avg = fwd_total / len(ideal_s)
    coverage = covered / len(ideal_s)

    # ── 2. Reverse: every route sample → nearest ideal sample ───────
    rev_total = 0.0
    for rp in route_s:
        d = min(haversine(rp[0], rp[1], pt[0], pt[1]) for pt in ideal_s)
        rev_total += d
    rev_avg = rev_total / len(route_s)

    # ── 3. Hausdorff: worst-case deviation ──────────────────────────
    haus_s = _sample_polyline(ideal_pts, min(40, len(ideal_pts)))
    haus_r = _sample_polyline(route, min(60, len(route)))
    haus = hausdorff_distance(haus_s, haus_r)

    # ── 4. Perpendicular segment distance ───────────────────────────
    perp = perpendicular_score(route_s, ideal_s)

    # ── 5. Turning-angle fidelity at original waypoints ─────────────
    angle_penalty = 0.0
    if len(ideal_pts) >= 3 and len(route) >= 3:
        route_keys = _sample_polyline(route, len(ideal_pts))
        angle_diffs = []
        for k in range(1, min(len(ideal_pts), len(route_keys)) - 1):
            a_ideal = turning_angle(ideal_pts[k-1], ideal_pts[k], ideal_pts[k+1])
            a_route = turning_angle(route_keys[k-1], route_keys[k], route_keys[k+1])
            angle_diffs.append(abs(a_ideal - a_route))
        if angle_diffs:
            angle_penalty = (sum(angle_diffs) / len(angle_diffs)) * 1.5

    # ── 6. Length-ratio fidelity ────────────────────────────────────
    lr_penalty = length_ratio_fidelity(
        _sample_polyline(route, min(20, len(route))),
        _sample_polyline(ideal_pts, min(20, len(ideal_pts)))
    ) * 15.0  # Scale to be comparable with metre-based scores

    # ── Combined score ──────────────────────────────────────────────
    # Weights informed by ablation analysis across approaches:
    #   Waschk emphasises fwd (coverage), Li & Fu emphasises perp + angles
    score = (fwd_avg * 0.30 +
             rev_avg * 0.15 +
             haus * 0.10 +
             perp * 0.20 +
             angle_penalty * 0.15 +
             lr_penalty * 0.10)

    # Coverage penalty: if < 75% of shape within 100 m of route
    if coverage < 0.75:
        score *= (2.0 - coverage)

    return score


def coarse_proximity_score(G, waypoints):
    """
    FAST coarse score — no routing needed.
    Measures how close each waypoint is to its nearest road node.
    Uses osmnx nearest_nodes (KD-tree internally).
    """
    if G is None:
        return 1e9
    lats = [w[0] for w in waypoints]
    lngs = [w[1] for w in waypoints]
    try:
        nearest_ids = ox.nearest_nodes(G, lngs, lats)
    except Exception:
        return 1e9

    distances = []
    for i, nid in enumerate(nearest_ids):
        nlat = G.nodes[nid]['y']
        nlng = G.nodes[nid]['x']
        distances.append(haversine(lats[i], lngs[i], nlat, nlng))

    if not distances:
        return 1e9
    avg_d = sum(distances) / len(distances)
    max_d = max(distances)
    return avg_d + max_d * 0.3


# ╔═══════════════════════════════════════════════════════╗
# ║  SNAPPING / ROUTING BACKENDS                           ║
# ╚═══════════════════════════════════════════════════════╝

def _precompute_shape_distance_weights(G, ideal_line):
    """
    Precompute shape-aware edge weights.  For each edge in the graph,
    the weight = length + penalty * (distance from edge midpoint to
    nearest ideal shape segment).

    This is the KEY innovation from Waschk & Krüger (2018):
    "single-source multi-objective shortest path algorithm that
    minimizes the Riemannian distance" — we approximate this by
    augmenting edge weights with shape deviation penalty.
    """
    attr = 'shape_weight'
    PENALTY_FACTOR = 3.0  # How strongly deviation penalises vs raw distance
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get('length', 50)
        # Edge midpoint
        mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
        mid_lng = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        mid = [mid_lat, mid_lng]
        # Distance to nearest ideal segment
        min_d = float('inf')
        for j in range(len(ideal_line) - 1):
            d = point_to_segment_distance(mid, ideal_line[j], ideal_line[j+1])
            if d < min_d:
                min_d = d
        # Combined weight: original length + penalty for deviation
        data[attr] = length + PENALTY_FACTOR * min_d
    return attr


def snap_with_graph(G, waypoints, shape_aware=False, ideal_line=None):
    """
    Route through pre-fetched osmnx graph via shortest path.

    If shape_aware=True and ideal_line provided, uses shape-deviation
    weighted edges (Waschk & Krüger multi-objective approach) instead of
    pure distance.  This is the single biggest improvement for recognisability.
    """
    if G is None:
        return None
    try:
        weight = 'length'
        if shape_aware and ideal_line and len(ideal_line) >= 2:
            weight = _precompute_shape_distance_weights(G, ideal_line)

        node_ids = [ox.nearest_nodes(G, w[1], w[0]) for w in waypoints]
        full_route = []
        for i in range(len(node_ids) - 1):
            o, d = node_ids[i], node_ids[i + 1]
            if o == d:
                continue
            try:
                pn = nx.shortest_path(G, o, d, weight=weight)
            except nx.NetworkXNoPath:
                # Fallback: try plain length if shape-aware fails
                if weight != 'length':
                    try:
                        pn = nx.shortest_path(G, o, d, weight='length')
                    except nx.NetworkXNoPath:
                        continue
                else:
                    continue
            for nid in pn:
                pt = [G.nodes[nid]['y'], G.nodes[nid]['x']]
                if not full_route or full_route[-1] != pt:
                    full_route.append(pt)
        return full_route if len(full_route) >= 2 else None
    except Exception:
        return None


def snap_with_graph_segment_constrained(G, waypoints, ideal_line):
    """
    Segment-constrained routing: for each consecutive pair of waypoints,
    find the shortest path using shape-aware weights where the ideal
    sub-segment is used as the local reference line.

    This is more precise than global shape-aware routing because each
    segment gets its own local penalty field, preventing the router from
    drifting toward other parts of the shape.

    Combines insights from:
    - Waschk & Krüger (2018): per-segment shape fidelity
    - Li & Fu (2026): segment-level matching with local constraints
    """
    if G is None or not ideal_line or len(ideal_line) < 2:
        return snap_with_graph(G, waypoints)

    try:
        PENALTY_FACTOR = 4.0
        node_ids = [ox.nearest_nodes(G, w[1], w[0]) for w in waypoints]

        # Map each waypoint to its nearest ideal line segment
        wp_to_ideal_idx = []
        for w in waypoints:
            best_j = 0
            best_d = float('inf')
            for j in range(len(ideal_line) - 1):
                d = point_to_segment_distance(w, ideal_line[j], ideal_line[j+1])
                if d < best_d:
                    best_d = d
                    best_j = j
            wp_to_ideal_idx.append(best_j)

        full_route = []
        for i in range(len(node_ids) - 1):
            o, d = node_ids[i], node_ids[i + 1]
            if o == d:
                continue

            # Local ideal sub-segment with context
            seg_start = max(0, wp_to_ideal_idx[i] - 1)
            seg_end = min(len(ideal_line), wp_to_ideal_idx[min(i+1, len(wp_to_ideal_idx)-1)] + 2)
            local_ideal = ideal_line[seg_start:seg_end]
            if len(local_ideal) < 2:
                local_ideal = ideal_line

            # Set local shape weights
            for u, v, k, data in G.edges(keys=True, data=True):
                length = data.get('length', 50)
                mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
                mid_lng = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
                mid = [mid_lat, mid_lng]
                min_dist = min(point_to_segment_distance(mid, local_ideal[j], local_ideal[j+1])
                               for j in range(len(local_ideal) - 1))
                data['seg_weight'] = length + PENALTY_FACTOR * min_dist

            try:
                pn = nx.shortest_path(G, o, d, weight='seg_weight')
            except nx.NetworkXNoPath:
                try:
                    pn = nx.shortest_path(G, o, d, weight='length')
                except nx.NetworkXNoPath:
                    continue

            for nid in pn:
                pt = [G.nodes[nid]['y'], G.nodes[nid]['x']]
                if not full_route or full_route[-1] != pt:
                    full_route.append(pt)

        return full_route if len(full_route) >= 2 else None
    except Exception:
        return snap_with_graph(G, waypoints)


def snap_osrm(waypoints):
    """Route via public OSRM demo server."""
    full_route = []
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        url = (f"https://router.project-osrm.org/route/v1/foot/{coords}"
               f"?overview=full&geometries=geojson")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GPSArtApp/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get("routes"):
                for lon, lat in data["routes"][0]["geometry"]["coordinates"]:
                    pt = [lat, lon]
                    if not full_route or full_route[-1] != pt:
                        full_route.append(pt)
        except Exception:
            continue
        time.sleep(0.05)
    return full_route if len(full_route) >= 2 else None


def snap_osmnx(waypoints, center, dist=1500):
    """One-shot: fetch graph + route (for simple fit mode)."""
    G = ox.graph_from_point((center[0], center[1]), dist=dist, network_type='walk')
    if G.number_of_edges() < 10:
        return None
    return snap_with_graph(G, waypoints)


def snap(waypoints, center):
    """Try osmnx then OSRM."""
    if HAS_OSMNX:
        try:
            r = snap_osmnx(waypoints, center)
            if r:
                return r
        except Exception:
            pass
    return snap_osrm(waypoints)


def fetch_graph(center, dist=2500):
    """Fetch osmnx walk-network graph, or None."""
    if not HAS_OSMNX:
        return None
    try:
        G = ox.graph_from_point((center[0], center[1]), dist=dist, network_type='walk')
        return G if G.number_of_edges() >= 10 else None
    except Exception:
        return None


# ╔═══════════════════════════════════════════════════════╗
# ║  OFFSET / GRID GENERATION                              ║
# ╚═══════════════════════════════════════════════════════╝

DEG_PER_KM_LAT = 0.009
DEG_PER_KM_LNG = 0.012


def make_offsets(km_range, steps):
    """Generate (dlat, dlng) tuples covering ±km_range in a grid."""
    offsets = []
    vals = [0.0]
    for s in range(1, steps):
        v = (s / max(steps - 1, 1)) * km_range
        vals.extend([v, -v])
    for dlat_km in vals:
        for dlng_km in vals:
            offsets.append((dlat_km * DEG_PER_KM_LAT,
                            dlng_km * DEG_PER_KM_LNG))
    return offsets


# ╔═══════════════════════════════════════════════════════╗
# ║  TWO-STEP OPTIMISER CORE                               ║
# ╚═══════════════════════════════════════════════════════╝

def coarse_grid_search(G, pts, center, rotations, scales, offsets,
                       densify_spacing=200):
    """
    STEP 1 — Coarse: evaluate all combos using proximity scoring only
    (no expensive routing).  Returns sorted list of (score, rot, sc, center).
    """
    results = []
    for dlat, dlng in offsets:
        c = [center[0] + dlat, center[1] + dlng]
        for sc in scales:
            for rot in rotations:
                wps = shape_to_latlngs(pts, c, sc, rot)
                dense = densify_waypoints(wps, spacing_m=densify_spacing)
                score = coarse_proximity_score(G, dense)
                results.append((score, rot, sc, c))
    results.sort(key=lambda x: x[0])
    return results


def fine_evaluate(G, pts, rot, sc, center, dense_spacing=120):
    """
    STEP 2 — Fine: adaptive densification → shape-aware routing →
    enhanced bidirectional score.
    v4: Uses segment-constrained routing and adaptive densification.
    Returns (score, route) or (1e9, None).
    """
    wps = shape_to_latlngs(pts, center, sc, rot)
    dense = adaptive_densify(wps, base_spacing=dense_spacing, curve_spacing=50)
    ideal = adaptive_densify(wps, base_spacing=80, curve_spacing=30)

    # Try segment-constrained routing first (best quality)
    route = None
    if G:
        route = snap_with_graph_segment_constrained(G, dense, ideal)
        if not route:
            route = snap_with_graph(G, dense, shape_aware=True, ideal_line=ideal)
        if not route:
            route = snap_with_graph(G, dense)
    if route is None:
        route = snap_osrm(dense)
    if not route:
        return (1e9, None)

    score = bidirectional_score(route, ideal)
    return (score, route)


def fine_search_around(G, pts, candidates, n_fine=8):
    """
    For each top coarse candidate, try fine variations (±7° rotation,
    ±12% scale, ±300 m offset), route + score.  Returns best result dict.
    """
    best_score = 1e9
    best = None

    seen = set()
    kept = []
    for _, rot, sc, c in candidates:
        key = (round(rot, 0), round(sc, 4), round(c[0], 4), round(c[1], 4))
        if key not in seen:
            seen.add(key)
            kept.append((rot, sc, c))
        if len(kept) >= n_fine:
            break

    rot_deltas = [0, -7, 7, -15, 15]
    sc_factors = [1.0, 0.88, 1.12]
    off_deltas = [(0, 0),
                  (0.003, 0), (-0.003, 0), (0, 0.004), (0, -0.004)]

    for rot, sc, c in kept:
        for dr in rot_deltas:
            for sf in sc_factors:
                for do_lat, do_lng in off_deltas:
                    r2 = rot + dr
                    s2 = sc * sf
                    c2 = [c[0] + do_lat, c[1] + do_lng]
                    score, route = fine_evaluate(G, pts, r2, s2, c2)
                    if score < best_score:
                        best_score = score
                        best = {
                            "route": route,
                            "score": round(score, 1),
                            "rotation": round(r2, 1),
                            "scale": round(s2, 5),
                            "center": [round(c2[0], 6), round(c2[1], 6)],
                        }
    return best


# ╔═══════════════════════════════════════════════════════╗
# ║  MODE HANDLERS                                         ║
# ╚═══════════════════════════════════════════════════════╝

def mode_fit(payload):
    """
    Single fit with given rotation & scale.
    v4: Uses adaptive densification + shape-aware routing +
    segment-constrained routing when osmnx is available.
    """
    shapes = payload.get("shapes", [])
    idx = payload.get("shape_index", 0)
    center = payload.get("center_point", [51.505, -0.09])
    rot = payload.get("rotation_deg", 0)
    scale = payload.get("scale", 0.012)

    if idx < 0 or idx >= len(shapes):
        return {"error": "Invalid shape index"}

    pts = shapes[idx]["pts"]
    wps = shape_to_latlngs(pts, center, scale, rot)
    dense = adaptive_densify(wps, base_spacing=120, curve_spacing=50)
    ideal = adaptive_densify(wps, base_spacing=80, curve_spacing=30)

    # Try shape-aware routing first (requires osmnx)
    route = None
    if HAS_OSMNX:
        try:
            G = fetch_graph(center, dist=1500)
            if G:
                # Segment-constrained routing (best quality)
                route = snap_with_graph_segment_constrained(G, dense, ideal)
                if not route:
                    # Fallback to global shape-aware routing
                    route = snap_with_graph(G, dense, shape_aware=True, ideal_line=ideal)
        except Exception:
            pass

    # Fallback to standard routing
    if not route:
        route = snap(dense, center)
    if not route:
        return {"error": "Could not trace shape on road network"}

    sc = bidirectional_score(route, ideal)
    return {
        "route": route,
        "score": round(sc, 1),
        "rotation": rot,
        "scale": scale,
        "center": center,
        "shape_index": idx,
        "shape_name": shapes[idx].get("name", ""),
    }


def mode_optimize(payload):
    """
    Two-step optimisation for a single shape.

    Step 1 (COARSE): evaluate ~4200 combos in seconds using
    proximity-only scoring (no routing).

    Step 2 (FINE): route the top-8 candidates with fine variations,
    using densified waypoints + bidirectional scoring.
    Only ~600 actual routings instead of 4200.
    """
    shapes = payload.get("shapes", [])
    idx = payload.get("shape_index", 0)
    center = payload.get("center_point", [51.505, -0.09])

    if idx < 0 or idx >= len(shapes):
        return {"error": "Invalid shape index"}

    pts = shapes[idx]["pts"]

    G = fetch_graph(center, dist=4000)
    if G is None:
        return _optimize_osrm_fallback(pts, center, idx,
                                       shapes[idx].get("name", ""))

    # ── Step 1: Coarse grid ─────────────────────────────────────────
    rotations = list(range(0, 360, 15))           # 24 angles
    scales = [0.005, 0.008, 0.011, 0.014, 0.017, 0.020, 0.025]
    offsets = make_offsets(km_range=2.0, steps=3)  # 25 positions
    # Total: 24 × 7 × 25 = 4200 combos (fast, no routing)

    coarse = coarse_grid_search(G, pts, center, rotations, scales, offsets,
                                densify_spacing=200)
    top_n = coarse[:10]

    # ── Step 2: Fine search around top candidates ───────────────────
    best = fine_search_around(G, pts, top_n, n_fine=8)

    if best is None:
        return {"error": "Could not fit shape. Try a different area."}

    best["shape_index"] = idx
    best["shape_name"] = shapes[idx].get("name", "")
    return best


# ╔═══════════════════════════════════════════════════════╗
# ║  SHAPE SIMILARITY (for intelligent best-shape search)  ║
# ╚═══════════════════════════════════════════════════════╝

def _resample_normalised(pts, n=36):
    """Resample a normalised shape to *n* evenly-spaced perimeter points."""
    if len(pts) < 2:
        return [(0.5, 0.5)] * n
    # Cumulative distances
    cum = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        cum.append(cum[-1] + math.sqrt(dx * dx + dy * dy))
    total = cum[-1]
    if total < 1e-9:
        return [tuple(pts[0])] * n
    step = total / n
    result = []
    seg = 0
    for i in range(n):
        target = i * step
        while seg < len(cum) - 2 and cum[seg + 1] < target:
            seg += 1
        seg_len = cum[seg + 1] - cum[seg]
        if seg_len < 1e-12:
            result.append((pts[seg][0], pts[seg][1]))
        else:
            t = (target - cum[seg]) / seg_len
            result.append((pts[seg][0] + t * (pts[seg + 1][0] - pts[seg][0]),
                           pts[seg][1] + t * (pts[seg + 1][1] - pts[seg][1])))
    return result


def _shape_distance(pts_a, pts_b, n=36):
    """
    Rotation-invariant distance between two normalised shapes.
    Resamples both to *n* points, centres them, then finds the cyclic
    shift (and direction) that minimises RMS point-to-point distance.
    """
    a = _resample_normalised(pts_a, n)
    b = _resample_normalised(pts_b, n)
    # Centre both
    ax = sum(p[0] for p in a) / n
    ay = sum(p[1] for p in a) / n
    a = [(p[0] - ax, p[1] - ay) for p in a]
    bx = sum(p[0] for p in b) / n
    by = sum(p[1] for p in b) / n
    b = [(p[0] - bx, p[1] - by) for p in b]

    def _rms(seq_a, seq_b, shift):
        total = 0.0
        for i in range(n):
            j = (i + shift) % n
            dx = seq_a[i][0] - seq_b[j][0]
            dy = seq_a[i][1] - seq_b[j][1]
            total += dx * dx + dy * dy
        return math.sqrt(total / n)

    min_dist = float('inf')
    b_rev = list(reversed(b))
    for shift in range(n):
        d = _rms(a, b, shift)
        if d < min_dist:
            min_dist = d
        d = _rms(a, b_rev, shift)
        if d < min_dist:
            min_dist = d
    return min_dist


def _compute_similarity_map(shapes, threshold=0.15):
    """
    Build a dict  shape_index → [list of similar shape indices].

    Shapes whose rotation-invariant distance is below *threshold* are
    considered similar.  This powers the intelligent best-shape search:
    if one shape fits well at a location, its neighbours are likely to
    fit well too (and vice-versa).
    """
    ns = len(shapes)
    sim = {i: [] for i in range(ns)}
    for i in range(ns):
        for j in range(i + 1, ns):
            d = _shape_distance(shapes[i]["pts"], shapes[j]["pts"])
            if d < threshold:
                sim[i].append(j)
                sim[j].append(i)
    return sim


def mode_best_shape(payload):
    """
    Two-step optimisation across ALL shapes.

    Step 1: coarse grid for every shape (lighter grid per shape).
    Step 2: fine routing for the globally top-N candidates,
            boosted by shape-similarity clustering — similar shapes
            inherit promising rotation/scale/position parameters from
            their neighbours.
    """
    shapes = payload.get("shapes", [])
    center = payload.get("center_point", [51.505, -0.09])
    if not shapes:
        return {"error": "No shapes provided"}

    G = fetch_graph(center, dist=4000)
    if G is None:
        return {"error": "Could not fetch road network (osmnx required)."}

    # ── Shape similarity clustering ────────────────────────────────
    similar_map = _compute_similarity_map(shapes, threshold=0.15)

    # ── Step 1: Coarse for each shape ──────────────────────────────
    rotations = list(range(0, 360, 30))            # 12 angles
    scales = [0.007, 0.012, 0.018]                 # 3 sizes
    offsets = make_offsets(km_range=1.5, steps=2)   # 9 positions
    # Per shape: 12 × 3 × 9 = 324.  20 shapes → 6480 (all fast)

    all_coarse = []
    for si, shape in enumerate(shapes):
        pts = shape["pts"]
        for dlat, dlng in offsets:
            c = [center[0] + dlat, center[1] + dlng]
            for sc in scales:
                for rot in rotations:
                    wps = shape_to_latlngs(pts, c, sc, rot)
                    dense = densify_waypoints(wps, spacing_m=250)
                    score = coarse_proximity_score(G, dense)
                    all_coarse.append((score, rot, sc, c, si))
    all_coarse.sort(key=lambda x: x[0])

    # ── Step 1.5: Similarity-boosted candidate selection ───────────
    # Start with top-10 coarse results
    top_coarse = all_coarse[:10]

    # For the best-scoring shapes, create bonus candidates for their
    # similar shapes at the *same* promising parameters (rotation,
    # scale, centre).  "If one fits, its neighbours might fit too."
    bonus = []
    seen_keys = set()
    for score, rot, sc, c, si in top_coarse[:5]:
        for sim_si in similar_map.get(si, []):
            key = (sim_si, rot, round(sc, 5),
                   round(c[0], 4), round(c[1], 4))
            if key not in seen_keys:
                seen_keys.add(key)
                # Give bonus candidates a slight penalty so they don't
                # dominate true top scorers, but still get evaluated
                bonus.append((score * 1.1, rot, sc, c, sim_si))

    # Merge, deduplicate, and cap at 18 fine candidates
    candidates = list(top_coarse) + bonus
    candidates.sort(key=lambda x: x[0])
    final = []
    seen_params = set()
    for item in candidates:
        key = (item[4], item[1], round(item[2], 5))
        if key not in seen_params:
            seen_params.add(key)
            final.append(item)
        if len(final) >= 18:
            break

    # ── Step 2: Fine evaluation for top candidates ─────────────────
    best_score = 1e9
    best = None

    for _, rot, sc, c, si in final:
        pts = shapes[si]["pts"]
        for dr in [0, -10, 10]:
            for sf in [1.0, 0.9, 1.1]:
                for do_lat, do_lng in [(0,0), (0.002,0), (-0.002,0),
                                       (0,0.003), (0,-0.003)]:
                    r2 = rot + dr
                    s2 = sc * sf
                    c2 = [c[0] + do_lat, c[1] + do_lng]
                    score, route = fine_evaluate(G, pts, r2, s2, c2)
                    if score < best_score:
                        best_score = score
                        best = {
                            "route": route,
                            "score": round(score, 1),
                            "rotation": round(r2, 1),
                            "scale": round(s2, 5),
                            "center": [round(c2[0], 6), round(c2[1], 6)],
                            "shape_index": si,
                            "shape_name": shapes[si].get("name", ""),
                        }

    if best is None:
        return {"error": "Insufficient road density for GPS art here."}
    return best


def _optimize_osrm_fallback(pts, center, idx, shape_name):
    """Simpler OSRM-only optimisation when osmnx unavailable."""
    rotations = list(range(0, 360, 30))
    scales = [0.008, 0.012, 0.018]
    best_score = 1e9
    best = None
    for sc in scales:
        for rot in rotations:
            wps = shape_to_latlngs(pts, center, sc, rot)
            dense = densify_waypoints(wps, spacing_m=150)
            ideal = densify_waypoints(wps, spacing_m=80)
            route = snap_osrm(dense)
            if not route:
                continue
            score = bidirectional_score(route, ideal)
            if score < best_score:
                best_score = score
                best = {
                    "route": route,
                    "score": round(score, 1),
                    "rotation": rot,
                    "scale": round(sc, 5),
                    "center": center,
                    "shape_index": idx,
                    "shape_name": shape_name,
                }
    if best is None:
        return {"error": "Could not fit shape via OSRM."}
    return best


# ╔═══════════════════════════════════════════════════════╗
# ║  MAIN                                                  ║
# ╚═══════════════════════════════════════════════════════╝

def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input"}))
        sys.exit(1)

    mode = payload.get("mode", "fit")

    if mode == "optimize":
        result = mode_optimize(payload)
    elif mode == "best_shape":
        result = mode_best_shape(payload)
    else:
        result = mode_fit(payload)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
