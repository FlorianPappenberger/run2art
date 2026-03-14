"""
geometry.py — Geometric helpers for Run2Art
============================================
Shape manipulation, haversine math, densification, and sampling.
"""

import math
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  HAVERSINE
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


# ═══════════════════════════════════════════════════════════════════════════
#  SHAPE MANIPULATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_heart_variants(base_pts, n_variants=5):
    """Generate parametric heart shape variants (Phase 6).

    Adjusts cusp sharpness, cleft depth, and lobe roundness to produce
    slightly different heart outlines that may better fit the road network.
    Returns list of point-lists including the original.
    """
    variants = [base_pts]  # original always included

    # Find key indices in the base heart shape
    # Cusp = max y, cleft = min y (or first/last point = same)
    cusp_idx = max(range(len(base_pts)), key=lambda i: base_pts[i][1])
    # Lobe peaks = leftmost and rightmost x
    left_idx = min(range(len(base_pts)), key=lambda i: base_pts[i][0])
    right_idx = max(range(len(base_pts)), key=lambda i: base_pts[i][0])

    perturbations = [
        (0, +0.06),   # sharper cusp (push cusp down)
        (0, -0.04),   # blunter cusp
        (+0.04, 0),   # wider lobes (push lobe peaks outward)
        (-0.03, 0),   # narrower lobes
    ]

    for dx, dy in perturbations[:n_variants - 1]:
        pts = [list(p) for p in base_pts]
        if dy != 0:
            # Adjust cusp and nearby points
            pts[cusp_idx] = [pts[cusp_idx][0], pts[cusp_idx][1] + dy]
            for i in [cusp_idx - 1, cusp_idx + 1]:
                if 0 <= i < len(pts):
                    pts[i] = [pts[i][0], pts[i][1] + dy * 0.3]
        if dx != 0:
            # Adjust lobe points
            if right_idx < len(pts):
                pts[right_idx] = [pts[right_idx][0] + dx, pts[right_idx][1]]
            if left_idx < len(pts):
                pts[left_idx] = [pts[left_idx][0] - dx, pts[left_idx][1]]
        # Ensure closure
        if pts[0] != pts[-1]:
            pts[-1] = list(pts[0])
        variants.append(pts)

    return variants

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


# ═══════════════════════════════════════════════════════════════════════════
#  POLYLINE SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  ANGLES & DISTANCES
# ═══════════════════════════════════════════════════════════════════════════

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


def point_to_segments_vectorized(points, seg_starts, seg_ends):
    """Vectorized min perpendicular distance from each point to nearest segment.

    Args:
        points: (N,2) array of [lat,lng]
        seg_starts: (S,2) array of segment start [lat,lng]
        seg_ends: (S,2) array of segment end [lat,lng]

    Returns:
        (N,) array of min distances in metres
    """
    pts = np.asarray(points, dtype=np.float64)   # (N, 2)
    sa = np.asarray(seg_starts, dtype=np.float64)  # (S, 2)
    sb = np.asarray(seg_ends, dtype=np.float64)    # (S, 2)

    dx = sb[:, 0] - sa[:, 0]  # (S,)
    dy = sb[:, 1] - sa[:, 1]
    seg_sq = dx * dx + dy * dy  # (S,)

    # Expand dims for broadcasting: pts (N,1,2), sa (1,S,2)
    p_lat = pts[:, 0:1]  # (N,1)
    p_lng = pts[:, 1:2]
    sa_lat = sa[:, 0:1].T  # (1,S)
    sa_lng = sa[:, 1:2].T
    dx_e = dx[None, :]  # (1,S)
    dy_e = dy[None, :]
    seg_sq_e = seg_sq[None, :]  # (1,S)

    # Project each point onto each segment
    safe_sq = np.where(seg_sq_e < 1e-14, 1.0, seg_sq_e)
    t = ((p_lat - sa_lat) * dx_e + (p_lng - sa_lng) * dy_e) / safe_sq  # (N,S)
    t = np.where(seg_sq_e < 1e-14, 0.0, t)
    t = np.clip(t, 0.0, 1.0)

    proj_lat = sa_lat + t * dx_e  # (N,S)
    proj_lng = sa_lng + t * dy_e

    # Haversine from each point to its projection on each segment
    R = 6_371_000.0
    p_rad = np.pi / 180.0
    dlat = (proj_lat - p_lat) * p_rad / 2.0
    dlng = (proj_lng - p_lng) * p_rad / 2.0
    a = (np.sin(dlat) ** 2 +
         np.cos(p_lat * p_rad) * np.cos(proj_lat * p_rad) *
         np.sin(dlng) ** 2)
    dists = 2.0 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))  # (N,S)

    return dists.min(axis=1)  # (N,)


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


def identify_anchor_points(waypoints, angle_threshold=30):
    """Identify high-curvature anchor points (turning angle > threshold).

    Returns indices into the waypoints list that are critical shape vertices.
    Always includes first and last point.
    """
    if len(waypoints) < 3:
        return list(range(len(waypoints)))

    anchors = {0, len(waypoints) - 1}
    for i in range(1, len(waypoints) - 1):
        angle = abs(turning_angle(waypoints[i-1], waypoints[i], waypoints[i+1]))
        if angle > angle_threshold:
            anchors.add(i)

    return sorted(anchors)


# ═══════════════════════════════════════════════════════════════════════════
#  v8.0 — TANGENT FIELD & APEX DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_tangent_field(ideal_line):
    """Compute tangent bearing (degrees, 0=N, CW) at every point of the ideal line.

    Uses central differences at interior points, forward/backward at endpoints.
    Returns (N,) array of bearings in [0, 360).
    """
    pts = np.asarray(ideal_line, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return np.zeros(n)

    # Forward differences for all segments
    dlat = np.diff(pts[:, 0])
    dlng = np.diff(pts[:, 1])

    # Bearing of each segment: atan2(dlng, dlat) converted to compass
    seg_bearings = np.degrees(np.arctan2(dlng, dlat)) % 360.0

    tangents = np.empty(n)
    tangents[0] = seg_bearings[0]
    tangents[-1] = seg_bearings[-1]

    if n > 2:
        # Central average of adjacent segment bearings
        # Handle wraparound via unit-vector averaging
        b1 = np.radians(seg_bearings[:-1])
        b2 = np.radians(seg_bearings[1:])
        avg_sin = (np.sin(b1) + np.sin(b2)) * 0.5
        avg_cos = (np.cos(b1) + np.cos(b2)) * 0.5
        tangents[1:-1] = np.degrees(np.arctan2(avg_sin, avg_cos)) % 360.0

    return tangents


def detect_sharp_vertices(ideal_line, angle_threshold=120):
    """Find points where the template has sharp turns (deflection > threshold).

    For a heart shape these are typically:
      - Bottom apex (~180° turn)
      - Top-center dip (~140° turn)

    Returns list of (lat, lng, deflection_degrees) for sharp vertices.
    """
    pts = np.asarray(ideal_line, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return []

    # Compute deflection at each interior point
    d1_lat = pts[1:-1, 0] - pts[:-2, 0]
    d1_lng = pts[1:-1, 1] - pts[:-2, 1]
    d2_lat = pts[2:, 0] - pts[1:-1, 0]
    d2_lng = pts[2:, 1] - pts[1:-1, 1]

    a1 = np.arctan2(d1_lng, d1_lat)
    a2 = np.arctan2(d2_lng, d2_lat)
    deflection = np.abs(((np.degrees(a2 - a1) + 180) % 360) - 180)

    sharp = []
    for i in range(len(deflection)):
        if deflection[i] > angle_threshold:
            sharp.append((pts[i + 1, 0], pts[i + 1, 1], float(deflection[i])))

    return sharp


def normalize_to_unit_box(polyline):
    """Translate and scale polyline so bounding box fits [0,1]×[0,1].

    Uses uniform scaling (max of width/height) to preserve aspect ratio.
    Returns (N,2) NumPy array.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    if len(pts) < 2:
        return pts

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    scale = max(span[0], span[1])
    if scale < 1e-12:
        return pts - mn

    return (pts - mn) / scale


def edge_bearing(lat1, lng1, lat2, lng2):
    """Compass bearing (degrees, 0=N, CW) from point 1 to point 2."""
    return math.degrees(math.atan2(lng2 - lng1, lat2 - lat1)) % 360.0


def bearing_deviation(bearing1, bearing2):
    """Absolute angular deviation between two bearings, accounting for
    bidirectional travel (a road can be traversed in either direction).
    Returns value in [0, 90] — 0 means aligned, 90 means perpendicular.
    """
    diff = abs(bearing1 - bearing2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    if diff > 90.0:
        diff = 180.0 - diff
    return diff
