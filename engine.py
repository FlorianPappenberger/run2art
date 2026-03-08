"""
engine.py — Run2Art Geospatial Engine v8.0 (No-Trunk Refactor)
===============================================================
Transforms geometric shapes into GPS art running routes on real road networks.

Architecture (v8.0):
  - geometry.py: haversine, shape manipulation, densification, tangent field, apex detection
  - scoring.py: 6-component vectorized bidirectional scorer (legacy)
  - scoring_v8.py: Fréchet-primary scorer with scale-invariant normalization
  - routing.py: Wide-Tube corridor, A* routing, pedestrian cost model
  - core_router.py: Soft-constraint CoreRouter — heading penalty, apex sensitivity, bidir A*
  - engine.py: Mode orchestrators (fit/optimize/best_shape)

v8.0 Innovations:
  - Soft-constraint edge weighting: proximity + heading + U-turn + biomech
  - Heading penalty: edges deviating >20° from local tangent are penalized
  - Apex sensitivity: U-turns only allowed within 15m of sharp vertices
  - Vectorized edge-weight precomputation (NumPy — no per-call Python)
  - Discrete Fréchet distance as primary recognizability metric
  - Adaptive wide-tube corridor (narrow at curves, wide on straights)
  - Bidirectional A* with shape-following heuristic
  - Edge-disjoint cycle finder for closed shapes

Modes:
  "fit"         v8 Smart quick fit — CoreRouter + Fréchet scoring
  "optimize"    v8 Full two-step coarse→fine with soft constraints
  "best_shape"  All shapes with similarity boost

I/O: stdin JSON → stdout JSON
"""

import sys
import json
import math
import time

import numpy as np

# ---------------------------------------------------------------------------
# Import from extracted modules
# ---------------------------------------------------------------------------

from geometry import (
    haversine, haversine_matrix, haversine_vector,
    rotate_shape, shape_to_latlngs, sample_polyline,
    turning_angle, point_to_segment_dist, min_dist_to_polyline,
    densify, adaptive_densify, identify_anchor_points,
)
from scoring import bidirectional_score, coarse_proximity_score
from scoring_v8 import score_v8, frechet_score
from routing import (
    log, HAS_OSMNX,
    fetch_graph, build_kdtree, build_corridor_subgraph,
    route_graph, route_shape_aware, route_with_anchors, route_osrm,
)
from core_router import CoreRouter


# ═══════════════════════════════════════════════════════════════════════════
#  V8 ROUTE + SCORE (CoreRouter + Fréchet)
# ═══════════════════════════════════════════════════════════════════════════

def fit_and_score(G, pts, rot, scale, center,
                  dense_spacing=40, curve_spacing=18, kdtree_data=None,
                  refine=False, use_corridor=True):
    """v8: Densify → CoreRouter (soft constraints + bidir A*) → Fréchet score."""
    wps = shape_to_latlngs(pts, center, scale, rot)
    ideal = adaptive_densify(wps, base_spacing=35, curve_spacing=15)

    if not G:
        dense = adaptive_densify(wps, base_spacing=dense_spacing,
                                 curve_spacing=curve_spacing)
        route = route_osrm(dense)
        if not route:
            return (1e9, None)
        return (score_v8(route, ideal), route)

    try:
        router = CoreRouter(G, ideal)
        route = router.route()
        if route and len(route) >= 2:
            return (router.score(route), route)
    except Exception as e:
        log(f"[v8] CoreRouter failed ({e}), falling back to v7 pipeline")

    # Fallback to v7 pipeline
    return _fit_and_score_v7(G, pts, rot, scale, center,
                              dense_spacing, curve_spacing, kdtree_data,
                              refine, use_corridor)


def _fit_and_score_v7(G, pts, rot, scale, center,
                       dense_spacing=40, curve_spacing=18, kdtree_data=None,
                       refine=False, use_corridor=True):
    """Original v7 pipeline (fallback)."""
    wps = shape_to_latlngs(pts, center, scale, rot)
    dense = adaptive_densify(wps, base_spacing=dense_spacing,
                             curve_spacing=curve_spacing)
    ideal = adaptive_densify(wps, base_spacing=35, curve_spacing=15)

    route = None
    if G:
        if use_corridor:
            G_sub, kd_sub = build_corridor_subgraph(G, ideal)
        else:
            G_sub, kd_sub = G, kdtree_data

        route = route_with_anchors(G_sub, dense, ideal, kdtree_data=kd_sub)
        if not route:
            route = route_shape_aware(G_sub, dense, ideal, kdtree_data=kd_sub)
        if not route:
            route = route_graph(G_sub, dense, kdtree_data=kd_sub)
        if not route and use_corridor:
            route = route_shape_aware(G, dense, ideal, kdtree_data=kdtree_data)
    if not route:
        route = route_osrm(dense)
    if not route:
        return (1e9, None)

    if refine and G and route and len(ideal) >= 4:
        route = _refine_route(G, route, ideal, dense, kdtree_data)

    return (score_v8(route, ideal), route)


def _refine_route(G, route, ideal, original_wps, kdtree_data):
    """One-pass refinement: find worst-deviation points and re-route."""
    ideal_s = sample_polyline(ideal, min(60, len(ideal) * 2))
    route_s = sample_polyline(route, min(80, len(route)))
    if len(ideal_s) < 4 or len(route_s) < 4:
        return route

    ia = np.asarray(ideal_s, dtype=np.float64)
    ra = np.asarray(route_s, dtype=np.float64)
    dists = haversine_matrix(ia[:, 0], ia[:, 1], ra[:, 0], ra[:, 1]).min(axis=1)

    threshold = 80.0
    bad_indices = np.where(dists > threshold)[0]
    if len(bad_indices) == 0:
        return route

    corrective = [ideal_s[i] for i in bad_indices[::2]]
    if not corrective:
        return route

    merged = list(original_wps)
    for cp in corrective:
        best_pos, best_d = 0, 1e9
        for k in range(len(merged) - 1):
            d = point_to_segment_dist(cp, merged[k], merged[k + 1])
            if d < best_d:
                best_d = d
                best_pos = k + 1
        merged.insert(best_pos, cp)

    new_route = route_shape_aware(G, merged, ideal, kdtree_data=kdtree_data)
    if new_route and bidirectional_score(new_route, ideal) < bidirectional_score(route, ideal):
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
    kd = build_kdtree(G) if G else None
    if G:
        log(f"[fit] Graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges")

    # Step 1: Coarse scan
    rotations = list(range(0, 360, 45))                # 8
    scales = [0.010, 0.014, 0.018, 0.023, 0.030]      # 5 — larger for better road coverage
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
                        log(f"[fit] Best: rot={r2:.0f}° scale={s2:.4f} "
                            f"score={score:.1f}m")

    # Refinement pass on the best result
    if best and best_score < 1e9:
        r2, s2, c2 = best['rotation'], best['scale'], best['center']
        ref_score, ref_route = fit_and_score(G, pts, r2, s2, c2,
                                              kdtree_data=kd, refine=True)
        if ref_score < best_score:
            best = make_result(ref_route, ref_score, r2, s2, c2, idx, name)
            best_score = ref_score
            log(f"[fit] Refined: score={ref_score:.1f}m")

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

    kd = build_kdtree(G)

    # Step 1: Coarse — larger scales for better road matching
    rotations = list(range(0, 360, 15))
    scales = [0.010, 0.014, 0.018, 0.022, 0.027, 0.033, 0.040]
    offsets = make_offsets(km_range=2.0, steps=3)
    coarse = coarse_grid_search(G, pts, center, rotations, scales,
                                offsets, densify_spacing=200,
                                kdtree_data=kd)

    # Step 2: Fine search
    best = _fine_search(G, pts, coarse[:10], n_fine=8, kdtree_data=kd)
    if best is None:
        return {"error": "Could not fit shape. Try a different area."}

    # Refinement pass on best candidate
    r2, s2, c2 = best['rotation'], best['scale'], best['center']
    ref_score, ref_route = fit_and_score(G, pts, r2, s2, c2,
                                          kdtree_data=kd, refine=True)
    if ref_route and ref_score < best.get('score', 1e9):
        best = make_result(ref_route, ref_score, r2, s2, c2)
        log(f"[optimize] Refined: score={ref_score:.1f}m")

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


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

HANDLERS = {"fit": mode_fit, "optimize": mode_optimize,
            "best_shape": mode_best_shape}

ENGINE_VERSION = "8.0"


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
