# Engine Algorithm Documentation

> Detailed technical reference for `engine.py` v4 — the Python geospatial engine behind GPS Art.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Geometry Helpers](#geometry-helpers)
3. [Shape Densification](#shape-densification)
4. [Scoring Functions](#scoring-functions)
5. [Routing Backends](#routing-backends)
6. [Two-Step Optimiser](#two-step-optimiser)
7. [Mode Handlers](#mode-handlers)

---

## Pipeline Overview

```
Shape Template (normalised [0,1] points)
         │
         ▼
  shape_to_latlngs()       ← project onto geographic coordinates
         │
         ▼
  adaptive_densify()       ← insert control points (50m curves, 120m straights)
         │
         ▼
  ┌──────┴──────┐
  │  Routing    │
  │  Priority:  │
  │  1. segment-constrained (best)
  │  2. global shape-aware (good)
  │  3. plain shortest-path (basic)
  │  4. OSRM fallback (last resort)
  └──────┬──────┘
         │
         ▼
  bidirectional_score()    ← 6-component quality evaluation
         │
         ▼
  JSON output { route, score, rotation, scale, center }
```


## Geometry Helpers

### `haversine(lat1, lon1, lat2, lon2) → float`
Great-circle distance in metres between two geographic points using the Haversine formula.

### `rotate_shape(pts, angle_deg) → list`
Rotates normalised `[0,1]` shape points around centre `(0.5, 0.5)` by the given angle in degrees.

### `shape_to_latlngs(pts, center, scale_deg, rotation_deg) → list`
Converts normalised shape points to geographic `[lat, lng]` coordinates:
- Y-axis maps to latitude (inverted: higher y = more south)
- X-axis maps to longitude (with `×1.4` stretch to compensate for latitude distortion at ~50°N)
- `scale_deg` controls the size in degrees (~0.012° ≈ 1.3 km)

### `turning_angle(a, b, c) → float`
Signed turning angle in degrees at point `b` along path `a→b→c`. Used for curve detection, scoring, and shape fidelity evaluation.

### `point_to_segment_distance(p, a, b) → float`
Perpendicular distance (metres) from point `p` to line segment `a→b`. Falls back to nearest endpoint if the perpendicular projection lies outside the segment. This is the geometric primitive behind Li & Fu's (2026) ScoreS formula.

### `length_ratio_fidelity(route_pts, ideal_pts) → float`
Measures how well consecutive-segment length ratios match between route and ideal shape. From Li & Fu (2026) invariant spatial relations. Returns a penalty (0 = perfect, higher = worse).

### `_sample_polyline(pts, n) → list`
Evenly samples `n` points along a lat/lng polyline by arc length. Used to normalise route and ideal shapes to comparable point counts.


## Shape Densification

### Why Densification Matters

Without densification, a heart shape has ~9 vertices. The router finds the shortest path between these 9 points, which often means cutting across the shape's interior (e.g., cutting the top lobes of a heart). Densification inserts intermediate waypoints every 50–120m, forcing the router to follow the shape's outline.

### `densify_waypoints(waypoints, spacing_m=120) → list`
Uniform densification — inserts points so that consecutive waypoints are ≤ `spacing_m` apart.

### `adaptive_densify(waypoints, base_spacing=120, curve_spacing=50) → list`
**v4 improvement.** Adaptive densification that uses the turning angle at each segment to decide spacing:

| Curvature | Spacing | Rationale |
|-----------|---------|-----------|
| > 60° | 50m (`curve_spacing`) | Tight curves need maximum control |
| 30°–60° | 85m (midpoint) | Moderate curves |
| < 30° | 120m (`base_spacing`) | Straight segments need fewer points |

In testing, adaptive densification produces ~2.5× more points on curved segments (91 vs 37 for a heart shape) while adding minimal overhead on straights.


## Scoring Functions

### `hausdorff_distance(pts_a, pts_b) → float`
Hausdorff distance (metres) between two point sets — the maximum of all minimum distances. Captures the **worst-case deviation** between route and ideal. Adapted from dsleo/stravart's OpenCV-based approach.

### `perpendicular_score(route_pts, ideal_pts) → float`
Average perpendicular distance from route points to the nearest ideal shape **segment** (not just nearest point). This is geometrically more accurate than point-to-point distance and comes from Li & Fu's (2026) ScoreS formula.

### `bidirectional_score(route, ideal_pts) → float`
The primary scoring function. Returns a single float (lower = better, in metres). Composed of 6 weighted components:

```
score = fwd_avg × 0.30      ← forward coverage: ideal→route distance
      + rev_avg × 0.15      ← reverse detour: route→ideal distance
      + haus    × 0.10      ← Hausdorff: worst-case deviation
      + perp    × 0.20      ← perpendicular: segment-level precision
      + angle_penalty × 0.15 ← turning-angle fidelity
      + lr_penalty × 0.10   ← length-ratio consistency
```

If coverage < 75% (shape points within 100m of route), a multiplicative penalty is applied: `score × (2.0 - coverage)`.

**Weight rationale:**
- Waschk & Krüger emphasise coverage (forward component)
- Li & Fu emphasise geometric precision (perpendicular + angles)
- Hausdorff catches outlier deviations that averages miss

### `coarse_proximity_score(G, waypoints) → float`
Fast approximate score using only nearest-node distances (no routing). Used in Step 1 of the two-step optimiser to evaluate ~4,200 combinations in seconds.
```
score = avg_distance + max_distance × 0.3
```


## Routing Backends

### Priority Order

1. **Segment-constrained routing** (`snap_with_graph_segment_constrained`) — Best quality
2. **Global shape-aware routing** (`snap_with_graph` with `shape_aware=True`) — Good fallback
3. **Plain shortest-path routing** (`snap_with_graph`) — Basic
4. **OSRM fallback** (`snap_osrm`) — Always available, no local graph

### `_precompute_shape_distance_weights(G, ideal_line) → str`
Precomputes custom edge weights for every edge in the graph:
```
edge_weight = length + 3.0 × distance_from_edge_midpoint_to_nearest_ideal_segment
```
This is the core implementation of Waschk & Krüger's (2018) multi-objective routing: edges far from the ideal shape are penalised, forcing routes to stay close to the shape outline even when a shorter path exists.

### `snap_with_graph(G, waypoints, shape_aware=False, ideal_line=None) → list`
Routes through a pre-fetched osmnx graph using Dijkstra's shortest path. When `shape_aware=True`, uses the precomputed shape-deviation weights instead of raw distance.

### `snap_with_graph_segment_constrained(G, waypoints, ideal_line) → list`
**v4 innovation.** For each consecutive pair of waypoints:
1. Maps the waypoint pair to its nearest ideal line sub-segment
2. Sets **local** edge weights using only that sub-segment as reference (`PENALTY_FACTOR=4.0`)
3. Finds shortest path with these local weights

This prevents a critical problem: with global shape-aware weights, the router for segment A might be attracted toward segment B of the shape if B happens to be geometrically closer. Segment-constrained routing eliminates this cross-contamination.

### `snap_osrm(waypoints) → list`
Fallback routing via the public OSRM demo server. Routes each consecutive pair of waypoints independently via `foot` profile. Rate-limited with 50ms delays. No shape awareness.

### `fetch_graph(center, dist=2500) → Graph`
Downloads the walkable road network from OpenStreetMap via osmnx. Graph is cached in memory for the duration of the process.


## Two-Step Optimiser

### Design Rationale

Evaluating every combination of rotation × scale × position with full routing would require thousands of expensive API calls. The two-step approach:

1. **Step 1 (Coarse)** — Evaluates combinations using only `coarse_proximity_score()` (nearest-node distances, no routing). This takes milliseconds per combination.
2. **Step 2 (Fine)** — Only the top candidates from Step 1 get full routing + scoring. This reduces ~4,200 evaluations to ~600 actual routings.

### `coarse_grid_search(G, pts, center, rotations, scales, offsets) → list`
Step 1 grid parameters (for `optimize` mode):
- **Rotations:** 0° to 345° in 15° steps → 24 angles
- **Scales:** 7 values from 0.005 to 0.025
- **Offsets:** ±2 km in 3 steps → 25 positions
- **Total:** 24 × 7 × 25 = **4,200 combinations** (evaluated in seconds)

### `fine_search_around(G, pts, candidates, n_fine=8) → dict`
Step 2 fine variations per candidate:
- **Rotation deltas:** 0°, ±7°, ±15°
- **Scale factors:** 1.0, 0.88, 1.12
- **Position offsets:** ±300m
- **Total per candidate:** 5 × 3 × 5 = 75 variations
- **With 8 candidates:** 600 routings


## Shape Similarity

### Why Similarity Matters

Some shapes are geometrically similar — for example, Heart and Butterfly share bilateral symmetry with lobes, while Star and Crown both have spiky pointed tops. When the engine finds that Heart fits well at a certain rotation, scale, and position, it is likely that Butterfly will also fit well at similar parameters (and vice-versa).

The shape similarity system exploits this to make `best_shape` mode more intelligent.

### `_resample_normalised(pts, n=36) → list`
Resamples a normalised shape outline to `n` evenly-spaced perimeter points. Used to create a uniform representation for comparison.

### `_shape_distance(pts_a, pts_b, n=36) → float`
Computes a **rotation-invariant** distance between two shapes:
1. Resample both shapes to 36 points
2. Centre both at origin
3. Try all cyclic shifts (and reversed direction) to find the alignment that minimises RMS point-to-point distance

This metric is insensitive to rotation, translation, and path direction — only the geometric silhouette matters.

### `_compute_similarity_map(shapes, threshold=0.15) → dict`
Builds a lookup table: `shape_index → [similar_shape_indices]`. Shapes with an RMS distance below 0.15 are considered similar. This threshold was tuned to capture meaningful geometric neighbours without being too permissive.


## Mode Handlers

### `mode_fit(payload) → dict`
Single fit with caller-specified rotation and scale. Routing priority: segment-constrained → global shape-aware → plain → OSRM.

### `mode_optimize(payload) → dict`
Two-step optimisation for a single shape. Searches over rotation, scale, and position to find the best fit.

### `mode_best_shape(payload) → dict`
Two-step optimisation across ALL 20 shapes, enhanced with **shape-similarity clustering**:

1. Compute pairwise shape similarity using rotation-invariant RMS distance
2. Coarse grid of 324 combos per shape = 6,480 total
3. **Similarity boost**: if Shape A scores well at a certain rotation/scale/position, geometrically similar shapes (e.g., Heart ↔ Butterfly, Star ↔ Crown) are seeded with the same promising parameters
4. Fine routing of up to 18 candidates (top coarse + similarity-boosted neighbours)

This exploits the insight that similar shapes tend to fit well at similar parameter configurations, improving coverage without increasing total computation.

### `_optimize_osrm_fallback(pts, center, idx, shape_name) → dict`
Simplified optimisation using OSRM when osmnx is not available. Tests 12 rotations × 3 scales = 36 combinations.


## Data Flow (stdin/stdout)

### Input (JSON via stdin)
```json
{
  "mode": "fit|optimize|best_shape",
  "shapes": [
    { "name": "Heart", "pts": [[0.5, 0], [0.75, -0.18], ...] }
  ],
  "shape_index": 0,
  "center_point": [51.505, -0.09],
  "rotation_deg": 0,
  "scale": 0.012,
  "zoom_level": 15,
  "bbox": "..."
}
```

### Output (JSON via stdout)
```json
{
  "route": [[51.505, -0.09], [51.506, -0.091], ...],
  "score": 42.3,
  "rotation": 15.0,
  "scale": 0.01200,
  "center": [51.505, -0.09],
  "shape_index": 0,
  "shape_name": "Heart"
}
```

Or on error:
```json
{
  "error": "Could not trace shape on road network"
}
```
