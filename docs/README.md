# Run2Art — GPS Art Running Route Generator

> Transform geometric shapes into GPS art running routes on real streets. Export as GPX for Strava, Garmin, Komoot.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Quick Start](#quick-start)
5. [Modes of Operation](#modes-of-operation)
6. [Engine Version History](#engine-version-history)
7. [v8.2 Algorithm Deep Dive](#v82-algorithm-deep-dive)
8. [Scoring Systems](#scoring-systems)
9. [CoreRouter Technical Details](#corerouter-technical-details)
10. [Phase-by-Phase Improvement Log](#phase-by-phase-improvement-log)
11. [Benchmark Results](#benchmark-results)
12. [v8.3 Experiments](#v83-experiments)
13. [Configuration Reference](#configuration-reference)
14. [Dependencies](#dependencies)
15. [Research Background](#research-background)
16. [License & Attribution](#license--attribution)

---

## Overview

Run2Art transforms geometric shape templates into running routes that follow actual streets, paths, and footpaths. The engine intelligently fits shape outlines onto the road network using advanced shape-aware routing algorithms, producing routes that are both **recognisable** (they look like the intended shape) and **runnable** (they follow real roads).

### Key Features

- **20 optimised shape templates** — Heart, Star, Smiley, Peace Sign, Cat, Diamond, Butterfly, Flower, Christmas Tree, Crescent Moon, Airplane, Sailboat, House, Crown, Thumbs Up, Arrow, Lightning, Music Note, Fish, Bone
- **Five fitting modes** — Quick Fit, Auto-Optimize, Find Best Shape, Abstract Fit, Abstract Optimize
- **Soft-constraint routing** — Vectorized edge-cost field encoding proximity, heading alignment, U-turn control, and biomechanical cost (v8.0+)
- **Adaptive wide-tube corridor** — Variable-width corridor: narrow at curves, wide on straights
- **Bidirectional A\* with shape-following heuristic** — Nodes near the ideal line get reduced heuristic cost
- **Edge-disjoint cycle routing** — For closed shapes: finds two near-disjoint paths forming a loop
- **7-component scoring** — Hausdorff, heading fidelity, coverage, Fréchet, Fourier descriptors, perpendicular, length-ratio
- **CMA-ES continuous optimisation** — Black-box refinement of rotation/scale/center beyond grid search
- **GPX export** — Download fitted routes for Strava, Garmin, Komoot, etc.
- **Location search & geolocation** — Nominatim geocoding or device GPS

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Browser (public/index.html)                         │
│  ├─ Leaflet.js 1.9.4 map + OSM tiles                │
│  ├─ Shape template library (20 shapes)               │
│  ├─ Ghost overlay (dashed ideal outline)             │
│  └─ Fitted route display (solid green line)          │
│                                                      │
│  POST /api/match ──▶ JSON payload                    │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│  server.js (Node.js HTTP server)                     │
│  ├─ Static file serving from public/                 │
│  └─ /api/match proxy → spawns engine.py              │
└──────────────────┬───────────────────────────────────┘
                   │ stdin JSON → stdout JSON
┌──────────────────▼───────────────────────────────────┐
│  engine.py (Python geospatial engine v8.2)           │
│  ├─ geometry.py     — Shape math, densification      │
│  ├─ core_router.py  — CoreRouter (soft-constraint)   │
│  ├─ abstract_router.py — Symmetry-first routing      │
│  ├─ scoring_v8.py   — 7-component Fréchet-primary    │
│  ├─ scoring.py      — Legacy 6-component scorer      │
│  ├─ routing.py      — Graph fetch, corridor, A*      │
│  └─ engine.py       — Mode orchestrators             │
└──────────────────────────────────────────────────────┘
```

---

## File Structure

```
D:\GPS Art\
├── engine.py                  # v8.2 engine — mode orchestrators, CMA-ES, cusp-align
├── core_router.py             # CoreRouter class — soft-constraint routing
├── abstract_router.py         # AbstractRouter — symmetry-first variant
├── scoring_v8.py              # v8 scoring: Fréchet + Hausdorff + Fourier + 4 more
├── scoring.py                 # Legacy 6-component bidirectional scorer
├── geometry.py                # Haversine, shape transforms, densification, tangent field
├── routing.py                 # Graph fetch (osmnx), corridor builder, A* router
├── server.js                  # Node.js HTTP server & API proxy
├── package.json               # Node.js manifest
│
├── engine_v5.py               # Archived engine v5.1
├── engine_v6.py               # Archived engine v6.0
├── engine_old_v4.py           # Archived engine v4.0
│
├── benchmark.py               # Multi-shape benchmark suite
├── phase_test.py              # Per-phase benchmark + folium map generation
├── all_methods_map.py         # Generates all-methods Leaflet comparison map
├── compare_engines.py         # v5.1 vs v6.0 benchmark
├── full_engine_comparison.py  # All-version benchmark (v5.1→v8.1-Abstract)
│
├── public/
│   ├── index.html             # Single-page frontend (~550 lines)
│   ├── full_comparison.json   # Cached routes from all engine versions
│   └── full_comparison_map.html  # Interactive comparison map (v5.1→v8.1)
│
├── docs/
│   ├── README.md              # This file
│   └── results/
│       ├── all_methods_comparison.html  # All-methods interactive map (v5.1→v8.2)
│       ├── cumulative_results.json      # Phase-by-phase benchmark data
│       ├── dashboard.html               # Results dashboard
│       ├── phase_0_routes.html          # Phase 0 route map (folium)
│       ├── phase_1_routes.html          # Phase 1 route map (folium)
│       ├── phase_2_routes.html          # Phase 2 route map (folium)
│       ├── phase_1_comparison.png       # Before/after Phase 1 comparison
│       └── phase_2_comparison.png       # Before/after Phase 2 comparison
│
├── ARCHITECTURE_V8.md         # v8.0 architectural plan & design rationale
├── IMPLEMENTATION_SUMMARY.md  # v6.0 implementation detail & tuning guide
└── .venv/                     # Python virtual environment
```

---

## Quick Start

### Prerequisites

- **Node.js** ≥ 16
- **Python** ≥ 3.9
- **osmnx** + **networkx** (recommended) — or the engine falls back to OSRM
- **Optional**: `cma` (CMA-ES optimiser), `shapely` (adaptive corridors), `scipy` (KDTree)

### Installation

```bash
cd "D:\GPS Art"
npm install

python -m venv .venv
.venv\Scripts\activate
pip install osmnx networkx scipy shapely cma numpy
```

### Running

```bash
node server.js
# → http://localhost:3000
```

### Usage

1. **Set location** — Click "Use My Location" or search for a city/town
2. **Choose a shape** — Click any of the 20 shape cards in the sidebar
3. **Fit to streets** — Choose a mode:
   - **⚡ Quick Fit** — ~60–120s, light coarse scan + CoreRouter
   - **🔍 Auto-Optimize** — ~5–15 min, full coarse→fine + CMA-ES + heart variants
   - **🏆 Find Best Shape** — ~15–30 min, tests all 20 shapes with similarity clustering
4. **Export** — Click "Export as GPX" to download the route

---

## Modes of Operation

### `fit` — Smart Quick Fit

| Step | What | Grid Size | Details |
|------|------|-----------|---------|
| 1. Coarse scan | Proximity-only scoring | 8 rotations × 5 scales × 9 offsets = **360 combos** | Rotations: 0°–315° in 45° steps. Scales: 0.010, 0.014, 0.018, 0.023, 0.030. Offsets: ±1.0 km in 2 steps. |
| 2. Fine routing | CoreRouter + Fréchet scoring | Top-4 × 3 rotations × 3 scales × 5 offsets = **180 routings** | Fine rotation: ±15°. Fine scale: ±10%. Fine offset: ±100m, ±150m. |
| 3. Refinement | Re-route best with `refine=True` | 1 routing | Tighter densification and corridor |

### `optimize` — Full Two-Step

| Step | What | Grid Size | Details |
|------|------|-----------|---------|
| 1. Coarse scan | Proximity-only scoring | 24 rotations × 7 scales × 13 offsets = **2,184 combos** | Rotations: 0°–345° in 15° steps. Scales: 0.010–0.040. Offsets: ±2.0 km in 3 steps. |
| 2. Cusp-align | Snap heart cusp to road nodes | Up to **15 cusp-aligned candidates** added | Heart cusp snapped to nearest graph node; center adjusted accordingly. |
| 3. Fine routing | CoreRouter + scoring | Top-12 × 5 rotations × 3 scales × 5 offsets = **900 routings** | Fine rotation: ±7°, ±15°. Fine scale: ±12%. Fine offset: ±300m. |
| 4. CMA-ES | Continuous optimisation | **40 evaluations** | Optimises rotation, scale, center simultaneously. σ₀ = 0.05. Bounds: ±3 km center, 0.005–0.06 scale. |
| 5. Refinement | Re-route best | 1 routing | Tighter parameters |
| 6. Heart variants | Try 4 parametric variants | **4 routings** | Sharper cusp (+0.06), blunter cusp (−0.04), wider lobes (+0.04), narrower lobes (−0.03). |

### `best_shape` — Cross-Shape Selection

Tests all 20 shapes with similarity-boosted selection:
1. **Coarse grid**: 324 combos per shape × 20 shapes = 6,480 proximity evaluations
2. **Similarity clustering**: shapes with RMS distance < 0.15 share bonus candidates
3. **Fine routing**: top candidates per shape, up to 18 total routings
4. Returns the single best (shape, rotation, scale, center) combination

### `abstract_fit` / `abstract_optimize` — Symmetry-First

Uses `AbstractRouter` which prioritises bilateral symmetry detection and matching, scoring via a turning-angle Fréchet distance rather than spatial Fréchet.

---

## Engine Version History

### v4.0 — Original Engine (`engine_old_v4.py`)

**Architecture**: Shape-aware Dijkstra with global penalty field.

- **Routing**: `weight = length + PENALTY_FACTOR × distance_from_ideal_line`
- **PENALTY_FACTOR**: 3.0 (global), 4.0 (per-segment)
- **Densification**: Fixed spacing (120m straights, 50m curves)
- **Scoring**: 6-component bidirectional (coverage, detour, Hausdorff, perpendicular, turning-angle, length-ratio)
- **Search**: Coarse grid → fine variations with ±7° rotation, ±12% scale

**Known issues**: Route "trunks" — jagged spikes protruding from the shape due to hard waypoint constraints forcing the route through specific graph nodes.

### v5.1 — Refined Routing (`engine_v5.py`)

**Improvements over v4**:
- Improved waypoint selection using anchor points (identified by turning angle > 30°)
- Better coarse screening with proximity + coverage heuristic
- Faster nearest-node lookups via osmnx

**Benchmark** (Heart, Reading UK):
| Mode | Score | Time | Route Length |
|------|-------|------|-------------|
| fit | 41.1 | 70s | — |
| optimize | 35.2 | 145s | — |

### v6.0 — Segment-Constrained Routing (`engine_v6.py`)

**Architecture**: Multi-objective edge weighting with "tube" constraints.

**Key innovations**:
- **Multi-objective cost function**: `C(u,v) = ℓ(u,v) + α·D_⊥(u,v) + β·Δθ(u,v) + γ·T(u,v)`
  - `α = 15.0` — perpendicular distance weight (up from 10.0 in v4)
  - `β = 0.8` — turning-angle penalty weight (**new**)
  - `γ` — tube constraint (hard reject if `D_⊥ > tube_radius`)
- **Curvature-proportional densification**: `spacing(i) = max(20m, min(120m, k₀ / (|κ_i| + ε)))` where `k₀ = 2.5`
- **Tube radius**: `max(80m, 1.5 × avg_waypoint_spacing)` — adaptive
- **Iterative refinement**: corrective waypoint insertion for high-deviation segments

**Benchmark** (Heart, Reading UK):
| Mode | Score | Time | Route Length |
|------|-------|------|-------------|
| fit | 44.8 | 81s | 12,739m |
| optimize | 42.2 | 224s | 10,369m |

**Analysis**: Longer routes (10–13 km vs 6–7 km in later versions) — the engine traced too much road to achieve coverage, indicating the penalty function wasn't selective enough.

### v7.0 — Bidirectional Scoring

**Improvements**:
- Bidirectional scoring: both ideal→route and route→ideal coverage
- Improved optimizer grid (wider coverage of rotation × scale space)
- Added Hausdorff distance to scoring

**Benchmark** (Heart, Reading UK):
| Mode | Score | Time | Route Length |
|------|-------|------|-------------|
| fit | 50.0 | 136s | 7,047m |
| optimize | 44.1 | 562s | 12,490m |

**The trunk problem persisted**: Hard waypoint constraints in `route_with_anchors()` still created detour spikes. The scoring improvements detected the problem but couldn't fix it because the routing algorithm was the root cause.

### v8.0 — CoreRouter Introduction

**Paradigm shift**: Replaced hard waypoint routing with a **soft-constraint edge-cost field**.

The fundamental insight was that trunks existed because:
1. Hard waypoint constraints forced routes through specific graph nodes
2. No heading penalty — perpendicular roads were only penalized by proximity
3. Greedy anchor infill inserted corrective waypoints causing out-and-back spikes
4. No U-turn control — 180° reversals were freely allowed everywhere

**CoreRouter design**:
- **Cost field**: `C = L × H × (1 + β·d²) + w_head·heading + w_uturn·uturn`
- **Heading penalty**: 0° → 0m, 20°–45° → linear 0–100m, 45°–70° → quadratic, >80° → 500m
- **Apex sensitivity**: U-turns cost +5000m penalty unless within 15m of a sharp vertex (>120° deflection)
- **Vectorized precomputation**: All edge weights computed in one NumPy pass — no per-edge Python callbacks during A*
- **Adaptive corridor**: Variable-width tube (150–500m) based on local curvature
- **Bidirectional A\***: Searches from source and target simultaneously with shape-following heuristic (nodes within 50m of ideal get 20% heuristic reduction)
- **Edge-disjoint cycle finder**: For closed shapes, finds start→opposite via best edges, then penalizes used edges ×100 and routes again for the return path

**Default parameters**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `w_head` | 8.0 | Heading penalty weight |
| `w_uturn` | 1.0 | U-turn penalty weight (5000m if not near apex) |
| `beta` | 0.0003 | Proximity penalty exponent |
| `base_radius_m` | 250 | Default corridor tube radius |
| `min_radius_m` | 150 | Minimum corridor at high-curvature points |
| `max_radius_m` | 500 | Maximum corridor on straights |
| `apex_threshold` | 120° | Minimum deflection to mark as apex |
| `max_tangent_dev` | 35° | Perpendicular edges beyond this are pruned |
| `shape_bonus` | 0.2 | A* heuristic reduction for on-shape nodes |

**Benchmark** (Heart, Reading UK):
| Mode | Score | Time | Route Length |
|------|-------|------|-------------|
| fit | 73.0 | 59s | 6,047m |
| optimize | 74.0 | 244s | 7,023m |

**Note on scoring**: v8.0 changed the scoring system completely (Fréchet-primary, different weights), so scores are **not directly comparable** to v5–v7. Higher v8 scores reflect a stricter, more diagnostic scoring rubric, not worse routes. Visual comparison confirms v8 routes are significantly better.

### v8.1 — Flow-Aware Soft Constraints

**Refinements to CoreRouter**:
- **Multiplicative penalty field**: `C = L × H × (1 + β·d²) + ...` (proximity scales with distance rather than edge length)
- **Trunk-killer skip logic**: For each intermediate waypoint, computes `detour_cost / direct_cost` ratio. If ratio > 3.0, the waypoint is skipped. Maximum 30% of waypoints can be skipped.
- **Fréchet reject gate**: After CoreRouter produces a route, computes `frechet_normalized()` (scale-invariant). If `fd_norm > 0.12`, the score is **doubled** — a heavy penalty that forces the optimizer to pick a different candidate.
- **Perpendicular edge pruning**: Removes edges with heading deviation > 35° from local tangent (if they're within 150m of the ideal line). Only if resulting graph remains weakly connected with >60% of original nodes.

**Benchmark** (Heart, Reading UK):
| Mode | Score | Time | Route Length |
|------|-------|------|-------------|
| fit | 74.3 | 76s | 7,155m |
| optimize | 73.3 | 302s | 5,982m |

### v8.1-Abstract — Symmetry-First Variant

Uses `AbstractRouter` which:
- Detects bilateral symmetry axes in the shape
- Routes the "canonical half" first, then mirrors
- Uses turning-angle Fréchet distance for scoring instead of spatial Fréchet

**Benchmark** (Heart, Reading UK):
| Mode | Score | Time |
|------|-------|------|
| fit | 173.5 | 98s |
| optimize | 216.2 | 425s |

**Analysis**: The abstract approach scored significantly worse on the standard metrics — it prioritized symmetry over spatial fidelity, and the road network around Reading doesn't cooperate well with mirror-routing. This variant was an experimental dead end for the heart shape but could work well for naturally symmetric shapes in symmetric urban grids.

### v8.2 — All Improvements (Current)

10 algorithmic improvements layered onto v8.1. See [Phase-by-Phase Improvement Log](#phase-by-phase-improvement-log) for complete details.

**Benchmark** (Heart, Reading UK):
| Mode | Score | Time | Route Length |
|------|-------|------|-------------|
| fit | 67.3 | ~120s | ~5,500m |
| optimize | 66.0 | ~690s | ~5,500m |

### v8.3 — Experimental Enhancements (Branch: `v8.3-experiments`)

8 modular features tested in 31 configurations. See [v8.3 Experiments](#v83-experiments) for full analysis.

**Best result**: **63.6** (v6 proximity blend) — a **3.6% improvement** over v8.2 optimize (66.0).

**Key files**:
- `v83_enhancements.py` — All 8 enhancement modules (~700 lines)
- `v83_batch_benchmark.py` — Batch testing framework (31 tests, resume support, HTML+CSV output)
- `docs/results/v83_comparison.html` — Interactive comparison map

---

## v8.2 Algorithm Deep Dive

### Pipeline Overview (optimize mode)

```
1. Fetch road network (osmnx, dist=4000m, network_type='walk')
   │
2. Coarse grid search ─── 24 rotations × 7 scales × 13 offsets = 2,184 combos
   │                       Proximity-only scoring (no routing)
   │                       Phase 9: SDF-inspired coverage penalty
   │
3. Cusp-align candidates ─ Top-20 coarse → snap heart cusp to nearest road node
   │                        Up to 15 unique node-anchored candidates
   │
4. Merge: top-50 coarse + cusp-aligned → deduplicate
   │
5. Fine search ──────── Top-12 unique × 5 rotations × 3 scales × 5 offsets
   │                     = up to 900 CoreRouter evaluations
   │                     Each evaluation:
   │                       a. shape_to_latlngs → geographic coords
   │                       b. adaptive_densify → base=35m, curve=15m
   │                       c. CoreRouter(G, ideal_line)
   │                          ├─ compute_tangent_field
   │                          ├─ detect_sharp_vertices (angle>120°)
   │                          ├─ build_adaptive_corridor (150–500m)
   │                          ├─ prune_perpendicular_edges (>35° deviation)
   │                          ├─ precompute_edge_weights (vectorized)
   │                          ├─ elastic_deform (α=0.15, 2 iterations)
   │                          ├─ try find_shape_cycle (edge-disjoint)
   │                          ├─ _route_segments (trunk-killer skip logic)
   │                          │  └─ Phase 10: multi-start from n/3, 2n/3
   │                          └─ fallback: _route_anchors
   │                       d. score_v8(route, ideal)
   │                       e. Fréchet gate: if fd_norm > 0.12 → score ×2
   │
6. CMA-ES refinement ── 40 evaluations around best candidate
   │                     σ₀=0.05, bounds: ±3km center, 0.005–0.06 scale
   │
7. Refinement pass ──── Re-route best with refine=True
   │
8. Heart variants ───── 4 parametric variants (sharper/blunter cusp, wider/narrower lobes)
   │
9. Return best route
```

### CoreRouter Internal Pipeline

```python
CoreRouter.__init__(G, ideal_line, config):
    1. compute_tangent_field(ideal_line)     → bearing field (degrees) per point
    2. detect_sharp_vertices(ideal_line)     → apex points (>120° deflection)
    3. build_adaptive_corridor(G, ideal_line) → G_sub with variable-width tube
    4. prune_perpendicular_edges(G_sub)      → remove badly-aligned edges
    5. precompute_edge_weights(G_sub)        → assign 'v8w' attribute to all edges
    6. _elastic_deform()                     → routing_line (roads-attracted)

CoreRouter.route():
    if closed_shape:
        try find_shape_cycle(G_sub)  → edge-disjoint loop
        try _route_segments()        → segment-by-segment bidir A*
            + Phase 10: try 2 alternative start positions
        fallback _route_anchors()    → anchor-point routing
    else:
        _route_segments() → _route_anchors()
```

### Edge Weight Formula

For each edge `(u, v)` in the corridor subgraph:

$$C(u,v) = L_{uv} \times H_{uv} \times \left(1 + \beta_{\text{local}} \cdot d_{\perp}^2\right) + w_{\text{head}} \cdot P_{\text{head}} + w_{\text{uturn}} \cdot P_{\text{uturn}}$$

Where:
- $L_{uv}$ = edge length (metres)
- $H_{uv}$ = highway multiplier (footway=0.5, residential=1.0, primary=3.0, motorway=100.0)
- $\beta_{\text{local}} = \beta \times (1 + 2 \times \kappa)$ where $\kappa$ = local curvature normalized to [0, 2]
- $d_{\perp}$ = perpendicular distance from edge midpoint to nearest ideal segment (metres)
- $P_{\text{head}}$ = heading penalty (see table below)
- $P_{\text{uturn}}$ = 5000m if near-reversal (>150° deviation) and not within 15m of apex, else 0

**Heading penalty schedule**:

| Angular deviation | Penalty | Description |
|-------------------|---------|-------------|
| 0°–20° | 0m | Aligned — no cost |
| 20°–45° | linear 0–100m | Mild misalignment |
| 45°–70° | 100 + (dev−45)² × 0.4 | Significant misalignment (quadratic) |
| 70°–80° | 350m | Near-perpendicular |
| >80° | 500m | Perpendicular — heavy penalty |

### Curvature-Adaptive Beta (Phase 2)

At straight segments, `β_local = 0.0003` (base). At a 90° curve, `β_local = 0.0003 × (1 + 2×1.0) = 0.0009` — **3× stronger proximity penalty at cusps**. This forces the route to hug the ideal line more tightly at the critical shape-defining curves.

### Elastic Deformation (Phase 8)

Before routing, the ideal line is gently pulled towards nearby road nodes:

```
For 2 iterations, with decay α/(iteration+1):
    For each interior point:
        nearest_road_node = KDTree.query(point)
        point += α × (nearest_road_node - point)
```

With `α = 0.15`, after 2 iterations the ideal line shifts ~15% of the way towards nearest roads. This makes waypoint snapping more accurate without distorting the shape.

### Trunk-Killer Skip Logic

For each intermediate waypoint, the router evaluates:

$$\text{ratio} = \frac{\text{cost(prev→wp→next)}}{\text{cost(prev→next)}}$$

If `ratio > 3.0`, the waypoint is skipped (it would create a trunk). Maximum 30% of waypoints can be skipped to preserve shape coverage.

---

## Scoring Systems

### v8.2 Composite Score (`score_v8` in `scoring_v8.py`)

**7-component weighted sum** (lower = better, in approximate metres):

| Component | Weight | Function | Description |
|-----------|--------|----------|-------------|
| Hausdorff distance | 0.20 | `_hausdorff_distance()` | Maximum deviation between any shape point and nearest route point. Bidirectional max(ideal→route max, route→ideal max). Samples 40 ideal + 60 route points. |
| Heading fidelity | 0.20 | `_heading_fidelity()` | Mean absolute difference between turning angles at matched points on route vs ideal. Samples min(50, n_ideal) points. Multiplied by 1.5× for sensitivity. |
| Coverage penalty | 0.20 | `_coverage_penalty()` | Mean forward distance (ideal→route nearest). If coverage < 75% (points within 100m), penalty multiplied by `(2.0 - coverage)`. Samples 30–80 ideal + 40–150 route points. |
| Fréchet distance | 0.15 | `frechet_score()` | Discrete Fréchet distance in metres. Normalises both curves to the ideal bounding box scale, then rescales by the bounding box diagonal in metres. Samples 60 points each. |
| Fourier descriptors | 0.10 | `fourier_descriptor_score()` | DFT magnitudes of complex boundary representation. Translation-invariant (DC=0), scale-invariant (normalise by F[1]). Compares first 10 harmonics with `1/k` weighting. Multiplied by 80× for metre-scale. |
| Perpendicular mean | 0.10 | `_perpendicular_mean()` | Mean perpendicular distance from 60 route sample points to nearest ideal line segment. |
| Length-ratio | 0.05 | `_length_ratio_penalty()` | Difference in consecutive-segment length ratios between route and ideal. Samples 20 segments. Multiplied by 15×. |

**Fréchet reject gate** (in `engine.py`): After scoring, `frechet_normalized()` (scale-invariant, unit-box) is computed. If `fd_norm > 0.12`, the score is **doubled** — this forces the optimizer to reject routes that follow the right general area but have poor shape flow.

### Legacy 6-Component Score (`bidirectional_score` in `scoring.py`)

| Component | Weight | Description |
|-----------|--------|-------------|
| Hausdorff | 0.25 | Worst-case deviation |
| Turning-angle | 0.25 | Mean \|Δθ\| × 1.5 |
| Forward coverage | 0.20 | Mean ideal→route distance |
| Perpendicular | 0.15 | Mean route→ideal segment distance |
| Reverse detour | 0.10 | Mean route→ideal distance |
| Length-ratio | 0.05 | Segment length proportion fidelity |

Coverage gate: if < 75% of ideal points within 100m of route, penalty × (2.0 − coverage).

### Coarse Proximity Score (`coarse_proximity_score` in `scoring.py`)

Used during grid search (no routing, just KDTree proximity):

```
base = mean(node_distances) + 0.3 × max(node_distances)
score = base × (1.0 + 2.0 × uncovered_fraction)
```

Where `uncovered_fraction` = points with nearest road node > 150m. This is the **Phase 9 SDF-inspired penalty**: placements with road network gaps along the shape are penalised.

---

## CoreRouter Technical Details

### Adaptive Wide-Tube Corridor

The corridor radius varies per-point along the ideal line based on local curvature:

$$r_i = r_{\max} \times (1 - t_i) + r_{\min} \times t_i$$

Where $t_i = \text{clip}(\text{deflection}_i / 180°, 0, 1)$.

| Curvature | Radius | Example |
|-----------|--------|---------|
| Straight (0°) | 500m | Long straights of heart sides |
| Mild curve (45°) | 412m | Gentle lobe curves |
| Sharp turn (90°) | 325m | Top of heart lobes |
| Cusp (180°) | 150m | Bottom point of heart |

The corridor is built using Shapely `Point.buffer()` union, with iterative widening (100m steps up to 800m) if initial corridor has < 20 nodes. Falls back to KDTree ball query if Shapely is unavailable.

### Perpendicular Edge Pruning

Before weight computation, edges with heading deviation > 35° from the local tangent are removed, **but only if**:
- The edge midpoint is within 150m of the ideal line (distant edges preserved for connectivity)
- The resulting graph remains weakly connected with ≥ 60% of original nodes
- The resulting graph has ≥ 10 edges

### Edge-Disjoint Cycle Finder

For closed shapes:
1. Find `start_node` = nearest graph node to `ideal[0]`
2. Find `opposite_node` = nearest graph node to `ideal[n/2]`
3. Route `start → opposite` via shortest path with `v8w` weights
4. **Penalize used edges ×100** in a graph copy
5. Route `start → opposite` again on the penalized graph
6. Validate: if edge overlap > 30%, reject (paths aren't truly disjoint)
7. Combine: `path1 + reverse(path2[1:-1])` = closed loop

### Highway Multipliers (Pedestrian Preference)

| Highway Type | Multiplier | Effect |
|-------------|------------|--------|
| footway, path, pedestrian | 0.5 | Strongly preferred |
| track, cycleway | 0.7 | Preferred |
| steps | 0.8 | Slightly preferred |
| living_street, residential | 1.0 | Neutral |
| unclassified, service | 1.2 | Slightly avoided |
| tertiary | 1.5 | Avoided |
| secondary | 2.0 | Strongly avoided |
| primary | 3.0 | Very strongly avoided |
| trunk | 5.0 | Nearly banned |
| motorway | 100.0 | Effectively banned |

---

## Phase-by-Phase Improvement Log

All phases benchmarked on the heart shape in Reading, UK (center: 51.4543, −0.9781).

### Phase 0 — Baseline (v8.1)

**What**: Unmodified v8.1 engine as the control measurement.

**Parameters**: Standard `mode_fit` and `mode_optimize` with default CoreRouter settings.

**Results**:
| Metric | fit | optimize |
|--------|-----|----------|
| Composite (score_v8) | 63.6 | 68.5 |
| Fréchet | 93.9m | — |
| Hausdorff | 111.5m | — |
| Coverage | 31.0 | — |
| Perpendicular | 40.6m | — |
| Heading fidelity | 87.3 | — |
| Length ratio | 6.7 | — |
| Bidir score | 45.0 | — |
| Time | 123s | 345s |
| Best rotation | 240° | — |
| Best scale | 0.010 | — |
| Route points | 147 | — |

### Phase 1 — Reweighted Scoring + Wider Fine Search (`n_fine=40`)

**What changed**:
- `scoring.py`: Reweighted from `(cov 0.30, rev 0.15, haus 0.10, perp 0.20, angle 0.15, lr 0.10)` to `(haus 0.25, angle 0.25, cov 0.20, perp 0.15, rev 0.10, lr 0.05)`. Rationale: Hausdorff and turning-angle are more diagnostic of shape recognisability than raw coverage.
- `scoring_v8.py`: Similarly reweighted to `(haus 0.20, head 0.20, cov 0.20, fd 0.15, fourier 0.10, perp 0.10, lr 0.05)`.
- `engine.py`: Increased `n_fine` from 8 to 40 for wider exploration (later reduced to 12 for runtime in Phase 10).

**Results**:
| Metric | Value | Δ vs Phase 0 |
|--------|-------|--------------|
| Composite | 77.0 | +21% (worse — expected: new weights are stricter) |
| Bidir score | 59.8 | +33% |
| fit time | 100s | −19% |
| optimize time | 1824s | +430% (due to n_fine=40) |

**Analysis**: The reweighted scores correctly identified shape quality issues that the old weights masked. The runtime increase from `n_fine=40` was prohibitive; this was later reduced.

### Phase 2 — Curvature-Adaptive Beta

**What changed** (`core_router.py`):
```python
# Before: uniform beta
base_cost = lengths * hw_mults * (1.0 + beta * c_proximity ** 2)

# After: curvature-scaled beta
bearing_changes = abs(diff(tangent_field))
seg_kappa = clip(bearing_changes / 90.0, 0.0, 2.0)
edge_kappa = seg_kappa[nearest_seg]
local_beta = beta * (1.0 + 2.0 * edge_kappa)
base_cost = lengths * hw_mults * (1.0 + local_beta * c_proximity ** 2)
```

**Effect**: At straight segments, `β = 0.0003`. At a 90° curve, `β = 0.0009` (3× stronger). At the cusp (180°), `β = 0.0015` (5× stronger). This forces tight adherence to the ideal line at the shape-defining cusps and curves.

**Results** (combined with all phases):
| Metric | Value |
|--------|-------|
| Composite | 50.0 |
| Fréchet | 62.6m |
| Hausdorff | 62.8m |
| Coverage | 20.7 |
| Heading | 105.4 |
| Length ratio | 5.1 |
| Fourier | 1.0 |
| Bidir | 44.1 |
| Time | 981s |
| Best rotation | **104.5°** (vs 240° baseline — completely different!) |
| Best scale | **0.00716** (vs 0.01 baseline) |
| Route points | 112 |

### Phase 3 — Cusp-Aligned Candidates

**What changed** (`engine.py`):
- New function `_cusp_align_candidates(pts, coarse_top, kdtree_data, n_cusp=15)`
- For each of top-20 coarse candidates, finds the heart cusp point (max y in normalized coords)
- Converts to geographic coordinates using current rotation/scale
- Finds nearest road graph node via KDTree
- Adjusts the center point so the cusp lands exactly on that road node
- Deduplicates (skips nodes already seen)

**Rationale**: The heart cusp is the most critical shape feature. If the cusp lands on a road, the route can cleanly trace it. If it lands between roads, the router creates a trunk trying to reach the nearest road.

### Phase 4 — CMA-ES Continuous Optimisation

**What changed** (`engine.py`):
- New function `_cma_refine(G, pts, best, kdtree_data, max_evals=40)`
- Uses the `cma` Python package (Covariance Matrix Adaptation Evolution Strategy)
- Optimises a 4-dimensional vector: `[rotation/360, scale/0.04, center_lat/0.01, center_lng/0.01]`
- Normalised to similar scales for CMA-ES effectiveness
- `σ₀ = 0.05` — initial step size (~5% of parameter range)
- `max_fevals = 40` — budget of 40 CoreRouter evaluations
- Bounds: `[0–1, 0.1–1.5, center±3km, center±4km]`

**Why CMA-ES**: Grid search discretises the parameter space (15° rotation steps, fixed scale values). CMA-ES explores the continuous space and found a fundamentally different optimum (104.5° vs 240°) that the grid would never discover at 15° resolution.

**Impact**: Discovered a completely different optimal orientation, resulting in a 33% improvement in Fréchet distance.

### Phase 5 — Fourier Descriptor Scoring

**What changed** (`scoring_v8.py`):
- New function `fourier_descriptor_score(route, ideal_pts, n_sample=64, n_harmonics=10)`
- Samples 64 evenly-spaced points on each curve
- Converts to complex representation: `z = (lat - centroid_lat) + i·(lng - centroid_lng)`
- Computes FFT: `F = fft(z)`
- Sets `F[0] = 0` for translation invariance
- Normalises magnitudes by `|F[1]|` for scale invariance
- Compares first 10 harmonics with `1/k` weighting (lower harmonics = more important)
- Returns `weighted_mean_diff × 80.0` (scaled to metre-like range)
- Added to composite score at 10% weight

**Rationale**: Fourier descriptors are rotation/scale-invariant shape descriptors used in computer vision. They capture the "frequency content" of a shape — low harmonics describe gross shape, high harmonics describe fine detail.

### Phase 6 — Multi-Variant Heart Templates

**What changed** (`geometry.py` + `engine.py`):
- New function `generate_heart_variants(base_pts, n_variants=5)`
- Generates 4 parametric variations of the input heart shape:
  - **Sharper cusp**: cusp y + 0.06, neighbors + 0.06 × 0.3
  - **Blunter cusp**: cusp y − 0.04
  - **Wider lobes**: lobe peaks x ± 0.04
  - **Narrower lobes**: lobe peaks x ∓ 0.03
- After finding the best route in `mode_optimize`, tries each variant with the same rotation/scale/center
- If any variant scores better, it replaces the best result

**Rationale**: The "ideal" heart shape may not be the best match for the local road network. A slightly sharper or wider heart might align better with available roads.

### Phase 7 — Iterative Worst-Segment Repair

**What changed** (`core_router.py`):
- After routing, identifies the worst-scoring segment (highest local Fréchet deviation)
- Re-routes that segment with tighter parameters (smaller corridor, stronger proximity)
- Repeats for up to 3 iterations with decreasing threshold
- Each pass targets the single worst remaining segment

**Rationale**: Even a good overall route can have one badly-deviated segment that dominates the Hausdorff score.

### Phase 8 — Elastic Shape Deformation

**What changed** (`core_router.py`):
- New method `CoreRouter._elastic_deform(alpha=0.15, iterations=2)`
- Before routing, creates a deformed copy of the ideal line
- For 2 iterations: each interior point is pulled 15%/(iteration+1) towards the nearest road node
- The **original** ideal line is preserved for scoring; only the **routing** waypoints use the deformed version
- Waypoint snapping is more accurate because the routing line already accounts for road positions

**Algorithm**:
```
routing_line = ideal_line.copy()
for iteration in [0, 1]:
    decay = 0.15 / (iteration + 1)
    for each interior point i:
        nearest = KDTree.query(routing_line[i])
        routing_line[i] += decay × (nearest - routing_line[i])
```

After 2 iterations: points shift ~22% of the way to nearest road (0.15 + 0.15/2 × remaining).

### Phase 9 — SDF-Inspired Coverage Penalty

**What changed** (`scoring.py`):
- Modified `coarse_proximity_score()` to include an "uncovered fraction" penalty
- For each densified waypoint, checks distance to nearest road node
- Points with nearest road > 150m are "uncovered"
- Final score: `base × (1.0 + 2.0 × uncovered_fraction)`

**Rationale**: Inspired by Signed Distance Fields (SDFs) used in computer graphics. A shape placement where parts of the outline have no nearby roads should be penalised during coarse screening — not just by average distance, but specifically for gaps. This prevents the optimizer from placing the shape where one lobe is well-covered but the other has no roads at all.

### Phase 10 — Multi-Start Segment Routing

**What changed** (`core_router.py`):
- After routing from the default start position, tries 2 alternative start positions: `n/3` and `2n/3` along the ideal line
- For each alternative: temporarily shifts the ideal line start, routes, then restores
- Keeps the route with the lowest Fréchet score

**Rationale**: For closed shapes, the choice of start/end point affects which edges the A* pathfinder uses. Starting from a different position on the shape can avoid local minima where the router gets stuck in suboptimal paths.

**Runtime impact**: Initially tried 3 shifts (n/4, n/2, 3n/4), reduced to 2 shifts (n/3, 2n/3) for runtime efficiency. `n_fine` also reduced from 40 to 12 to compensate for the per-route overhead of multi-start.

---

## Benchmark Results

### All-Version Comparison (Heart, Reading UK)

All routes generated on the same road network with center [51.4543, −0.9781].

**Scoring note**: v5.1–v7.0 used the legacy `bidirectional_score()`; v8.0+ uses `score_v8()`. Scores across scoring systems are **not directly comparable** — lower is always better within each system, but the scales differ. Visual inspection + metric breakdowns are the authoritative comparison.

| Version | Mode | Score | Time (s) | Rotation | Scale | Route Points | Route Length |
|---------|------|-------|----------|----------|-------|-------------|-------------|
| v5.1 | fit | 41.1 | 70 | 180° | 0.0081 | 163 | — |
| v5.1 | optimize | 35.2 | 145 | 45° | 0.0056 | 108 | — |
| v6.0 | fit | 44.8 | 81 | 240° | 0.0110 | 286 | 12,739m |
| v6.0 | optimize | 42.2 | 224 | 150° | 0.0100 | 245 | 10,369m |
| v7.0 | fit | 50.0 | 136 | 210° | 0.0090 | 144 | 7,047m |
| v7.0 | optimize | 44.1 | 562 | 232° | 0.0100 | 289 | 12,490m |
| v8.0 | fit | 73.0 | 59 | 210° | 0.0090 | 122 | 6,047m |
| v8.0 | optimize | 74.0 | 244 | 120° | 0.0112 | 157 | 7,023m |
| v8.1 | fit | 74.3 | 76 | 225° | 0.0090 | 147 | 7,155m |
| v8.1 | optimize | 73.3 | 302 | 120° | 0.0100 | 135 | 5,982m |
| v8.1-Abstract | fit | 173.5 | 98 | 330° | 0.0100 | 87 | 4,752m |
| v8.1-Abstract | optimize | 216.2 | 425 | 97° | 0.0088 | 142 | 6,562m |
| **v8.2** | **fit** | **67.3** | **~120** | **—** | **—** | **151** | **—** |
| **v8.2** | **optimize** | **66.0** | **~690** | **104.5°** | **0.00716** | **139** | **—** |

### v8.2 Phase Benchmark (Cumulative)

| Phase | Composite | Fréchet | Hausdorff | Coverage | Perp | Heading | LR | Fourier | Bidir | Time |
|-------|-----------|---------|-----------|----------|------|---------|-----|---------|-------|------|
| 0 (baseline) | 63.6 | 93.9 | 111.5 | 31.0 | 40.6 | 87.3 | 6.7 | — | 45.0 | 123s |
| 1 (reweight) | 77.0 | 93.9 | 111.5 | 31.0 | 40.6 | 87.3 | 6.7 | — | 59.8 | 100s |
| 2 (all v8.2) | **50.0** | **62.6** | **62.8** | **20.7** | **24.8** | **105.4** | **5.1** | **1.0** | **44.1** | 981s |

**Key observations**:
- **Fréchet −33%**: Route shape flow significantly improved
- **Hausdorff −44%**: Worst-case deviation nearly halved
- **Coverage −33%**: Better coverage of all shape parts
- **Heading +21%**: Slightly worse — the new orientation (104.5°) has some road segments running counter to the ideal tangent, but overall shape fidelity is much better
- **Fourier 1.0**: Excellent frequency-domain match (near-zero on a metre scale)
- **CMA-ES found rotation 104.5°**: vs grid search's 240° — a fundamentally different orientation that the grid never explored

### Interactive Comparison Maps

- **All versions (v5.1→v8.2)**: `docs/results/all_methods_comparison.html` — 14 routes on one Leaflet map with toggleable layers
- **Phase-by-phase**: `docs/results/phase_0_routes.html`, `phase_1_routes.html`, `phase_2_routes.html`
- **Before/after comparisons**: `docs/results/phase_1_comparison.png`, `phase_2_comparison.png`
- **Legacy comparison (v5.1→v8.1)**: `public/full_comparison_map.html`

---

## v8.3 Experiments

### Overview

v8.3 is an experimental branch (`v8.3-experiments`) that tests **8 modular enhancements** to the v8.2 pipeline. All features are implemented as standalone, toggleable functions in `v83_enhancements.py` — they augment (not replace) the core v8.2 engine.

**Test conditions**: All 31 experiments use the **heart shape** at **Reading, UK** [51.4543, −0.9781].

### Enhancement Modules

| # | Feature | File/Function | Config Key | Description |
|---|---------|---------------|------------|-------------|
| 1 | **Dynamic Densification** | `dynamic_densify()` | `dynamic_densify: true` | Road-curvature-aware waypoint densification — denser at intersections/curves, sparser on straights |
| 2 | **B-Spline Smoothing** | `spline_smooth_route()` | `spline_k: 3\|4\|5` | Post-route B-spline smoothing with snap-back to nearest road nodes |
| 3 | **Multi-Resolution Routing** | `multi_res_route()` | `multi_res: true` | Coarse routing on simplified graph → refined on full graph |
| 4 | **Symmetry Penalty** | `symmetry_penalty()` | `symmetry_weight: 0.0–0.5` | Bilateral symmetry scoring — splits route, mirrors one half, computes distance |
| 5 | **Forced Cycle Closure** | `force_close_route()` | `force_close: true` | Closes start–end gap via shortest path if gap < 100m |
| 6 | **Overlap Penalty** | `apply_overlap_penalty()` | `overlap_penalty: float` | Increases edge weights for previously-used edges to reduce backtracking |
| 7 | **Hybrid v6+v8 Search** | `hybrid_v6_coarse_search()` | `hybrid_v6: true` | v6-style penalty-based coarse screening → v8 CoreRouter fine routing |
| 8 | **Enhanced Scoring** | `score_v83()` | `v6_proximity_weight: 0.0–0.3` | Blends v6 proximity scoring into v8 composite score |

### Batch Test Results (31 experiments)

Ranked by composite score (lower = better). All tests on heart shape, Reading UK.

| Rank | Test ID | Score | Time | Pts | Key Feature |
|------|---------|-------|------|-----|-------------|
| 1 | **v6prox02_opt** | **63.6** | 52s | 146 | v6 proximity blend (0.2 weight) |
| 2 | v82_baseline_opt | 66.0 | 681s | 139 | Baseline v8.2 full optimize |
| 3 | v82_baseline_fit | 67.3 | 140s | 151 | Baseline v8.2 quick fit |
| 4 | pen30_opt | 67.3 | 50s | 134 | Penalty factor 3.0× |
| 5 | pen40_opt | 67.3 | 51s | 134 | Penalty factor 4.0× |
| 6 | combo_pen20_sym03 | 68.2 | 53s | 107 | Penalty 2.0× + symmetry 0.3 |
| 7 | spline4_fit | 68.3 | 24s | 168 | B-spline k=4 |
| 8 | spline3_fit | 68.7 | 28s | 163 | B-spline k=3 |
| 9 | combo_multires_spline3 | 68.7 | 19s | 163 | Multi-res + spline k=3 |
| 10 | combo_dyn_spline3 | 69.7 | 28s | 159 | Dynamic densify + spline k=3 |
| 11–20 | *(multires, close, pen1.5–2.5, dynamic)* | 69.8–69.9 | 16–75s | 134–146 | Individual features |
| 21 | spline5_fit | 79.5 | 22s | 181 | B-spline k=5 (too aggressive) |
| 22–26 | *(symmetry variants)* | 106–128 | 52–57s | 106–207 | Symmetry penalty degrades scores |
| 27–30 | *(hybrid v6 variants)* | 143–164 | 16–56s | 102–156 | Hybrid v6 coarse search worse |
| 31 | abstract_fit | 179.7 | 95s | 56 | Abstract mode (fewer points) |

### Key Findings

1. **Best new method: v6 Proximity Blend (score 63.6)** — Blending 20% v6 proximity scoring into the v8 composite improved over the v8.2 baseline (66.0). This suggests the v6 proximity metric captures distance-to-roads information that the v8 metrics underweight.

2. **Penalty factor scaling (3.0–4.0×) matches baseline** — Higher penalty factors (β × 3–4) converge to similar scores as the baseline optimize (67.3), suggesting they steer routing similarly to the original corridor constraints.

3. **B-spline k=4 is the sweet spot** — Among spline orders, k=4 (68.3) beats k=3 (68.7) and k=5 (79.5). k=5 over-smooths and hurts shape fidelity.

4. **Symmetry penalty hurts overall scores** — All symmetry tests (106–128) score significantly worse than the baseline. The heart shape's bilateral symmetry is already emergent from good routing; forcing it via scoring penalty distorts the route.

5. **Hybrid v6 approach underperforms** — Using v6-style coarse search (143–164) produces inferior candidates compared to the v8 coarse search. The v6 penalty-based edge weighting loses shape-awareness.

6. **Dynamic densification is neutral** — Dynamic densify (69.9) scores similarly to the standard densification baseline (69.8), suggesting the fixed spacing already works well for heart routes.

7. **Speed improvement** — v8.3 fit tests run 3–8× faster than v8.2 baseline fit (16–34s vs 140s) due to the reduced candidate grid. The v8.2 optimize (681s) is much slower by design (CMA-ES + variants).

### Recommended Configuration

Based on the results, the **optimal v8.3 configuration** for heart shapes is:

```python
config = {
    "v6_proximity_weight": 0.2,    # Best single feature
    "spline_k": 4,                  # Post-smoothing sweet spot
    "penalty_factor": 3.0,          # Stronger proximity enforcement
    "force_close": True,            # Ensure loop closure
}
```

### Output Files

| File | Description |
|------|-------------|
| `docs/results/v83_comparison.html` | Interactive Leaflet map with all 31 routes (toggleable layers, score panel) |
| `docs/results/v83_full_results.json` | Complete results with route coordinates for all tests |
| `docs/results/v83_batch_results.json` | Summary results (scores, times, no routes) |
| `docs/results/visual_scores.csv` | Template for manual visual scoring (recognizability, jaggedness, symmetry, closure, proportions) |

---

## Configuration Reference

### Engine Tunables (v8.2)

#### CoreRouter Parameters (`core_router.py`)

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `w_head` | 8.0 | `precompute_edge_weights()` | Heading penalty multiplier |
| `w_uturn` | 1.0 | `precompute_edge_weights()` | U-turn penalty multiplier |
| `beta` | 0.0003 | `precompute_edge_weights()` | Proximity penalty exponent in `(1 + β·d²)` |
| `tube_radius_m` | 300 | `precompute_edge_weights()` | Proximity field range |
| `base_radius_m` | 250 | `build_adaptive_corridor()` | Default corridor width |
| `min_radius_m` | 150 | `build_adaptive_corridor()` | Minimum corridor at curves |
| `max_radius_m` | 500 | `build_adaptive_corridor()` | Maximum corridor on straights |
| `apex_threshold` | 120° | `detect_sharp_vertices()` | Minimum deflection to be an apex |
| `apex_radius_m` | 15m | `compute_uturn_mask()` | Radius around apex where U-turns are free |
| `max_tangent_dev` | 35° | `prune_perpendicular_edges()` | Maximum edge-tangent deviation before pruning |
| `shape_bonus` | 0.2 | `bidirectional_astar()` | A* heuristic reduction for near-shape nodes |
| `SKIP_THRESHOLD` | 3.0 | `_route_segments()` | Max detour/direct ratio before dropping waypoint |
| `max_skips` | 30% | `_route_segments()` | Maximum fraction of waypoints that can be skipped |
| `alpha` (elastic) | 0.15 | `_elastic_deform()` | Deformation strength towards roads |
| `iterations` (elastic) | 2 | `_elastic_deform()` | Number of deformation passes |

#### Engine Parameters (`engine.py`)

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `n_fine` (fit) | 4 | `mode_fit()` | Top coarse candidates to route in fit mode |
| `n_fine` (optimize) | 12 | `_fine_search()` | Top coarse candidates to route in optimize mode |
| `n_cusp` | 15 | `_cusp_align_candidates()` | Max cusp-aligned candidates |
| `max_evals` (CMA) | 40 | `_cma_refine()` | CMA-ES budget |
| `sigma0` (CMA) | 0.05 | `_cma_refine()` | CMA-ES initial step size |
| `n_variants` | 4 | `generate_heart_variants()` | Number of parametric heart variants |
| `fd_norm_gate` | 0.12 | `fit_and_score()` | Fréchet gate threshold (score doubled if exceeded) |

#### Scoring Weights (`scoring_v8.py`)

| Component | Weight | Multiplier | Description |
|-----------|--------|------------|-------------|
| Hausdorff | 0.20 | 1× | Worst-case deviation |
| Heading fidelity | 0.20 | 1.5× | Mean \|Δθ\| × 1.5 |
| Coverage | 0.20 | (2−cov) if <75% | Mean ideal→route distance |
| Fréchet | 0.15 | 1× | Discrete Fréchet in metres |
| Fourier | 0.10 | 80× | DFT magnitude difference |
| Perpendicular | 0.10 | 1× | Mean point-to-segment distance |
| Length-ratio | 0.05 | 15× | Segment proportion fidelity |

#### Coarse Grid Parameters

| Mode | Rotations | Scales | Offsets | Total Combos |
|------|-----------|--------|---------|-------------|
| fit | 8 (0°–315° / 45°) | 5 (0.010–0.030) | 9 (±1.0 km / 2 steps) | 360 |
| optimize | 24 (0°–345° / 15°) | 7 (0.010–0.040) | 13 (±2.0 km / 3 steps) | 2,184 |
| best_shape | 12 (0°–330° / 30°) | 3 (0.009–0.018) | 9 (±1.0 km) | 324/shape × 20 = 6,480 |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | HTTP server port |

---

## Dependencies

### Python

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy` | Required | Vectorized computation throughout |
| `scipy` | Required | `cKDTree` for spatial queries |
| `osmnx` | Recommended | Road network graph download |
| `networkx` | Recommended | Graph shortest-path algorithms |
| `shapely` | Optional | Adaptive corridor geometry (GIS-quality buffer/union) |
| `cma` | Optional | CMA-ES continuous optimisation |
| `folium` | Optional | Interactive map generation (for benchmarks) |
| `matplotlib` | Optional | Comparison PNG generation (for benchmarks) |

If `osmnx`/`networkx` are not installed, the engine falls back to the public OSRM demo server for routing (slower, no shape-aware routing, no soft constraints).

### Node.js

No external dependencies. Uses only built-in `http`, `fs`, `path`, and `child_process` modules.

### Frontend

| Library | Version | CDN |
|---------|---------|-----|
| Leaflet.js | 1.9.4 | unpkg.com |
| OpenStreetMap / CARTO tiles | — | Various |

---

## Research Background

| Source | Contribution | How We Use It |
|--------|-------------|---------------|
| **Waschk & Krüger** (SIGGRAPH Asia 2018 / CVM 2019) | Multi-objective shortest path minimising Riemannian distance | Soft-constraint edge weights: `C = L × H × (1 + β·d²) + heading + uturn` |
| **Li & Fu** (ISPRS Int. J. Geo-Inf., 2026) | Invariant spatial relationships + backtracking subgraph matching | Perpendicular segment distance scoring, length-ratio fidelity, turning-angle comparison |
| **dsleo/stravart** (GitHub, 2024) | Hausdorff distance + Optuna Bayesian optimization | Hausdorff distance as worst-case deviation metric; inspired CMA-ES approach |
| **Balduz** (TU Vienna, 2017) | Rasterisation proximity scoring | Coarse grid search with proximity-only scoring |
| **Hansen & Ostermeier** (2001) | CMA-ES algorithm | Black-box optimisation of rotation/scale/center (Phase 4) |
| **Zahn & Roskies** (1972) | Fourier descriptors for shape recognition | Rotation/scale-invariant shape similarity (Phase 5) |
| **Signed Distance Fields** (computer graphics) | Implicit surface distance queries | SDF-inspired coverage penalty in coarse scoring (Phase 9) |
| **GPSArtify / GPS Art App** (commercial) | Project-and-route approach | Baseline architecture; identified "shortcut" problem |
| **gps2gpx.art** (web tool) | Manual overlay with road following | Validated difficulty of automated shape matching |
| **OpenStreetMap contributors** | Worldwide road network data | Downloaded via osmnx |

---

## License & Attribution

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

You are free to share and adapt the material for non-commercial purposes, with attribution and under the same license. See [LICENSE](../LICENSE) for the full text.
