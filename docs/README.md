# Run2Art — GPS Art Running Route Generator

> Create GPS art running routes from geometric shapes, fitted to real streets. Export as GPX for Strava, Garmin, Komoot.

## Overview

Run2Art transforms geometric shape templates into running routes that follow actual streets, paths, and footpaths. The engine intelligently fits shape outlines onto the road network using advanced shape-aware routing algorithms, producing routes that are both **recognisable** (they look like the intended shape) and **runnable** (they follow real roads).

### Key Features

- **20 optimised shape templates** — Heart, Star, Smiley, Peace Sign, Cat, Diamond, and more, all fine-tuned for road-network fidelity
- **Three fitting modes** — Quick Fit, Auto-Optimize, and Find Best Shape
- **Shape-aware routing** — custom edge weights that penalise deviation from the ideal shape (Waschk & Krüger 2018)
- **Segment-constrained routing** — per-segment local penalty fields prevent route drift
- **Adaptive densification** — tighter control points on curves, looser on straights
- **6-component scoring** — coverage, detour, Hausdorff, perpendicular, turning-angle, length-ratio
- **GPX export** — download fitted routes for use in Strava, Garmin, Komoot, etc.
- **Location search & geolocation** — find any city/town via Nominatim, or use device GPS


## Architecture

```
┌─────────────────────────────────────────────────┐
│  Browser (public/index.html)                    │
│  ├─ Leaflet.js map                              │
│  ├─ Shape template library (20 shapes)          │
│  ├─ Ghost overlay (dashed ideal outline)        │
│  └─ Fitted route display (solid green line)     │
│                                                 │
│  POST /api/match ──▶ JSON payload               │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  server.js (Node.js HTTP server)                │
│  ├─ Static file serving from public/            │
│  └─ /api/match proxy → spawns engine.py         │
└─────────────────┬───────────────────────────────┘
                  │ stdin JSON → stdout JSON
┌─────────────────▼───────────────────────────────┐
│  engine.py (Python geospatial engine v4)        │
│  ├─ Shape ↔ geographic coordinate transform     │
│  ├─ Adaptive densification                      │
│  ├─ Road network graph (osmnx + networkx)       │
│  ├─ Shape-aware & segment-constrained routing   │
│  ├─ 6-component bidirectional scoring           │
│  ├─ Two-step coarse→fine optimisation           │
│  └─ OSRM fallback (no osmnx required)           │
└─────────────────────────────────────────────────┘
```


## File Structure

```
D:\GPS Art\
├── engine.py              # Python geospatial engine (v4, ~950 lines)
├── server.js              # Node.js HTTP server & API proxy (~112 lines)
├── package.json           # Node.js project manifest
├── public/
│   └── index.html         # Single-page frontend (~550 lines)
├── docs/
│   ├── README.md          # This file
│   ├── ENGINE.md          # Engine algorithm documentation
│   ├── SHAPES.md          # Shape template design guide
│   └── API.md             # API reference
└── .venv/                 # Python virtual environment
```


## Quick Start

### Prerequisites

- **Node.js** ≥ 16
- **Python** ≥ 3.9
- **osmnx** + **networkx** (recommended) — or the engine falls back to OSRM

### Installation

```bash
# Clone and enter the project
cd "D:\GPS Art"

# Install Node.js dependencies (none currently required)
npm install

# Set up Python environment
python -m venv .venv
.venv\Scripts\activate
pip install osmnx networkx
```

### Running

```bash
# Start the server
npm start
# or
node server.js

# Open in browser
# → http://localhost:3000
```

### Usage

1. **Set location** — Click "Use My Location" or search for a city/town
2. **Choose a shape** — Click any of the 20 shape cards in the sidebar
3. **Fit to streets** — Choose one of three modes:
   - **⚡ Quick Fit** — Fast single fit with default rotation/scale
   - **🔍 Auto-Optimize** — Two-step coarse→fine search for best rotation, scale, and position
   - **🏆 Find Best Shape** — Tests all 20 shapes to find the best match for the current area
4. **Export** — Click "Export as GPX" to download the route


## Modes of Operation

| Mode | Speed | What It Does |
|------|-------|--------------|
| `fit` | ~30-120s | Single fit with given rotation (0°) and scale (0.012°). Uses adaptive densification + shape-aware routing. |
| `optimize` | ~5-15 min | Two-step: (1) Coarse grid search over 4,200 combos using proximity-only scoring, then (2) Fine routing of top-8 candidates with ±7° rotation, ±12% scale, ±300m offset variations (~600 actual routings). |
| `best_shape` | ~15-30 min | Two-step across ALL 20 shapes: (1) Coarse grid of 324 combos per shape = 6,480 total, then (2) Similarity-boosted selection: top coarse results seed bonus candidates for geometrically similar shapes, then fine routing of up to 18 candidates. |


## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | HTTP server port |

### Engine Tunables (in `engine.py`)

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `PENALTY_FACTOR` (global) | `3.0` | `_precompute_shape_distance_weights()` | How strongly shape deviation penalises edge weight |
| `PENALTY_FACTOR` (segment) | `4.0` | `snap_with_graph_segment_constrained()` | Local per-segment penalty strength |
| `base_spacing` | `120m` | `adaptive_densify()` | Waypoint spacing on straight segments |
| `curve_spacing` | `50m` | `adaptive_densify()` | Waypoint spacing on curves >30° |
| `coverage threshold` | `75%/100m` | `bidirectional_score()` | Minimum shape coverage before penalty |
| `n_fine` | `8` | `fine_search_around()` | Number of coarse candidates to refine |

### Scoring Weights

| Component | Weight | Source |
|-----------|--------|--------|
| Forward coverage (ideal→route) | 0.30 | Shape coverage |
| Reverse detour (route→ideal) | 0.15 | Detour penalty |
| Hausdorff distance | 0.10 | dsleo/stravart |
| Perpendicular segment distance | 0.20 | Li & Fu 2026 |
| Turning-angle fidelity | 0.15 | Li & Fu 2026 |
| Length-ratio fidelity | 0.10 | Li & Fu 2026 |


## Dependencies

### Python

| Package | Required | Purpose |
|---------|----------|---------|
| `osmnx` | Recommended | Road network graph download & nearest-node lookup |
| `networkx` | Recommended | Graph shortest-path algorithms |
| *(stdlib)* | Required | `json`, `sys`, `math`, `urllib.request`, `time` |

If `osmnx`/`networkx` are not installed, the engine falls back to the public OSRM demo server for routing (slower, no shape-aware routing).

### Node.js

No external dependencies. Uses only built-in `http`, `fs`, `path`, and `child_process` modules.

### Frontend

| Library | Version | CDN |
|---------|---------|-----|
| Leaflet.js | 1.9.4 | unpkg.com |
| OpenStreetMap tiles | — | tile.openstreetmap.org |


## Research Background

This engine incorporates techniques from the GPS art research ecosystem:

| Source | Contribution | How We Use It |
|--------|-------------|---------------|
| **Waschk & Krüger** (SIGGRAPH Asia 2018 / CVM 2019) | Multi-objective shortest path minimising Riemannian distance | Shape-aware edge weights: `weight = length + penalty × distance_from_shape` |
| **Li & Fu** (ISPRS Int. J. Geo-Inf., 2026) | Invariant spatial relationships + backtracking subgraph matching | Perpendicular segment distance scoring, length-ratio fidelity, turning-angle comparison |
| **dsleo/stravart** (GitHub, 2024) | Hausdorff distance + Optuna Bayesian optimization | Hausdorff distance as worst-case deviation metric |
| **Balduz** (TU Vienna, 2017) | Rasterisation proximity scoring | Inspired coarse grid search with proximity-only scoring |
| **GPSArtify / GPS Art App** (commercial) | Project-and-route approach | Baseline architecture; identified "shortcut" problem |
| **gps2gpx.art** (web tool) | Manual overlay with road following | Validated difficulty of automated shape matching |


## Algorithm Evolution (v8.2)

The v8.2 engine introduces 10 algorithmic improvements over the v8.1 baseline, systematically benchmarked on Reading, UK:

| Phase | Feature | Impact |
|-------|---------|--------|
| **2** | Curvature-adaptive beta | Edge cost penalty varies with local ideal-line curvature — stronger at cusps/clefts |
| **3** | Cusp-aligned candidates | Snaps heart cusp to nearest road intersection node before fine search |
| **4** | CMA-ES optimisation | Black-box continuous optimisation of rotation/scale/center (40 evaluations) |
| **5** | Fourier descriptor scoring | DFT-based shape similarity metric (rotation/scale invariant) |
| **6** | Multi-variant templates | Parametric heart variants with adjusted cusp/lobe geometry |
| **7** | Iterative worst-segment repair | 3-pass fix cycle with decreasing threshold |
| **8** | Elastic shape deformation | Gently pulls ideal line towards nearby roads for waypoint placement |
| **9** | SDF-inspired coverage penalty | Coarse scoring penalises road network gaps along shape |
| **10** | Multi-start segment routing | Routes from alternative loop start positions for closed shapes |

### Benchmark Results (Heart shape, Reading UK)

| Version | Composite | Frechet | Hausdorff | Coverage | Time |
|---------|-----------|---------|-----------|----------|------|
| v8.1 baseline | 63.6 | 93.9m | 111.5m | 31.0 | 123s |
| v8.2 (all phases) | **50.0** | **62.6m** | **62.8m** | **20.7** | 981s |
| **Improvement** | **-21%** | **-33%** | **-44%** | **-33%** | -- |

CMA-ES discovered a completely different optimal orientation (104.5 deg vs 240 deg), demonstrating that continuous optimisation finds configurations the grid search misses.

See `docs/results/dashboard.html` for the interactive dashboard with comparison maps.


## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

You are free to share and adapt the material for non-commercial purposes, with attribution and under the same license. See [LICENSE](../LICENSE) for the full text.

## Attribution

This project builds on research and open-source work from the GPS art community:

- **Waschk & Krüger** (2018/2019) — Multi-objective shortest-path routing for shape fidelity
- **Li & Fu** (2026) — Perpendicular distance scoring, turning-angle and length-ratio metrics
- **dsleo/stravart** (2024) — Hausdorff distance evaluation, open-source GPS art tools
- **Balduz** (2017) — Rasterisation proximity scoring for fast screening
- **OpenStreetMap contributors** — Road network data via osmnx
- **Leaflet.js** — Interactive map visualisation
- **OSRM** — Fallback routing engine
