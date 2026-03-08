# Run2Art v8.0 — Architectural Plan
## "No-Trunk" Performance-First Refactor

---

## 1. ROOT CAUSE ANALYSIS: WHY TRUNKS EXIST

### The Problem
Current v7.0 creates 'trunks' (jagged spikes protruding from the shape) because:

1. **Hard Waypoint Constraints**: `route_with_anchors()` forces the route through 
   specific graph nodes nearest to ideal waypoints. When the nearest walkable road 
   diverges from the template, the router creates a detour spike to reach that node 
   and then returns — forming a "trunk."

2. **No Heading Penalty**: The cost function `make_pedestrian_weight_fn()` penalizes 
   distance-from-ideal and road-type, but **never checks if the edge direction matches 
   the local shape tangent**. A road running perpendicular to the heart outline gets 
   penalized only by proximity, not by heading deviation.

3. **Greedy Anchor Infill**: When `route_with_anchors()` detects deviation > 80m, 
   it inserts corrective waypoints from the ideal line. These insertions force 
   out-and-back spikes when no through-road exists along the shape.

4. **No U-Turn Control**: The router freely allows 180° reversals anywhere. On a 
   heart shape, the only valid U-turn location is the bottom apex (and arguably 
   the top dip). All other U-turns produce visible trunks.

### The Solution
Replace **hard waypoint routing** with a **soft-constraint field** where every edge 
in the graph carries a cost that encodes:
- Distance to the template center-line (proximity field)
- Heading alignment with the local tangent (heading penalty)  
- U-turn legality (apex sensitivity mask)
- Road suitability for runners (pedestrian preference)

---

## 2. MODULE ARCHITECTURE (v8.0)

```
engine.py          — Mode orchestrators (unchanged API contract)
  ├── core_router.py   — NEW: Soft-constraint CoreRouter class
  ├── scoring_v8.py    — NEW: Fréchet-primary ScoringEngine class  
  ├── routing.py       — MODIFIED: Bidirectional A*, cycle basis
  ├── geometry.py      — EXTENDED: Tangent vector field, apex detection
  └── scoring.py       — PRESERVED: Legacy scorer (fallback)
```

---

## 3. COMPONENT DESIGN

### 3.1 CoreRouter (core_router.py)

**Purpose**: Single class that encapsulates the entire routing pipeline with 
soft-constraint edge weighting.

```
CoreRouter(G, ideal_line, sharp_vertices, config)
  .route() → List[[lat, lng]]
  .score() → float (Fréchet distance, normalized)
```

#### 3.1.1 Soft-Constraint Edge Weight Function

For every edge (u, v) in the corridor subgraph:

```
C(u,v) = w_net · C_network 
       + w_prox · C_proximity 
       + w_head · C_heading 
       + w_uturn · C_uturn
       + w_bio · C_biomech

Where:
  C_network   = edge_length × highway_multiplier  (pedestrian preference)
  C_proximity = perpendicular_distance_to_centerline(midpoint(u,v))
  C_heading   = heading_deviation_penalty(bearing(u,v), local_tangent)
  C_uturn     = uturn_penalty(u, v, prev_edge)  (∞ unless near apex)
  C_biomech   = surface_gradient_penalty(elevation_change)

Default weights:
  w_net  = 1.0
  w_prox = 12.0   (strong pull toward centerline)
  w_head = 8.0    (penalize perpendicular roads)
  w_uturn = 1.0   (pass-through; C_uturn handles magnitude)
  w_bio  = 2.0
```

#### 3.1.2 Heading Penalty (eliminates perpendicular detours)

```python
def heading_penalty(edge_bearing, local_tangent_bearing):
    """
    Cost multiplier based on angular deviation from shape tangent.
    
    deviation = |edge_bearing - local_tangent_bearing|  (mod 180°, since
                we allow travel in either direction along the shape)
    
    If deviation < 20°:  penalty = 0          (aligned — no cost)
    If deviation 20-45°: penalty = (dev-20)/25 × 100m  (linear ramp)
    If deviation > 45°:  penalty = 100 + (dev-45)² × 2  (quadratic) 
    If deviation > 70°:  penalty = 500m       (near-perpendicular — heavy)
    """
```

The local tangent is precomputed for every point along the ideal line using 
vectorized `np.arctan2` on consecutive differences.

#### 3.1.3 Apex Sensitivity (U-Turn Control)

```python
def detect_sharp_vertices(ideal_line, angle_threshold=120):
    """
    Find points where the template has sharp turns (> 120° deflection).
    For a heart: bottom point (~180° turn) and top-center dip (~140°).
    
    Returns: List[(lat, lng, allowed_uturn_radius_m)]
      - Bottom apex: 15m radius (tight U-turn allowed)
      - Top dip: 15m radius  
      - All other points: U-turns cost +2000m (effectively banned)
    """
```

During routing, a 180° turn at edge (u,v) is:
- **Free** if midpoint(u,v) is within 15m of a sharp vertex
- **+2000m penalty** otherwise (forces the router to find an alternative)

#### 3.1.4 Vectorized Edge-Weight Precomputation (NumPy)

**Key optimization**: Instead of computing weights lazily per-edge during A*, 
precompute all edge weights in one vectorized pass:

```python
def precompute_edge_weights(G_sub, ideal_line, tangent_field, apex_points):
    """
    Vectorized weight assignment for all edges in the corridor subgraph.
    
    1. Extract all edge midpoints as (E, 2) NumPy array
    2. Compute all perpendicular distances: vectorized (E,) array
    3. Compute all headings vs tangents: vectorized (E,) array  
    4. Compute all apex distances: vectorized (E,) array
    5. Combine into final weight: vectorized (E,) scalar
    6. Assign nx.set_edge_attributes(G_sub, weights, 'v8_weight')
    
    Result: A* uses string attribute 'v8_weight' — no Python callback.
    This removes ALL per-edge Python function calls during graph search.
    """
```

Estimated speedup: **5-10×** on corridor routing (eliminates ~500-1000 
Python function calls per A* invocation).

### 3.2 ScoringEngine (scoring_v8.py)

**Primary metric**: Discrete Fréchet Distance (the "dog-walker" metric)

#### 3.2.1 Discrete Fréchet Distance

```python
def discrete_frechet(P, Q):
    """
    The discrete Fréchet distance between polylines P and Q.
    
    Measures the minimum "leash length" needed for a dog-walker:
    - Person walks along P (only forward)
    - Dog walks along Q (only forward)
    - Neither can go backward
    - What's the shortest leash that allows both to complete their walk?
    
    Algorithm: O(n*m) dynamic programming
      dp[i][j] = max(dist(P[i], Q[j]), min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))
    
    Implementation: NumPy-accelerated with rolling-minimum optimization.
    """
```

#### 3.2.2 Scale-Invariant Normalization

Before computing Fréchet distance, both the blueprint and route are 
normalized to a 1×1 bounding box:

```python
def normalize_to_unit_box(polyline):
    """
    Translate and scale polyline so bounding box is [0,1] × [0,1].
    Preserves aspect ratio by scaling uniformly (max of width/height).
    
    This ensures a 500m heart and a 2km heart are scored identically
    for shape fidelity — size doesn't affect recognizability.
    """
```

#### 3.2.3 Composite Score (v8)

```
score_v8 = 0.40 × fréchet_normalized      (shape flow — primary)
         + 0.20 × coverage_penalty         (did we hit all parts?)
         + 0.15 × perpendicular_mean       (how close to center-line?)
         + 0.15 × heading_fidelity         (do turns match template?)  
         + 0.10 × length_ratio             (proportional fidelity)

Where:
  fréchet_normalized = discrete_frechet(normalize(route), normalize(ideal))
                       × diagonal_meters / √2
  
  coverage_penalty = same as v7 (forward ideal→route coverage)
  perpendicular_mean = same as v7 (vectorized segment distances)
  heading_fidelity = mean |Δθ| between route and ideal turning angles
  length_ratio = same as v7
```

The Hausdorff component (v7: 10%) is **removed** — Fréchet subsumes it 
(Fréchet ≥ Hausdorff always, and captures sequential flow that Hausdorff misses).

The reverse-detour component (v7: 15%) is **absorbed** into the Fréchet metric 
(the "dog-walker" naturally penalizes route wandering).

### 3.3 Graph Theory Refactor (routing.py modifications)

#### 3.3.1 Bidirectional A* with Shape Heuristic

Replace `nx.astar_path` with a custom bidirectional A*:

```python
def bidirectional_astar(G, source, target, weight_attr, ideal_line):
    """
    Bidirectional A* that searches from both source and target simultaneously.
    
    Forward heuristic:  h_f(n) = haversine(n, target) × 0.8  (admissible)
    Backward heuristic: h_b(n) = haversine(n, source) × 0.8
    
    Shape-following boost: If node n is within 50m of ideal_line, 
    reduce heuristic by 20% to prefer shape-adjacent nodes.
    
    Termination: When forward and backward frontiers meet.
    
    Expected pruning: ~50% of nodes explored vs unidirectional A*.
    """
```

NetworkX provides `nx.bidirectional_dijkstra` but not bidirectional A*.
We implement a custom version using heapq with two open sets.

#### 3.3.2 Minimum Cycle Basis (for closed shapes)

For closed shapes (heart, circle, etc.), find optimal closed loops:

```python
def find_shape_cycle(G_sub, ideal_line, start_node):
    """
    Find the best closed cycle in the corridor that approximates the shape.
    
    Strategy:
    1. Compute minimum cycle basis: nx.minimum_cycle_basis(G_undirected)
    2. Score each cycle against ideal_line using Fréchet distance  
    3. Return the cycle with lowest Fréchet distance
    
    For large graphs, use edge-disjoint path pairs instead:
    1. Pick start_node (nearest to ideal[0])
    2. Pick opposite_node (nearest to ideal[n//2])
    3. Find 2 edge-disjoint paths: start→opposite (clockwise, counter-clockwise)  
    4. Concatenate: path_cw + reverse(path_ccw) = closed loop
    
    This runs in O(V·E) vs O(2^E) for full cycle enumeration.
    """
```

### 3.4 Adaptive Wide-Tube Buffer (routing.py)

Replace static `base_radius_m=200` with adaptive logic:

```python
def adaptive_tube_radius(G, ideal_line):
    """
    Compute optimal tube width based on local road density AND shape curvature.
    
    Rules:
      - High curvature regions (heart apex): NARROW tube (200m)
        → Prevents cutting corners
      - Straight regions: WIDE tube (500m)  
        → More road options, better path quality
      - Low density areas: WIDER base (300-500m)
        → Ensure connectivity
      - High density areas: NARROWER base (200-350m)
        → Faster search, sufficient options
    
    Returns: List[(lat, lng, radius_m)] — per-point adaptive radius
    """
```

The tube becomes a **variable-width corridor** — narrow at curves, wide on straights.
GeoPandas implementation: series of circles with varying radii, `unary_union` merged.

---

## 4. IMPLEMENTATION SEQUENCE

### Phase 1: Core Infrastructure
1. `geometry.py` — Add tangent_field(), detect_sharp_vertices(), normalize_to_unit_box()
2. `scoring_v8.py` — Discrete Fréchet + composite scorer
3. `core_router.py` — Soft-constraint weight precomputation + CoreRouter class

### Phase 2: Graph Optimizations  
4. `routing.py` — Bidirectional A*, adaptive wide-tube, cycle basis
5. `core_router.py` — Integrate bidirectional A* and cycle-based routing

### Phase 3: Integration
6. `engine.py` — Wire CoreRouter + ScoringEngine into mode_fit/optimize
7. Benchmark against v5.1, v6.0, v7.0

---

## 5. ESTIMATED PERFORMANCE

| Metric | v7.0 | v8.0 (projected) |
|--------|------|-------------------|
| Fit mode (heart) | 5-8s | 2-4s |
| Optimize mode (heart) | 30-60s | 15-30s |
| Fréchet score (heart) | N/A | ~25m (normalized) |
| Trunk occurrences | frequent | eliminated |
| Edge weight computation | per-call Python | precomputed NumPy |
| A* nodes explored | ~500 | ~250 (bidirectional) |
| Corridor width | static 200m | adaptive 200-500m |

---

## 6. BACKWARD COMPATIBILITY

- `mode_fit()`, `mode_optimize()`, `mode_best_shape()` retain identical JSON API
- `bidirectional_score()` preserved as `scoring.py` (legacy fallback)
- Existing corridor/KDTree infrastructure reused
- Cache format unchanged (same graph pkl files)
