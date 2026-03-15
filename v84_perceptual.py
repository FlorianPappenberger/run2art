"""
v84_perceptual.py — v8.4 "Perceptual & Road-Aware" Layer
==========================================================
Sits on top of v8.3 enhancements. Every feature is an optional flag
that can be toggled via the config dict passed through fit_and_score_v83.

New features (all optional):
  1. Road-Density Auto-Scaling          (density_auto_scale)
  2. Road-Hierarchy Bonus               (use_road_hierarchy)
  3. Medial-Axis / Skeleton Score       (use_skeleton_score)
  4. Fused Gromov-Wasserstein Matching  (use_fgw)
  5. Perceptual Loss (MobileNet)        (use_perceptual_loss)
  6. Persistent Homology Topology Score (use_ph_topology)

Each function returns a float score (lower = better) or transforms
shape points in-place.  All heavy imports are deferred so missing
packages only cause a warning, never a crash.
"""

import math
import sys
import numpy as np
from scipy.spatial import cKDTree

# ── project imports ──
from geometry import (
    haversine, haversine_vector, haversine_matrix,
    sample_polyline, shape_to_latlngs,
)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  1. ROAD-DENSITY AUTO-SCALING  (flag: density_auto_scale)
# ═══════════════════════════════════════════════════════════════════════════

def compute_local_density(center, G, radius_m=800):
    """Count road nodes per km² around *center* [lat, lng]."""
    nodes = list(G.nodes(data=True))
    coords = np.array([[n['y'], n['x']] for _, n in nodes], dtype=np.float64)
    dists = haversine_vector(coords[:, 0], coords[:, 1], center[0], center[1])
    count = int((dists <= radius_m).sum())
    area_km2 = math.pi * (radius_m / 1000.0) ** 2
    return count / max(area_km2, 0.01)


def find_best_density_pocket(G, center, max_move_m=500, grid_step_m=100):
    """Grid-search for the highest-density pocket within *max_move_m*."""
    best_center = list(center)
    best_density = compute_local_density(center, G)

    step_deg_lat = grid_step_m * 9e-6          # ~9e-6 deg per metre lat
    step_deg_lng = grid_step_m * 1.4e-5        # ~1.4e-5 deg per metre lng
    n_steps = int(max_move_m / grid_step_m)

    for di in range(-n_steps, n_steps + 1):
        for dj in range(-n_steps, n_steps + 1):
            c = [center[0] + di * step_deg_lat,
                 center[1] + dj * step_deg_lng]
            dist = haversine(center[0], center[1], c[0], c[1])
            if dist > max_move_m:
                continue
            d = compute_local_density(c, G)
            if d > best_density:
                best_density = d
                best_center = c

    return best_center, best_density


def density_auto_scale(shape_pts, center, G,
                       target_density=800, max_move_m=500):
    """If road density at *center* is below *target_density* nodes/km²,
    translate the template centre toward the nearest high-density pocket.

    Returns (new_center, density) — shape_pts are NOT mutated.
    """
    density = compute_local_density(center, G)
    if density >= target_density:
        log(f"[v8.4-density] Density {density:.0f} ≥ {target_density} — no shift")
        return center, density

    new_center, new_density = find_best_density_pocket(
        G, center, max_move_m=max_move_m)
    shift_m = haversine(center[0], center[1], new_center[0], new_center[1])
    log(f"[v8.4-density] {density:.0f} → {new_density:.0f} nodes/km² "
        f"(shifted {shift_m:.0f}m)")
    return new_center, new_density


# ═══════════════════════════════════════════════════════════════════════════
#  2. ROAD-HIERARCHY BONUS  (flag: use_road_hierarchy)
# ═══════════════════════════════════════════════════════════════════════════

# Multipliers applied *on top of* the existing HIGHWAY_MULTIPLIERS in
# core_router.  A value < 1 rewards the edge; > 1 penalises it.
HIERARCHY_BONUS = {
    'footway':       0.75,
    'path':          0.75,
    'pedestrian':    0.75,
    'cycleway':      0.80,
    'steps':         0.85,
    'track':         0.80,
    'living_street': 0.90,
    'residential':   0.85,
    'unclassified':  1.00,
    'service':       1.00,
    'tertiary':      1.10,
    'tertiary_link': 1.10,
    'secondary':     1.20,
    'secondary_link':1.20,
    'primary':       1.30,
    'primary_link':  1.30,
    'trunk':         1.50,
    'trunk_link':    1.50,
    'motorway':      5.00,
    'motorway_link': 5.00,
}


def apply_road_hierarchy_bonus(G_sub):
    """Multiply every edge's 'v8w' by a hierarchy bonus factor.

    Must be called *after* precompute_edge_weights has set 'v8w'.
    Modifies G_sub in-place.
    """
    n_modified = 0
    for u, v, key, data in G_sub.edges(data=True, keys=True):
        hw = data.get('highway', 'unclassified')
        if isinstance(hw, list):
            hw = hw[0]
        factor = HIERARCHY_BONUS.get(hw, 1.0)
        if factor != 1.0:
            old_w = data.get('v8w', data.get('length', 50.0))
            G_sub[u][v][key]['v8w'] = old_w * factor
            n_modified += 1
    if n_modified:
        log(f"[v8.4-hierarchy] Adjusted {n_modified} edges")


# ═══════════════════════════════════════════════════════════════════════════
#  3. MEDIAL-AXIS / SKELETON SCORE  (flag: use_skeleton_score)
# ═══════════════════════════════════════════════════════════════════════════

_SKEL_AVAILABLE = None          # lazy probe


def _check_skeleton_deps():
    global _SKEL_AVAILABLE
    if _SKEL_AVAILABLE is not None:
        return _SKEL_AVAILABLE
    try:
        from skimage.morphology import skeletonize    # noqa: F401  # type: ignore[import-untyped]
        from scipy.spatial.distance import directed_hausdorff  # noqa: F401
        _SKEL_AVAILABLE = True
    except ImportError:
        log("[v8.4-skel] scikit-image not available — skeleton score disabled")
        _SKEL_AVAILABLE = False
    return _SKEL_AVAILABLE


def _rasterize_polyline(pts, img_size=128):
    """Rasterize [lat,lng] polyline to a binary image."""
    from PIL import Image, ImageDraw

    pa = np.asarray(pts, dtype=np.float64)
    mn = pa.min(axis=0)
    mx = pa.max(axis=0)
    span = max(mx[0] - mn[0], mx[1] - mn[1])
    if span < 1e-9:
        return np.zeros((img_size, img_size), dtype=bool)

    margin = 0.05
    scaled = (pa - mn) / span * (1 - 2 * margin) + margin
    pixels = (scaled * (img_size - 1)).astype(int)

    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    coords = [(int(p[1]), int(p[0])) for p in pixels]  # (x, y)
    if len(coords) >= 2:
        draw.line(coords, fill=255, width=2)
    return np.array(img) > 0


def _extract_skeleton(pts, img_size=128):
    """Return skeleton pixel coordinates from a polyline."""
    from skimage.morphology import skeletonize  # type: ignore[import-untyped]
    bimg = _rasterize_polyline(pts, img_size)
    skel = skeletonize(bimg)
    ys, xs = np.where(skel)
    if len(ys) == 0:
        return np.empty((0, 2))
    return np.column_stack([ys, xs]).astype(np.float64) / img_size


def skeleton_score(route, ideal_pts, img_size=128, weight=1.0):
    """Hausdorff distance between skeletons of route and ideal shape.

    Returns float penalty (lower = better).  0 if deps missing.
    """
    if not _check_skeleton_deps():
        return 0.0
    from scipy.spatial.distance import directed_hausdorff

    skel_route = _extract_skeleton(route, img_size)
    skel_ideal = _extract_skeleton(ideal_pts, img_size)
    if len(skel_route) < 2 or len(skel_ideal) < 2:
        return 0.0

    fwd = directed_hausdorff(skel_route, skel_ideal)[0]
    bwd = directed_hausdorff(skel_ideal, skel_route)[0]
    raw = (fwd + bwd) / 2.0
    # Normalise to ~0-100 range (skeleton coords are in [0,1])
    return float(raw * 500.0 * weight)


# ─── pre-compute ideal skeleton once ───
_IDEAL_SKEL_CACHE = {}


def get_ideal_skeleton(ideal_pts, img_size=128):
    key = (len(ideal_pts), img_size)
    if key not in _IDEAL_SKEL_CACHE:
        if _check_skeleton_deps():
            _IDEAL_SKEL_CACHE[key] = _extract_skeleton(ideal_pts, img_size)
        else:
            _IDEAL_SKEL_CACHE[key] = np.empty((0, 2))
    return _IDEAL_SKEL_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════════
#  4. FUSED GROMOV-WASSERSTEIN  (flag: use_fgw)
# ═══════════════════════════════════════════════════════════════════════════

_OT_AVAILABLE = None


def _check_ot_deps():
    global _OT_AVAILABLE
    if _OT_AVAILABLE is not None:
        return _OT_AVAILABLE
    try:
        import ot   # noqa: F401  # type: ignore[import-not-found]
        _OT_AVAILABLE = True
    except Exception:
        log("[v8.4-fgw] POT library not available — FGW score disabled")
        _OT_AVAILABLE = False
    return _OT_AVAILABLE


def _build_internal_distance_matrix(pts):
    """Geodesic-like internal cost matrix (pairwise haversine)."""
    pa = np.asarray(pts, dtype=np.float64)
    n = len(pa)
    if n > 120:
        # Downsample to keep cost matrix tractable
        idx = np.linspace(0, n - 1, 120).astype(int)
        pa = pa[idx]
        n = len(pa)
    return haversine_matrix(pa[:, 0], pa[:, 1], pa[:, 0], pa[:, 1])


def fgw_score(route, ideal_pts, alpha=0.5, weight=1.0):
    """Fused Gromov-Wasserstein distance between route and ideal.

    Uses: geometry (Wasserstein) + internal structure (Gromov).
    Returns float penalty (lower = better).  0 if POT missing.
    """
    if not _check_ot_deps():
        return 0.0
    import ot  # type: ignore[import-not-found]

    # Sub-sample both to same size
    n_sample = 80
    r_pts = np.asarray(sample_polyline(route, n_sample), dtype=np.float64)
    i_pts = np.asarray(sample_polyline(ideal_pts, n_sample), dtype=np.float64)

    # Cost matrices (geometry: pairwise haversine between route ↔ ideal)
    M = haversine_matrix(r_pts[:, 0], r_pts[:, 1],
                         i_pts[:, 0], i_pts[:, 1])
    M = M / max(M.max(), 1e-9)         # normalise to [0,1]

    # Internal structure matrices
    C1 = _build_internal_distance_matrix(r_pts)
    C2 = _build_internal_distance_matrix(i_pts)
    C1 = C1 / max(C1.max(), 1e-9)
    C2 = C2 / max(C2.max(), 1e-9)

    p = np.ones(len(r_pts)) / len(r_pts)
    q = np.ones(len(i_pts)) / len(i_pts)

    try:
        fgw, fgw_log = ot.gromov.fused_gromov_wasserstein2(
            M, C1, C2, p, q, alpha=alpha, log=True)
        val = float(fgw)
    except Exception as e:
        log(f"[v8.4-fgw] Computation failed: {e}")
        val = 0.0

    return val * 200.0 * weight   # Scale to ~0-100 range


# ═══════════════════════════════════════════════════════════════════════════
#  5. PERCEPTUAL LOSS  (flag: use_perceptual_loss)
# ═══════════════════════════════════════════════════════════════════════════

_PERCEPTUAL_MODEL = None
_PERCEPTUAL_AVAILABLE = None
_IDEAL_HEART_EMB = None


def _check_perceptual_deps():
    global _PERCEPTUAL_AVAILABLE
    if _PERCEPTUAL_AVAILABLE is not None:
        return _PERCEPTUAL_AVAILABLE
    try:
        import torch                              # noqa: F401  # type: ignore[import-not-found]
        from torchvision.models import mobilenet_v3_small  # noqa: F401  # type: ignore[import-not-found]
        from torchvision import transforms        # noqa: F401  # type: ignore[import-not-found]
        _PERCEPTUAL_AVAILABLE = True
    except Exception:
        log("[v8.4-percep] torch/torchvision not available — "
            "perceptual loss disabled")
        _PERCEPTUAL_AVAILABLE = False
    return _PERCEPTUAL_AVAILABLE


def _get_perceptual_model():
    """Lazy-load and freeze MobileNetV3-Small."""
    global _PERCEPTUAL_MODEL
    if _PERCEPTUAL_MODEL is not None:
        return _PERCEPTUAL_MODEL
    import torch  # type: ignore[import-not-found]
    from torchvision.models import mobilenet_v3_small  # type: ignore[import-not-found]
    _PERCEPTUAL_MODEL = mobilenet_v3_small(
        weights='DEFAULT').eval()
    for p in _PERCEPTUAL_MODEL.parameters():
        p.requires_grad = False
    log("[v8.4-percep] MobileNetV3-Small loaded")
    return _PERCEPTUAL_MODEL


def _render_route_image(pts, img_size=224):
    """Render [lat,lng] polyline as a white-on-black PIL image."""
    from PIL import Image, ImageDraw

    pa = np.asarray(pts, dtype=np.float64)
    mn = pa.min(axis=0)
    mx = pa.max(axis=0)
    span = max(mx[0] - mn[0], mx[1] - mn[1])
    if span < 1e-9:
        return Image.new('RGB', (img_size, img_size), (0, 0, 0))

    margin = 0.08
    scaled = (pa - mn) / span * (1 - 2 * margin) + margin
    pixels = (scaled * (img_size - 1)).astype(int)

    img = Image.new('RGB', (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    coords = [(int(p[1]), int(p[0])) for p in pixels]
    if len(coords) >= 2:
        draw.line(coords, fill=(255, 255, 255), width=3)
    return img


def _image_to_embedding(img):
    """Run image through MobileNet, return embedding tensor."""
    import torch  # type: ignore[import-not-found]
    from torchvision import transforms  # type: ignore[import-not-found]

    preproc = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = preproc(img).unsqueeze(0)
    model = _get_perceptual_model()
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze(0)


def _get_ideal_heart_embedding(ideal_pts):
    """Compute (and cache) the ideal heart shape embedding."""
    global _IDEAL_HEART_EMB
    if _IDEAL_HEART_EMB is not None:
        return _IDEAL_HEART_EMB
    img = _render_route_image(ideal_pts)
    _IDEAL_HEART_EMB = _image_to_embedding(img)
    return _IDEAL_HEART_EMB


def perceptual_heart_score(route, ideal_pts, weight=1.0):
    """Cosine distance between MobileNet embeddings of route vs ideal heart.

    Returns float penalty in [0, ~200].  0 if deps missing.
    """
    if not _check_perceptual_deps():
        return 0.0
    import torch  # type: ignore[import-not-found]

    route_img = _render_route_image(route)
    route_emb = _image_to_embedding(route_img)
    ideal_emb = _get_ideal_heart_embedding(ideal_pts)

    cos_sim = torch.nn.functional.cosine_similarity(
        route_emb.unsqueeze(0), ideal_emb.unsqueeze(0)).item()
    # cos_sim ∈ [-1,1]; convert to penalty: 0 = identical, 200 = opposite
    penalty = (1.0 - cos_sim) * 100.0
    return float(penalty * weight)


# ═══════════════════════════════════════════════════════════════════════════
#  6. PERSISTENT HOMOLOGY TOPOLOGY SCORE  (flag: use_ph_topology)
# ═══════════════════════════════════════════════════════════════════════════

_PH_AVAILABLE = None


def _check_ph_deps():
    global _PH_AVAILABLE
    if _PH_AVAILABLE is not None:
        return _PH_AVAILABLE
    try:
        import gudhi   # noqa: F401  # type: ignore[import-not-found]
        _PH_AVAILABLE = True
    except ImportError:
        log("[v8.4-ph] gudhi not available — PH topology score disabled")
        _PH_AVAILABLE = False
    return _PH_AVAILABLE


def _compute_persistence_diagram(pts, max_edge_m=300):
    """Compute persistence diagram (H0 + H1) from point cloud.

    Uses Rips complex on normalised coordinates.
    """
    import gudhi  # type: ignore[import-not-found]

    pa = np.asarray(pts, dtype=np.float64)
    # Sub-sample for speed
    if len(pa) > 150:
        idx = np.linspace(0, len(pa) - 1, 150).astype(int)
        pa = pa[idx]

    # Normalise to unit box
    mn = pa.min(axis=0)
    mx = pa.max(axis=0)
    span = max(mx[0] - mn[0], mx[1] - mn[1])
    if span < 1e-9:
        return []
    norm = (pa - mn) / span

    rips = gudhi.RipsComplex(points=norm.tolist(), max_edge_length=0.5)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()
    return simplex_tree.persistence()


def _persistence_to_arrays(diag, dim):
    """Extract (birth, death) arrays for a specific dimension."""
    pairs = [(b, d) for (dm, (b, d)) in diag
             if dm == dim and d != float('inf')]
    if not pairs:
        return np.empty((0, 2))
    return np.array(pairs, dtype=np.float64)


def ph_topology_score(route, ideal_pts, weight=1.0):
    """Bottleneck distance between persistence diagrams of route and ideal.

    Focuses on H1 (1-dimensional holes = loops / lobes).
    Returns float penalty (lower = better).  0 if gudhi missing.
    """
    if not _check_ph_deps():
        return 0.0
    import gudhi  # type: ignore[import-not-found]

    diag_route = _compute_persistence_diagram(route)
    diag_ideal = _compute_persistence_diagram(ideal_pts)

    h1_route = _persistence_to_arrays(diag_route, dim=1)
    h1_ideal = _persistence_to_arrays(diag_ideal, dim=1)

    # Count topological features
    n_loops_route = len(h1_route)
    n_loops_ideal = len(h1_ideal)

    # Bottleneck distance on H1
    try:
        bd = gudhi.bottleneck_distance(
            [(b, d) for b, d in h1_route] if len(h1_route) > 0 else [],
            [(b, d) for b, d in h1_ideal] if len(h1_ideal) > 0 else [],
        )
    except Exception:
        bd = abs(n_loops_route - n_loops_ideal) * 0.3

    # Missing-loop penalty: hearts should have ~1 main loop; if the route
    # has 0 loops (open, not closed) or 2+ loops (self-intersection), penalise.
    loop_penalty = abs(n_loops_route - n_loops_ideal) * 20.0

    raw = float(bd) * 200.0 + loop_penalty
    log(f"[v8.4-ph] loops: route={n_loops_route} ideal={n_loops_ideal} "
        f"bottleneck={bd:.4f} penalty={raw:.1f}")
    return raw * weight


# ═══════════════════════════════════════════════════════════════════════════
#  COMPOSITE v8.4 SCORER
# ═══════════════════════════════════════════════════════════════════════════

def score_v84(route, ideal_pts, config, G=None, kdtree_data=None):
    """Compute all enabled v8.4 perceptual scores.

    Config keys (all optional, float weight 0.0–1.0):
        skeleton_weight:    weight for skeleton score
        fgw_weight:         weight for FGW score
        perceptual_weight:  weight for perceptual loss
        ph_weight:          weight for persistent homology

    Returns dict:
        { 'skeleton': float, 'fgw': float, 'perceptual': float,
          'ph': float, 'v84_total': float }
    """
    results = {
        'skeleton': 0.0,
        'fgw': 0.0,
        'perceptual': 0.0,
        'ph': 0.0,
        'v84_total': 0.0,
    }

    sk_w = config.get('skeleton_weight', 0.0)
    fgw_w = config.get('fgw_weight', 0.0)
    perc_w = config.get('perceptual_weight', 0.0)
    ph_w = config.get('ph_weight', 0.0)

    if sk_w > 0 and config.get('use_skeleton_score', False):
        results['skeleton'] = skeleton_score(route, ideal_pts, weight=sk_w)

    if fgw_w > 0 and config.get('use_fgw', False):
        results['fgw'] = fgw_score(route, ideal_pts, weight=fgw_w)

    if perc_w > 0 and config.get('use_perceptual_loss', False):
        results['perceptual'] = perceptual_heart_score(
            route, ideal_pts, weight=perc_w)

    if ph_w > 0 and config.get('use_ph_topology', False):
        results['ph'] = ph_topology_score(route, ideal_pts, weight=ph_w)

    results['v84_total'] = (results['skeleton'] + results['fgw'] +
                            results['perceptual'] + results['ph'])
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  ROAD-DENSITY PRE-ROUTING HOOK
# ═══════════════════════════════════════════════════════════════════════════

def maybe_auto_scale_center(center, shape_pts, G, config):
    """If density_auto_scale is enabled, shift center to a denser area.

    Returns (possibly shifted) center.
    """
    if not config.get('density_auto_scale', False):
        return center
    target = config.get('density_target', 800)
    max_move = config.get('density_max_move_m', 500)
    new_center, _ = density_auto_scale(
        shape_pts, center, G,
        target_density=target, max_move_m=max_move)
    return new_center
