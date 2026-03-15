"""v86_open_experiments.py - Open-space heart search experiments.

Provides alternate heart blueprints, raster-contour extraction, and a
lightweight genetic search over route placement parameters.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry import sample_polyline
from v83_enhancements import fit_and_score_v83


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _normalize_closed(points):
    pa = np.asarray(points, dtype=np.float64)
    if len(pa) < 3:
        return pa.tolist()

    mn = pa.min(axis=0)
    mx = pa.max(axis=0)
    span = np.maximum(mx - mn, 1e-9)
    scale = max(span[0], span[1])
    centered = (pa - mn) / scale
    centered[:, 0] += (1.0 - (span[0] / scale)) * 0.5
    centered[:, 1] += (1.0 - (span[1] / scale)) * 0.5

    if np.linalg.norm(centered[0] - centered[-1]) > 1e-9:
        centered = np.vstack([centered, centered[0]])
    return centered.tolist()


def _resample_closed(points, n=96):
    sampled = sample_polyline(points, n)
    if not sampled:
        return points
    if sampled[0] != sampled[-1]:
        sampled.append(sampled[0])
    return sampled


def _valentine_points(n=160):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return _normalize_closed(np.column_stack([x, y]))


def _cardioid_points(n=160):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 - np.sin(t)
    x = r * np.cos(t)
    y = -1.15 * r * np.sin(t)
    return _normalize_closed(np.column_stack([x, y]))


def _taubin_like_points(n=160):
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    x = np.sin(t)
    y = -(np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** 0.65) / (np.abs(np.sin(t)) + 1.35))
    y += 0.18 * np.cos(2 * t)
    return _normalize_closed(np.column_stack([x, y]))


def _lowpoly_points():
    pts = [
        [0.50, 0.30], [0.62, 0.12], [0.82, 0.08], [0.95, 0.18], [0.98, 0.34],
        [0.90, 0.52], [0.73, 0.70], [0.58, 0.87], [0.50, 0.98], [0.42, 0.87],
        [0.27, 0.70], [0.10, 0.52], [0.02, 0.34], [0.05, 0.18], [0.18, 0.08],
        [0.38, 0.12], [0.50, 0.30],
    ]
    return _normalize_closed(pts)


def _raster_boundary_points(kind, img_size=192, bins=144):
    from PIL import Image, ImageDraw

    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)

    if kind == 'simple_png_contour':
        draw.ellipse((20, 18, 92, 94), fill=255)
        draw.ellipse((100, 18, 172, 94), fill=255)
        draw.polygon([(30, 68), (96, 170), (162, 68), (96, 36)], fill=255)
    elif kind == 'emoji_contour':
        draw.ellipse((18, 20, 94, 96), fill=255)
        draw.ellipse((98, 20, 174, 96), fill=255)
        draw.polygon([(26, 74), (96, 176), (166, 74), (96, 48)], fill=255)
    elif kind == 'handdrawn_contour':
        pts = [
            (96, 54), (130, 18), (164, 24), (178, 62), (165, 95),
            (142, 122), (120, 146), (96, 176), (72, 146), (50, 122),
            (27, 95), (14, 62), (28, 24), (62, 18), (96, 54),
        ]
        draw.line(pts, fill=255, width=16, joint='curve')
    else:
        raise ValueError(f'Unknown raster contour kind: {kind}')

    arr = np.array(img, dtype=np.uint8) > 0
    if not arr.any():
        return _lowpoly_points()

    edge = np.zeros_like(arr, dtype=bool)
    edge[1:-1, 1:-1] = arr[1:-1, 1:-1] & (~arr[:-2, 1:-1] | ~arr[2:, 1:-1] |
                                           ~arr[1:-1, :-2] | ~arr[1:-1, 2:])
    ys, xs = np.where(edge)
    if len(xs) < 20:
        ys, xs = np.where(arr)

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = pts.mean(axis=0)
    rel = pts - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    radii = np.sqrt((rel ** 2).sum(axis=1))

    outline = []
    bins_edges = np.linspace(-np.pi, np.pi, bins + 1)
    for idx in range(bins):
        mask = (angles >= bins_edges[idx]) & (angles < bins_edges[idx + 1])
        if not mask.any():
            continue
        best_idx = np.where(mask)[0][np.argmax(radii[mask])]
        outline.append(pts[best_idx])

    if len(outline) < 12:
        return _lowpoly_points()

    outline = np.array(outline)
    return _normalize_closed(outline)


def get_heart_blueprint(kind, n=96):
    """Return normalized closed heart points for the requested blueprint."""
    if kind == 'valentine':
        pts = _valentine_points(max(120, n))
    elif kind == 'cardioid':
        pts = _cardioid_points(max(120, n))
    elif kind == 'taubin_like':
        pts = _taubin_like_points(max(120, n))
    elif kind == 'lowpoly':
        pts = _lowpoly_points()
    elif kind in ('simple_png_contour', 'emoji_contour', 'handdrawn_contour'):
        pts = _raster_boundary_points(kind)
    else:
        raise ValueError(f'Unknown heart blueprint: {kind}')

    return _resample_closed(pts, n=n)


def _route_flags(config):
    ignored = {
        'ga_scale_min', 'ga_scale_max', 'ga_km_range', 'ga_generations',
        'ga_population', 'ga_seed_rot_step', 'ga_seed_scales', 'wide_search',
        'wide_km_range', 'wide_offset_steps', 'wide_n_coarse', 'wide_n_refine',
        'wide_n_fine', 'graph_mode', 'graph_dist', 'open_space_graph',
    }
    return {k: v for k, v in config.items() if k not in ignored}


def genetic_parameter_search(G, pts, center, kdtree_data=None, config=None,
                             mode='optimize'):
    """GA-style search over rotation, scale, and center offset parameters."""
    config = config or {}
    route_config = _route_flags(config)
    rng = np.random.default_rng(42)

    population = int(config.get('ga_population', 10))
    generations = int(config.get('ga_generations', 4))
    scale_min = float(config.get('ga_scale_min', 0.010))
    scale_max = float(config.get('ga_scale_max', 0.028))
    km_range = float(config.get('ga_km_range', 1.5))
    rot_step = int(config.get('ga_seed_rot_step', 30))
    seed_scales = int(config.get('ga_seed_scales', 5))

    deg_lat = km_range * 0.009
    deg_lng = km_range * 0.012
    scales = np.linspace(scale_min, scale_max, seed_scales)
    rotations = list(range(0, 360, rot_step))
    offsets = [(0.0, 0.0), (deg_lat * 0.5, 0.0), (-deg_lat * 0.5, 0.0),
               (0.0, deg_lng * 0.5), (0.0, -deg_lng * 0.5)]

    seeds = []
    for rot in rotations:
        for sc in scales:
            for dlat, dlng in offsets:
                seeds.append((rot, sc, center[0] + dlat, center[1] + dlng))
    rng.shuffle(seeds)
    population_members = list(seeds[:population])

    best = None
    best_route = None
    best_score = 1e9

    def evaluate(candidate):
        rot, sc, lat, lng = candidate
        score, route = fit_and_score_v83(
            G, pts, rot, sc, [lat, lng], config=route_config,
            kdtree_data=kdtree_data,
        )
        return score, route

    for gen in range(generations):
        scored = []
        for cand in population_members:
            score, route = evaluate(cand)
            scored.append((score, cand, route))
            if route and score < best_score:
                best_score = score
                best = cand
                best_route = route

        scored.sort(key=lambda item: item[0])
        elites = scored[:max(3, population // 3)]
        log(f"[v8.6-ga] gen={gen+1}/{generations} best={elites[0][0]:.1f}")

        new_population = [cand for _, cand, _ in elites]
        while len(new_population) < population:
            parent_a = elites[rng.integers(0, len(elites))][1]
            parent_b = elites[rng.integers(0, len(elites))][1]
            rot = ((parent_a[0] + parent_b[0]) * 0.5 + rng.normal(0, 12)) % 360
            sc = np.clip((parent_a[1] + parent_b[1]) * 0.5 + rng.normal(0, 0.003),
                         scale_min, scale_max)
            lat = np.clip((parent_a[2] + parent_b[2]) * 0.5 + rng.normal(0, deg_lat * 0.18),
                          center[0] - deg_lat, center[0] + deg_lat)
            lng = np.clip((parent_a[3] + parent_b[3]) * 0.5 + rng.normal(0, deg_lng * 0.18),
                          center[1] - deg_lng, center[1] + deg_lng)
            new_population.append((rot, sc, lat, lng))
        population_members = new_population

    if best is None:
        return 1e9, None, {}

    return best_score, best_route, {
        'rotation': round(float(best[0]), 1),
        'scale': round(float(best[1]), 5),
        'center': [float(best[2]), float(best[3])],
        'search': 'genetic',
        'mode': mode,
    }
