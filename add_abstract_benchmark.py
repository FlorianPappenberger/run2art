"""
add_abstract_benchmark.py — Run v8.1-Abstract benchmarks and add to full_comparison.json
"""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from engine import (
    fit_and_score_abstract, make_result,
    coarse_grid_search, make_offsets,
)
from geometry import shape_to_latlngs, densify, adaptive_densify
from scoring import coarse_proximity_score
from routing import log, fetch_graph, build_kdtree

# ── Heart parametric shape (same as other benchmarks) ──
import math
heart_pts = []
for i in range(200):
    t = 2 * math.pi * i / 200
    x = 16 * math.sin(t) ** 3
    y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
    heart_pts.append([x / 16.0, y / 17.0])

CENTER = [51.4543, -0.9781]  # Reading, UK

def run_abstract(mode_name, fit_func_kwargs):
    """Run abstract routing and return result dict."""
    t0 = time.time()
    G = fetch_graph(CENTER, dist=fit_func_kwargs.get('dist', 2500))
    kd = build_kdtree(G) if G else None

    rotations = fit_func_kwargs.get('rotations', list(range(0, 360, 45)))
    scales = fit_func_kwargs.get('scales', [0.010, 0.014, 0.018, 0.023, 0.030])
    offsets = make_offsets(km_range=fit_func_kwargs.get('km_range', 1.0),
                          steps=fit_func_kwargs.get('steps', 2))

    coarse = coarse_grid_search(G, heart_pts, CENTER, rotations, scales,
                                offsets, densify_spacing=150, kdtree_data=kd)

    best_score, best_result = 1e9, None
    n_fine = fit_func_kwargs.get('n_fine', 4)
    
    for _, rot, sc, c in coarse[:n_fine]:
        for dr in fit_func_kwargs.get('dr_list', [0, -15, 15]):
            for sf in fit_func_kwargs.get('sf_list', [1.0, 0.90, 1.10]):
                for dlat, dlng in [(0, 0), (0.001, 0), (-0.001, 0),
                                   (0, 0.0015), (0, -0.0015)]:
                    r2, s2 = rot + dr, sc * sf
                    c2 = [c[0] + dlat, c[1] + dlng]
                    score, route = fit_and_score_abstract(G, heart_pts, r2, s2, c2,
                                                          kdtree_data=kd)
                    if score < best_score:
                        best_score = score
                        best_result = make_result(route, score, r2, s2, c2)
                        log(f"[{mode_name}] New best: rot={r2:.0f} sc={s2:.4f} score={score:.1f}")

    elapsed = time.time() - t0
    if best_result:
        best_result['time_seconds'] = round(elapsed, 1)
        best_result['version'] = 'v8.1-abs'
        best_result['mode'] = mode_name.split('_')[-1]
    return best_result, elapsed


def main():
    comp_path = os.path.join(os.path.dirname(__file__), 'public', 'full_comparison.json')
    with open(comp_path, 'r') as f:
        comp = json.load(f)

    # ── FIT mode ──
    log("=" * 60)
    log("Running v8.1-Abstract FIT...")
    log("=" * 60)
    fit_result, fit_time = run_abstract('abstract_fit', {
        'dist': 2500,
        'rotations': list(range(0, 360, 45)),
        'scales': [0.010, 0.014, 0.018, 0.023, 0.030],
        'km_range': 1.0, 'steps': 2, 'n_fine': 4,
        'dr_list': [0, -15, 15],
        'sf_list': [1.0, 0.90, 1.10],
    })
    if fit_result:
        comp['routes']['v8.1_abstract_fit'] = fit_result
        log(f"FIT: score={fit_result['score']}, time={fit_time:.1f}s, "
            f"rot={fit_result['rotation']}, scale={fit_result['scale']}")

    # ── OPTIMIZE mode ──
    log("=" * 60)
    log("Running v8.1-Abstract OPTIMIZE...")
    log("=" * 60)
    opt_result, opt_time = run_abstract('abstract_optimize', {
        'dist': 4000,
        'rotations': list(range(0, 360, 15)),
        'scales': [0.010, 0.014, 0.018, 0.022, 0.027, 0.033, 0.040],
        'km_range': 2.0, 'steps': 3, 'n_fine': 8,
        'dr_list': [0, -7, 7, -15, 15],
        'sf_list': [1.0, 0.88, 1.12],
    })
    if opt_result:
        comp['routes']['v8.1_abstract_optimize'] = opt_result
        log(f"OPTIMIZE: score={opt_result['score']}, time={opt_time:.1f}s, "
            f"rot={opt_result['rotation']}, scale={opt_result['scale']}")

    with open(comp_path, 'w') as f:
        json.dump(comp, f, indent=2)
    log(f"Saved to {comp_path}")


if __name__ == '__main__':
    main()
