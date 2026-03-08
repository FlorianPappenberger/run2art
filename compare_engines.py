"""
compare_engines.py — Benchmark v5.1 vs v6.0 Performance
========================================================
Tests both engines on the heart shape and compares:
  - Route fidelity (bidirectional score)
  - Shape recognizability (visual inspection metrics)
  - Computational efficiency
"""

import json
import math
import time
from datetime import datetime

# Import both engine versions
from engine import mode_fit as mode_fit_v5, mode_optimize as mode_optimize_v5
from engine_v6 import mode_fit as mode_fit_v6, mode_optimize as mode_optimize_v6

CENTER = [51.4543, -0.9781]


def parametric_heart(n=50):
    """Generate a smooth heart using parametric equations."""
    raw = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x = 16.0 * math.sin(t) ** 3
        y = 13.0 * math.cos(t) - 5.0 * math.cos(2*t) \
            - 2.0 * math.cos(3*t) - math.cos(4*t)
        raw.append((x, y))
    xs, ys = zip(*raw)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pts = [[round((x - xmin) / (xmax - xmin), 4),
            round(1.0 - (y - ymin) / (ymax - ymin), 4)]
           for x, y in raw]
    pts.append(pts[0])
    return pts


HEART_SHAPE = {
    "pts": parametric_heart(50),
    "name": "Heart"
}

payload = {
    "shapes": [HEART_SHAPE],
    "shape_index": 0,
    "center_point": CENTER
}


def run_comparison():
    """Run both engines and compare results."""
    print("=" * 70)
    print("GPS ART ENGINE COMPARISON: v5.1 vs v6.0")
    print("=" * 70)
    print(f"Location: Reading, UK ({CENTER[0]}, {CENTER[1]})")
    print(f"Shape: Heart (parametric, {len(HEART_SHAPE['pts'])} points)")
    print()
    
    results = {}
    
    # ==============================
    # Test 1: Quick Fit Mode
    # ==============================
    print("─" * 70)
    print("TEST 1: QUICK FIT MODE")
    print("─" * 70)
    
    print("\n[v5.1] Running Quick Fit...")
    t0 = time.time()
    fit_v5 = mode_fit_v5(payload)
    time_v5_fit = time.time() - t0
    
    print(f"  ✓ Completed in {time_v5_fit:.1f}s")
    print(f"    Score: {fit_v5.get('score', 'N/A')}m")
    print(f"    Route length: {fit_v5.get('route_length_m', 0):.0f}m")
    print(f"    Rotation: {fit_v5.get('rotation', 0):.0f}°")
    print(f"    Scale: {fit_v5.get('scale', 0):.5f}")
    
    print("\n[v6.0] Running Quick Fit...")
    t0 = time.time()
    fit_v6 = mode_fit_v6(payload)
    time_v6_fit = time.time() - t0
    
    print(f"  ✓ Completed in {time_v6_fit:.1f}s")
    print(f"    Score: {fit_v6.get('score', 'N/A')}m")
    print(f"    Route length: {fit_v6.get('route_length_m', 0):.0f}m")
    print(f"    Rotation: {fit_v6.get('rotation', 0):.0f}°")
    print(f"    Scale: {fit_v6.get('scale', 0):.5f}")
    
    # Compute improvement
    score_improvement_fit = ((fit_v5.get('score', 1e9) - fit_v6.get('score', 1e9)) 
                             / fit_v5.get('score', 1e9) * 100)
    
    print(f"\n📊 Quick Fit Comparison:")
    print(f"    Score improvement: {score_improvement_fit:+.1f}%")
    print(f"    Time delta: {time_v6_fit - time_v5_fit:+.1f}s")
    
    results['fit'] = {
        'v5': fit_v5,
        'v6': fit_v6,
        'time_v5': time_v5_fit,
        'time_v6': time_v6_fit,
        'improvement': score_improvement_fit
    }
    
    # ==============================
    # Test 2: Auto-Optimize Mode
    # ==============================
    print("\n" + "─" * 70)
    print("TEST 2: AUTO-OPTIMIZE MODE")
    print("─" * 70)
    
    print("\n[v5.1] Running Auto-Optimize...")
    t0 = time.time()
    opt_v5 = mode_optimize_v5(payload)
    time_v5_opt = time.time() - t0
    
    print(f"  ✓ Completed in {time_v5_opt:.1f}s")
    print(f"    Score: {opt_v5.get('score', 'N/A')}m")
    print(f"    Route length: {opt_v5.get('route_length_m', 0):.0f}m")
    print(f"    Rotation: {opt_v5.get('rotation', 0):.0f}°")
    print(f"    Scale: {opt_v5.get('scale', 0):.5f}")
    
    print("\n[v6.0] Running Auto-Optimize...")
    t0 = time.time()
    opt_v6 = mode_optimize_v6(payload)
    time_v6_opt = time.time() - t0
    
    print(f"  ✓ Completed in {time_v6_opt:.1f}s")
    print(f"    Score: {opt_v6.get('score', 'N/A')}m")
    print(f"    Route length: {opt_v6.get('route_length_m', 0):.0f}m")
    print(f"    Rotation: {opt_v6.get('rotation', 0):.0f}°")
    print(f"    Scale: {opt_v6.get('scale', 0):.5f}")
    
    # Compute improvement
    score_improvement_opt = ((opt_v5.get('score', 1e9) - opt_v6.get('score', 1e9)) 
                             / opt_v5.get('score', 1e9) * 100)
    
    print(f"\n📊 Auto-Optimize Comparison:")
    print(f"    Score improvement: {score_improvement_opt:+.1f}%")
    print(f"    Time delta: {time_v6_opt - time_v5_opt:+.1f}s")
    
    results['optimize'] = {
        'v5': opt_v5,
        'v6': opt_v6,
        'time_v5': time_v5_opt,
        'time_v6': time_v6_opt,
        'improvement': score_improvement_opt
    }
    
    # ==============================
    # Summary
    # ==============================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n🎯 SHAPE FIDELITY IMPROVEMENTS:")
    print(f"    Quick Fit:     {score_improvement_fit:+.1f}% (v5: {fit_v5.get('score', 0):.1f}m → v6: {fit_v6.get('score', 0):.1f}m)")
    print(f"    Auto-Optimize: {score_improvement_opt:+.1f}% (v5: {opt_v5.get('score', 0):.1f}m → v6: {opt_v6.get('score', 0):.1f}m)")
    
    print(f"\n⏱️  PERFORMANCE:")
    print(f"    Quick Fit:     v5: {time_v5_fit:.1f}s vs v6: {time_v6_fit:.1f}s")
    print(f"    Auto-Optimize: v5: {time_v5_opt:.1f}s vs v6: {time_v6_opt:.1f}s")
    
    print(f"\n🔑 KEY ENHANCEMENTS IN v6.0:")
    print(f"    ✓ Curvature-based adaptive densification")
    print(f"    ✓ Segment-constrained routing with 'tube' penalties")
    print(f"    ✓ Multi-objective edge weighting (α=15.0, β=0.8)")
    print(f"    ✓ Turning-angle penalties integrated into Dijkstra")
    print(f"    ✓ Iterative refinement with tighter constraints")
    
    print("\n" + "=" * 70)
    
    # Save detailed results WITH ROUTES for visualization
    output = {
        "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Reading, UK",
        "center": CENTER,
        "shape": "Heart (parametric)",
        "results": {
            "quick_fit": {
                "v5.1": {
                    "score": fit_v5.get('score'),
                    "route_length_m": fit_v5.get('route_length_m'),
                    "time_seconds": time_v5_fit,
                    "rotation": fit_v5.get('rotation'),
                    "scale": fit_v5.get('scale'),
                    "route": fit_v5.get('route', [])
                },
                "v6.0": {
                    "score": fit_v6.get('score'),
                    "route_length_m": fit_v6.get('route_length_m'),
                    "time_seconds": time_v6_fit,
                    "rotation": fit_v6.get('rotation'),
                    "scale": fit_v6.get('scale'),
                    "route": fit_v6.get('route', [])
                },
                "improvement_percent": score_improvement_fit
            },
            "auto_optimize": {
                "v5.1": {
                    "score": opt_v5.get('score'),
                    "route_length_m": opt_v5.get('route_length_m'),
                    "time_seconds": time_v5_opt,
                    "rotation": opt_v5.get('rotation'),
                    "scale": opt_v5.get('scale'),
                    "route": opt_v5.get('route', [])
                },
                "v6.0": {
                    "score": opt_v6.get('score'),
                    "route_length_m": opt_v6.get('route_length_m'),
                    "time_seconds": time_v6_opt,
                    "rotation": opt_v6.get('rotation'),
                    "scale": opt_v6.get('scale'),
                    "route": opt_v6.get('route', [])
                },
                "improvement_percent": score_improvement_opt
            }
        }
    }
    
    with open("public/engine_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n📄 Detailed results saved to: public/engine_comparison.json")
    
    return results


if __name__ == "__main__":
    run_comparison()
