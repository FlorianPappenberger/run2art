"""Super quick comparison - fit mode only for all versions"""
import json
import time
import math
from datetime import datetime

# Import all engine versions
from engine import mode_fit as mode_fit_v7
from engine_v6 import mode_fit as mode_fit_v6
from engine_v5 import mode_fit as mode_fit_v5

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

print("=" * 80)
print("Quick Comparison - Fit Mode Only")
print("=" * 80)

results = {
    "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "location": "Reading, UK",
    "center": CENTER,
    "shape": "Heart (parametric)",
    "versions": {}
}

# v5.1
print("\n[v5.1] Fit...")
t0 = time.time()
fit_v5 = mode_fit_v5(payload)
time_v5 = time.time() - t0
score_v5 = fit_v5.get('score', 0)
print(f"  Score: {score_v5:.1f}m in {time_v5:.1f}s")

results["versions"]["v5.1_fit"] = {
    "version": "v5.1",
    "mode": "fit",
    "score": score_v5,
    "route_length_m": fit_v5.get('route_length_m'),
    "time_seconds": round(time_v5, 2),
    "rotation": fit_v5.get('rotation'),
    "scale": fit_v5.get('scale'),
    "route": fit_v5.get('route', [])
}

# v6.0
print("\n[v6.0] Fit...")
t0 = time.time()
fit_v6 = mode_fit_v6(payload)
time_v6 = time.time() - t0
score_v6 = fit_v6.get('score', 0)
print(f"  Score: {score_v6:.1f}m in {time_v6:.1f}s")

results["versions"]["v6.0_fit"] = {
    "version": "v6.0",
    "mode": "fit",
    "score": score_v6,
    "route_length_m": fit_v6.get('route_length_m'),
    "time_seconds": round(time_v6, 2),
    "rotation": fit_v6.get('rotation'),
    "scale": fit_v6.get('scale'),
    "route": fit_v6.get('route', [])
}

# v7.0
print("\n[v7.0] Fit...")
t0 = time.time()
fit_v7 = mode_fit_v7(payload)
time_v7 = time.time() - t0
score_v7 = fit_v7.get('score', 0)
print(f"  Score: {score_v7:.1f}m in {time_v7:.1f}s")

results["versions"]["v7.0_fit"] = {
    "version": "v7.0",
    "mode": "fit",
    "score": score_v7,
    "route_length_m": fit_v7.get('route_length_m'),
    "time_seconds": round(time_v7, 2),
    "rotation": fit_v7.get('rotation'),
    "scale": fit_v7.get('scale'),
    "route": fit_v7.get('route', [])
}

# Save
output_file = "public/all_versions_comparison.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\n{'Version':<15} {'Score (m)':<12} {'Time (s)':<10}")
print("-" * 40)
print(f"{'v5.1 (fit)':<15} {score_v5:<12.1f} {time_v5:<10.1f}")
print(f"{'v6.0 (fit)':<15} {score_v6:<12.1f} {time_v6:<10.1f}")
print(f"{'v7.0 (fit)':<15} {score_v7:<12.1f} {time_v7:<10.1f}")

# Improvement
if score_v5 > 0:
    improv_v6 = ((score_v5 - score_v6) / score_v5) * 100
    improv_v7 = ((score_v5 - score_v7) / score_v5) * 100
    print(f"\nv6.0 vs v5.1: {improv_v6:+.1f}%")
    print(f"v7.0 vs v5.1: {improv_v7:+.1f}%")
    print(f"v7.0 vs v6.0: {((score_v6 - score_v7) / score_v6) * 100:+.1f}%")

print(f"\n[OK] Results saved to {output_file}")
