"""Quick comparison of all three engine versions - both modes, heart only"""
import json
import time
import math
from datetime import datetime

# Import all engine versions
from engine import mode_fit as mode_fit_v7, mode_optimize as mode_optimize_v7
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
print("Quick Multi-Version Comparison - Heart Shape Only")
print("=" * 80)
print(f"Location: Reading, UK {CENTER}")
print(f"Shape: Heart (parametric, {len(HEART_SHAPE['pts'])} points)")
print("=" * 80)

results = {
    "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "location": "Reading, UK",
    "center": CENTER,
    "shape": "Heart (parametric)",
    "versions": {}
}

# v5.1 - Both modes
print("\n[v5.1] Running Quick Fit...")
t0 = time.time()
fit_v5 = mode_fit_v5(payload)
time_v5_fit = time.time() - t0
print(f"  Score: {fit_v5.get('score')}m in {time_v5_fit:.1f}s")

results["versions"]["v5.1"] = {
    "fit": {
        "mode": "fit",
        "score": fit_v5.get('score'),
        "route_length_m": fit_v5.get('route_length_m'),
        "time_seconds": round(time_v5_fit, 2),
        "rotation": fit_v5.get('rotation'),
        "scale": fit_v5.get('scale'),
        "route": fit_v5.get('route', [])
    }
}

# v6.0 - Fit only (optimize is too slow)
print("\n[v6.0] Running Quick Fit...")
t0 = time.time()
fit_v6 = mode_fit_v6(payload)
time_v6_fit = time.time() - t0
print(f"  Score: {fit_v6.get('score')}m in {time_v6_fit:.1f}s")

results["versions"]["v6.0"] = {
    "fit": {
        "mode": "fit",
        "score": fit_v6.get('score'),
        "route_length_m": fit_v6.get('route_length_m'),
        "time_seconds": round(time_v6_fit, 2),
        "rotation": fit_v6.get('rotation'),
        "scale": fit_v6.get('scale'),
        "route": fit_v6.get('route', [])
    }
}

# v7.0 - Both modes
print("\n[v7.0] Running Quick Fit...")
t0 = time.time()
fit_v7 = mode_fit_v7(payload)
time_v7_fit = time.time() - t0
print(f"  Score: {fit_v7.get('score')}m in {time_v7_fit:.1f}s")

print("\n[v7.0] Running Auto-Optimize...")
t0 = time.time()
opt_v7 = mode_optimize_v7(payload)
time_v7_opt = time.time() - t0
print(f"  Score: {opt_v7.get('score')}m in {time_v7_opt:.1f}s")

results["versions"]["v7.0"] = {
    "fit": {
        "mode": "fit",
        "score": fit_v7.get('score'),
        "route_length_m": fit_v7.get('route_length_m'),
        "time_seconds": round(time_v7_fit, 2),
        "rotation": fit_v7.get('rotation'),
        "scale": fit_v7.get('scale'),
        "route": fit_v7.get('route', [])
    },
    "optimize": {
        "mode": "optimize",
        "score": opt_v7.get('score'),
        "route_length_m": opt_v7.get('route_length_m'),
        "time_seconds": round(time_v7_opt, 2),
        "rotation": opt_v7.get('rotation'),
        "scale": opt_v7.get('scale'),
        "route": opt_v7.get('route', [])
    }
}

# Save results
output_file = "public/quick_multi_comparison.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{'Version':<10} {'Mode':<10} {'Score (m)':<12} {'Time (s)':<10}")
print("-" * 50)

for version in ["v5.1", "v6.0", "v7.0"]:
    if version in results["versions"]:
        for mode in ["fit", "optimize"]:
            if mode in results["versions"][version]:
                data = results["versions"][version][mode]
                score = f"{data['score']:.1f}" if data['score'] else "N/A"
                time_s = f"{data['time_seconds']:.1f}"
                print(f"{version:<10} {mode:<10} {score:<12} {time_s:<10}")

print(f"\n[OK] Results saved to {output_file}")
print(f"[OK] Open browser: http://127.0.0.1:5000/quick_multi_map.html")
