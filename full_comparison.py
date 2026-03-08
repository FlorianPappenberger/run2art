"""Full comparison: fit + optimize for v5.1, v6.0, v7.0"""
import json
import time
import math
from datetime import datetime

from engine_v5 import mode_fit as fit_v5, mode_optimize as opt_v5
from engine_v6 import mode_fit as fit_v6, mode_optimize as opt_v6
from engine import mode_fit as fit_v7, mode_optimize as opt_v7

CENTER = [51.4543, -0.9781]

def parametric_heart(n=50):
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

HEART_SHAPE = {"pts": parametric_heart(50), "name": "Heart"}
payload = {"shapes": [HEART_SHAPE], "shape_index": 0, "center_point": CENTER}

engines = [
    ("v5.1", "fit",      fit_v5),
    ("v5.1", "optimize", opt_v5),
    ("v6.0", "fit",      fit_v6),
    ("v6.0", "optimize", opt_v6),
    ("v7.0", "fit",      fit_v7),
    ("v7.0", "optimize", opt_v7),
]

results = {
    "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "location": "Reading, UK",
    "center": CENTER,
    "shape": "Heart (parametric)",
    "routes": {}
}

print("=" * 70)
print("FULL COMPARISON: fit + optimize for v5.1 / v6.0 / v7.0")
print("=" * 70)

for version, mode, func in engines:
    key = f"{version}_{mode}"
    print(f"\n[{version}] {mode}...")
    t0 = time.time()
    r = func(payload)
    elapsed = time.time() - t0
    score = r.get("score", 0)
    print(f"  Score: {score:.1f}m in {elapsed:.1f}s  ({len(r.get('route', []))} pts)")

    results["routes"][key] = {
        "version": version,
        "mode": mode,
        "score": score,
        "route_length_m": r.get("route_length_m"),
        "time_seconds": round(elapsed, 2),
        "rotation": r.get("rotation"),
        "scale": r.get("scale"),
        "route": r.get("route", [])
    }

    # Save after each result so partial results are available
    with open("public/full_comparison.json", "w") as f:
        json.dump(results, f)
    print(f"  [saved]")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Key':<20} {'Score (m)':<12} {'Time (s)':<10} {'Points'}")
print("-" * 55)
for key, data in results["routes"].items():
    print(f"{key:<20} {data['score']:<12.1f} {data['time_seconds']:<10.1f} {len(data['route'])}")

print(f"\nSaved to public/full_comparison.json")
