"""Quick engine comparison - Quick Fit mode only"""
import json
import time
from datetime import datetime
from engine import mode_fit as mode_fit_v5
from engine_v6 import mode_fit as mode_fit_v6
import math

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

print("Running Quick Comparison (Quick Fit only)...")
print("=" * 60)

# v5.1
print("\n[v5.1] Running Quick Fit...")
t0 = time.time()
fit_v5 = mode_fit_v5(payload)
time_v5 = time.time() - t0
print(f"  ✓ Score: {fit_v5.get('score', 'N/A')}m in {time_v5:.1f}s")

# v6.0
print("\n[v6.0] Running Quick Fit...")
t0 = time.time()
fit_v6 = mode_fit_v6(payload)
time_v6 = time.time() - t0
print(f"  ✓ Score: {fit_v6.get('score', 'N/A')}m in {time_v6:.1f}s")

# Save results
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
                "time_seconds": time_v5,
                "rotation": fit_v5.get('rotation'),
                "scale": fit_v5.get('scale'),
                "route": fit_v5.get('route', [])
            },
            "v6.0": {
                "score": fit_v6.get('score'),
                "route_length_m": fit_v6.get('route_length_m'),
                "time_seconds": time_v6,
                "rotation": fit_v6.get('rotation'),
                "scale": fit_v6.get('scale'),
                "route": fit_v6.get('route', [])
            },
            "improvement_percent": ((fit_v5.get('score', 1e9) - fit_v6.get('score', 1e9)) / fit_v5.get('score', 1e9) * 100)
        }
    }
}

with open("public/engine_comparison.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved to public/engine_comparison.json")
print(f"View at: http://127.0.0.1:5000/compare")
