"""Quick smoke test: v8 engine fit on heart shape."""
import sys, json, math, time

from engine import mode_fit, mode_optimize, ENGINE_VERSION

print(f"Engine version: {ENGINE_VERSION}")

def parametric_heart(n=50):
    raw = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x = 16.0 * math.sin(t) ** 3
        y = 13.0 * math.cos(t) - 5.0 * math.cos(2*t) - 2.0 * math.cos(3*t) - math.cos(4*t)
        raw.append((x, y))
    xs, ys = zip(*raw)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pts = [[round((x - xmin) / (xmax - xmin), 4),
            round(1.0 - (y - ymin) / (ymax - ymin), 4)]
           for x, y in raw]
    pts.append(pts[0])
    return pts

shape = {"pts": parametric_heart(50), "name": "Heart"}
payload = {"shapes": [shape], "shape_index": 0, "center_point": [51.4543, -0.9781]}

print("\n--- v8 FIT ---")
t0 = time.time()
r = mode_fit(payload)
elapsed = time.time() - t0
print(f"Score: {r.get('score', 'N/A')}m")
print(f"Points: {len(r.get('route', []))}")
print(f"Time: {elapsed:.1f}s")
print(f"Route length: {r.get('route_length_m', 'N/A')}m")
if "error" in r:
    print(f"ERROR: {r['error']}")
else:
    print("FIT OK")
