"""Add v8 benchmark results to full_comparison.json."""
import json, math, time, sys

from engine import mode_fit, mode_optimize, ENGINE_VERSION

CENTER = [51.4543, -0.9781]

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
payload = {"shapes": [shape], "shape_index": 0, "center_point": CENTER}

# Load existing comparison data
with open("public/full_comparison.json", "r") as f:
    data = json.load(f)

engines = [
    ("v8.0", "fit", mode_fit),
    ("v8.0", "optimize", mode_optimize),
]

print("=" * 70)
print(f"ADDING v8.0 ({ENGINE_VERSION}) BENCHMARKS TO COMPARISON")
print("=" * 70)

for version, mode, func in engines:
    key = f"{version}_{mode}"
    print(f"\n[{version}] {mode}...", flush=True)
    t0 = time.time()
    r = func(payload)
    elapsed = time.time() - t0
    score = r.get("score", 0)
    n_pts = len(r.get("route", []))
    print(f"  Score: {score:.1f}m in {elapsed:.1f}s  ({n_pts} pts)")

    if "error" in r:
        print(f"  ERROR: {r['error']}")
        continue

    data["routes"][key] = {
        "version": version,
        "mode": mode,
        "score": score,
        "route_length_m": r.get("route_length_m"),
        "time_seconds": round(elapsed, 2),
        "rotation": r.get("rotation"),
        "scale": r.get("scale"),
        "route": r.get("route", [])
    }

    with open("public/full_comparison.json", "w") as f:
        json.dump(data, f)
    print(f"  [saved]")

print("\n" + "=" * 70)
print("FULL SUMMARY")
print("=" * 70)
print(f"\n{'Key':<20} {'Score (m)':<12} {'Length (m)':<12} {'Time (s)':<10} {'Points'}")
print("-" * 70)
for key in sorted(data["routes"].keys()):
    d = data["routes"][key]
    length = d.get("route_length_m") or 0
    print(f"{key:<20} {d['score']:<12.1f} {length:<12.0f} {d['time_seconds']:<10.1f} {len(d['route'])}")
