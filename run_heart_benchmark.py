import json
import math
import time
from datetime import datetime
from engine import mode_fit, mode_optimize
import os

CENTER = [51.4543, -0.9781]

# Check if benchmark results already exist
benchmark_path = "public/benchmark_results.json"
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r") as f:
        try:
            benchmark = json.load(f)
            print("Loaded existing benchmark results.")
        except json.JSONDecodeError:
            print("Existing benchmark file is corrupted. Starting fresh.")
            benchmark = None
else:
    benchmark = None

# Initialize benchmark structure if not loaded
if not benchmark:
    benchmark = {
        "location": "Reading, UK",
        "center": CENTER,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fit": [],
        "optimize": [],
        "best_shape": None,
        "summary": {}
    }

# Move parametric_heart function above HEART_SHAPE initialization

def parametric_heart(n=50):
    """Generate a smooth heart using the parametric equation:
       x(t) = 16 sin^3(t)
       y(t) = 13 cos(t) - 5 cos(2t) - 2 cos(3t) - cos(4t)
    Normalised to [0,1] with Y flipped so the point faces south
    (matching shape_to_latlngs Y-inversion)."""
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
    pts.append(pts[0])  # close the loop
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

# --- Quick Fit ---
if not benchmark["fit"]:
    print("Running Quick Fit...")
    t0 = time.time()
    fit_result = mode_fit(payload)
    fit_time = time.time() - t0
    fit_result["time_seconds"] = round(fit_time, 2)
    fit_result["mode"] = "fit"
    fit_result["shape_name"] = "Heart"
    fit_result["global_shape_index"] = 0
    fit_score = fit_result.get("score", 0)
    print(f"  Quick Fit done: score={fit_score}m, time={fit_time:.1f}s")
    benchmark["fit"].append(fit_result)
else:
    print("Quick Fit results already available. Skipping.")

# --- Auto-Optimize ---
if not benchmark["optimize"]:
    print("Running Auto-Optimize...")
    t0 = time.time()
    opt_result = mode_optimize(payload)
    opt_time = time.time() - t0
    opt_result["time_seconds"] = round(opt_time, 2)
    opt_result["mode"] = "optimize"
    opt_result["shape_name"] = "Heart"
    opt_result["global_shape_index"] = 0
    opt_score = opt_result.get("score", 0)
    print(f"  Auto-Optimize done: score={opt_score}m, time={opt_time:.1f}s")
    benchmark["optimize"].append(opt_result)
else:
    print("Auto-Optimize results already available. Skipping.")

# Update summary
benchmark["summary"].update({
    "fit_count": len(benchmark["fit"]),
    "fit_success": sum(1 for r in benchmark["fit"] if "error" not in r),
    "fit_avg_score": sum(r.get("score", 0) for r in benchmark["fit"]) / len(benchmark["fit"]),
    "fit_best_score": min(r.get("score", float("inf")) for r in benchmark["fit"]),
    "fit_worst_score": max(r.get("score", 0) for r in benchmark["fit"]),
    "fit_avg_time": sum(r.get("time_seconds", 0) for r in benchmark["fit"]) / len(benchmark["fit"]),
    "fit_total_time": sum(r.get("time_seconds", 0) for r in benchmark["fit"]),
    "opt_count": len(benchmark["optimize"]),
    "opt_success": sum(1 for r in benchmark["optimize"] if "error" not in r),
    "opt_avg_score": sum(r.get("score", 0) for r in benchmark["optimize"]) / len(benchmark["optimize"]),
    "opt_best_score": min(r.get("score", float("inf")) for r in benchmark["optimize"]),
    "opt_worst_score": max(r.get("score", 0) for r in benchmark["optimize"]),
    "opt_avg_time": sum(r.get("time_seconds", 0) for r in benchmark["optimize"]) / len(benchmark["optimize"]),
    "opt_total_time": sum(r.get("time_seconds", 0) for r in benchmark["optimize"]),
})

# Save updated benchmark
with open(benchmark_path, "w") as f:
    json.dump(benchmark, f)

print(f"\nBenchmark written to {benchmark_path}")
if benchmark["fit"]:
    print(f"  Quick Fit:     score={benchmark['fit'][0].get('score', 0)}m, time={benchmark['fit'][0].get('time_seconds', 0):.1f}s")
if benchmark["optimize"]:
    print(f"  Auto-Optimize: score={benchmark['optimize'][0].get('score', 0)}m, time={benchmark['optimize'][0].get('time_seconds', 0):.1f}s")