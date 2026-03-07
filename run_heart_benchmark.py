import json
import time
from datetime import datetime
from engine import mode_fit, mode_optimize

CENTER = [51.4543, -0.9781]

HEART_SHAPE = {
    "pts": [
        [0.50, 0.14], [0.66, -0.04], [0.90, -0.06], [1.0, 0.18],
        [0.94, 0.46], [0.76, 0.70], [0.50, 1.0], [0.24, 0.70],
        [0.06, 0.46], [0.0, 0.18], [0.10, -0.06], [0.34, -0.04],
        [0.50, 0.14]
    ],
    "name": "Heart"
}

payload = {
    "shapes": [HEART_SHAPE],
    "shape_index": 0,
    "center_point": CENTER
}

# --- Quick Fit ---
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

# --- Auto-Optimize ---
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

# Build full benchmark structure
benchmark = {
    "location": "Reading, UK",
    "center": CENTER,
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "fit": [fit_result],
    "optimize": [opt_result],
    "best_shape": None,
    "summary": {
        "fit_count": 1,
        "fit_success": 1 if "error" not in fit_result else 0,
        "fit_avg_score": fit_score,
        "fit_best_score": fit_score,
        "fit_worst_score": fit_score,
        "fit_avg_time": round(fit_time, 1),
        "fit_total_time": round(fit_time, 1),
        "opt_count": 1,
        "opt_success": 1 if "error" not in opt_result else 0,
        "opt_avg_score": opt_score,
        "opt_best_score": opt_score,
        "opt_worst_score": opt_score,
        "opt_avg_time": round(opt_time, 1),
        "opt_total_time": round(opt_time, 1),
        "best_shape_name": None,
        "best_shape_score": None,
        "best_shape_time": None,
        "status": "complete"
    }
}

with open("public/benchmark_results.json", "w") as f:
    json.dump(benchmark, f)

print(f"\nBenchmark written to public/benchmark_results.json")
print(f"  Quick Fit:     score={fit_score}m, time={fit_time:.1f}s")
print(f"  Auto-Optimize: score={opt_score}m, time={opt_time:.1f}s")