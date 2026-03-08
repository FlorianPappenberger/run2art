"""Full engine comparison - All versions, both modes, heart shape only"""
import json
import time
import math
from datetime import datetime

# Import all engine versions
from engine import mode_fit as mode_fit_v7, mode_optimize as mode_optimize_v7
from engine_v6 import mode_fit as mode_fit_v6, mode_optimize as mode_optimize_v6

# For v5.1, we need to use the original engine before refactor
# We'll load it from a backup or git history if available
# For now, let's check if we have engine_v5.py

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

print("Running Full Engine Comparison...")
print("=" * 80)
print(f"Location: Reading, UK {CENTER}")
print(f"Shape: Heart (parametric, {len(HEART_SHAPE['pts'])} points)")
print("=" * 80)

# Try to import v5.1
try:
    import sys
    import os
    # Check if we have engine_v5.py
    if os.path.exists('engine_v5.py'):
        from engine_v5 import mode_fit as mode_fit_v5, mode_optimize as mode_optimize_v5
        has_v5 = True
        print("[OK] Found engine_v5.py")
    else:
        has_v5 = False
        print("[WARN] engine_v5.py not found - will skip v5.1")
except ImportError:
    has_v5 = False
    print("[WARN] Could not import engine_v5 - will skip v5.1")

results = {
    "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "location": "Reading, UK",
    "center": CENTER,
    "shape": "Heart (parametric)",
    "versions": {}
}

# Test each version with both modes
versions = []
if has_v5:
    versions.append(("v5.1", mode_fit_v5, mode_optimize_v5))
versions.extend([
    ("v6.0", mode_fit_v6, mode_optimize_v6),
    ("v7.0", mode_fit_v7, mode_optimize_v7)
])

for version_name, fit_fn, optimize_fn in versions:
    print(f"\n{'=' * 80}")
    print(f"Testing {version_name}")
    print(f"{'=' * 80}")
    
    results["versions"][version_name] = {}
    
    # Quick Fit mode
    print(f"\n[{version_name}] Running Quick Fit...")
    try:
        t0 = time.time()
        fit_result = fit_fn(payload)
        time_fit = time.time() - t0
        
        score = fit_result.get('score', 'N/A')
        route_len = fit_result.get('route_length_m', 'N/A')
        print(f"  [OK] Score: {score}m | Route: {route_len}m | Time: {time_fit:.1f}s")
        
        results["versions"][version_name]["fit"] = {
            "mode": "fit",
            "score": fit_result.get('score'),
            "route_length_m": fit_result.get('route_length_m'),
            "time_seconds": round(time_fit, 2),
            "rotation": fit_result.get('rotation'),
            "scale": fit_result.get('scale'),
            "route": fit_result.get('route', [])
        }
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["versions"][version_name]["fit"] = {"error": str(e)}
    
    # Auto-Optimize mode
    print(f"\n[{version_name}] Running Auto-Optimize...")
    try:
        t0 = time.time()
        opt_result = optimize_fn(payload)
        time_opt = time.time() - t0
        
        score = opt_result.get('score', 'N/A')
        route_len = opt_result.get('route_length_m', 'N/A')
        print(f"  [OK] Score: {score}m | Route: {route_len}m | Time: {time_opt:.1f}s")
        
        results["versions"][version_name]["optimize"] = {
            "mode": "optimize",
            "score": opt_result.get('score'),
            "route_length_m": opt_result.get('route_length_m'),
            "time_seconds": round(time_opt, 2),
            "rotation": opt_result.get('rotation'),
            "scale": opt_result.get('scale'),
            "route": opt_result.get('route', [])
        }
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["versions"][version_name]["optimize"] = {"error": str(e)}

# Save results
output_file = "public/full_engine_comparison.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print(f"\n{'Version':<10} {'Mode':<10} {'Score (m)':<12} {'Time (s)':<10}")
print("-" * 50)

for version_name in results["versions"]:
    for mode in ["fit", "optimize"]:
        if mode in results["versions"][version_name]:
            data = results["versions"][version_name][mode]
            if "error" not in data:
                score = f"{data['score']:.1f}" if data['score'] else "N/A"
                time_s = f"{data['time_seconds']:.1f}"
                print(f"{version_name:<10} {mode:<10} {score:<12} {time_s:<10}")
            else:
                print(f"{version_name:<10} {mode:<10} {'ERROR':<12} {'-':<10}")

print(f"\n[OK] Results saved to {output_file}")
print(f"[OK] View map at: http://127.0.0.1:5000/full_comparison.html")
