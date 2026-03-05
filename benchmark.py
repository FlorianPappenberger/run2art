"""
benchmark.py — Run2Art Batch Benchmark
=======================================
Precomputes routes for all 20 shapes on Reading, UK using all 3 modes.
Saves results + statistics to public/benchmark_results.json for the UI.

Usage:  python benchmark.py
"""

import json
import time
import sys
import os

# Ensure engine is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

# ── Reading, UK centre ──────────────────────────────────
CENTER = [51.4543, -0.9781]

# ── All 20 shape templates (must match index.html) ─────
SHAPES = [
    {"name": "Heart",          "pts": [[.50,.14],[.66,-.04],[.90,-.06],[1.0,.18],[.94,.46],[.76,.70],[.50,1.0],[.24,.70],[.06,.46],[0,.18],[.10,-.06],[.34,-.04],[.50,.14]]},
    {"name": "Star",           "pts": [[.50,.00],[.62,.34],[1.0,.38],[.72,.60],[.80,1.0],[.50,.76],[.20,1.0],[.28,.60],[0,.38],[.38,.34],[.50,.00]]},
    {"name": "Smiley",         "pts": [[.50,.00],[.78,.08],[.96,.30],[1.0,.55],[.94,.78],[.74,.66],[.60,.76],[.50,.80],[.40,.76],[.26,.66],[.06,.78],[.00,.55],[.04,.30],[.22,.08],[.50,.00]]},
    {"name": "Peace Sign",     "pts": [[.50,.02],[.50,.50],[.22,.90],[.06,.72],[.02,.50],[.06,.28],[.22,.10],[.50,.02],[.78,.10],[.94,.28],[.98,.50],[.94,.72],[.78,.90],[.50,.50],[.50,.98],[.50,.02]]},
    {"name": "Cat",            "pts": [[.18,.00],[.10,.24],[.04,.42],[.02,.60],[.14,.78],[.34,.92],[.50,.98],[.66,.92],[.86,.78],[.98,.60],[.96,.42],[.90,.24],[.82,.00],[.72,.20],[.50,.14],[.28,.20],[.18,.00]]},
    {"name": "Bone",           "pts": [[.14,.28],[.06,.22],[.00,.30],[.00,.50],[.00,.70],[.06,.78],[.14,.72],[.30,.58],[.70,.58],[.86,.72],[.94,.78],[1.0,.70],[1.0,.50],[1.0,.30],[.94,.22],[.86,.28],[.70,.42],[.30,.42],[.14,.28]]},
    {"name": "Fish",           "pts": [[.00,.22],[.16,.44],[.00,.78],[.22,.64],[.44,.74],[.66,.74],[.84,.62],[1.0,.50],[.84,.38],[.66,.26],[.44,.26],[.22,.36],[.00,.22]]},
    {"name": "Diamond",        "pts": [[.50,.00],[.82,.12],[1.0,.28],[.84,.28],[.62,.58],[.50,1.0],[.38,.58],[.16,.28],[.00,.28],[.18,.12],[.50,.00]]},
    {"name": "Butterfly",      "pts": [[.50,.50],[.70,.15],[1.0,.00],[.95,.45],[.75,.50],[.95,.55],[1.0,1.0],[.70,.85],[.50,.50],[.30,.85],[0,1.0],[.05,.55],[.25,.50],[.05,.45],[0,.00],[.30,.15],[.50,.50]]},
    {"name": "Flower",         "pts": [[.50,.08],[.62,.24],[.82,.12],[.76,.38],[.94,.50],[.76,.62],[.82,.88],[.62,.76],[.50,.92],[.38,.76],[.18,.88],[.24,.62],[.06,.50],[.24,.38],[.18,.12],[.38,.24],[.50,.08]]},
    {"name": "Christmas Tree", "pts": [[.50,.00],[.70,.30],[.60,.30],[.80,.55],[.65,.55],[.85,.85],[.60,.85],[.60,1.0],[.40,1.0],[.40,.85],[.15,.85],[.35,.55],[.20,.55],[.40,.30],[.30,.30],[.50,.00]]},
    {"name": "Crescent Moon",  "pts": [[.82,.02],[.56,.08],[.32,.22],[.14,.42],[.10,.65],[.20,.84],[.42,.96],[.72,.98],[.56,.82],[.38,.62],[.32,.42],[.42,.25],[.62,.12],[.82,.02]]},
    {"name": "Airplane",       "pts": [[.50,.00],[.54,.16],[.94,.38],[.58,.42],[.56,.62],[.72,.80],[.54,.76],[.50,.88],[.46,.76],[.28,.80],[.44,.62],[.42,.42],[.06,.38],[.46,.16],[.50,.00]]},
    {"name": "Sailboat",       "pts": [[.42,.00],[.84,.54],[.92,.54],[.96,.66],[.86,.82],[.50,.90],[.14,.82],[.04,.66],[.08,.54],[.42,.54],[.42,.00]]},
    {"name": "House",          "pts": [[.50,.00],[.72,.18],[.72,.06],[.82,.06],[.82,.26],[.96,.36],[.96,1.0],[.04,1.0],[.04,.36],[.50,.00]]},
    {"name": "Crown",          "pts": [[.05,.92],[.05,.38],[.22,.58],[.36,.14],[.50,.44],[.64,.14],[.78,.58],[.95,.38],[.95,.92],[.05,.92]]},
    {"name": "Thumbs Up",      "pts": [[.28,.00],[.42,.00],[.44,.14],[.46,.30],[.46,.42],[.70,.42],[.88,.46],[.96,.56],[.96,.70],[.86,.82],[.66,.90],[.46,.92],[.28,.86],[.18,.74],[.14,.56],[.14,.42],[.16,.26],[.20,.12],[.28,.00]]},
    {"name": "Arrow",          "pts": [[.50,.00],[1.0,.40],[.68,.40],[.68,1.0],[.32,1.0],[.32,.40],[0,.40],[.50,.00]]},
    {"name": "Lightning",      "pts": [[.35,.00],[.72,.38],[.52,.38],[.80,1.0],[.28,.58],[.48,.58],[.20,.00],[.35,.00]]},
    {"name": "Music Note",     "pts": [[.58,.00],[.80,.08],[.86,.24],[.68,.32],[.58,.24],[.58,.60],[.46,.72],[.28,.82],[.18,.94],[.32,1.0],[.48,.92],[.58,.78],[.58,.00]]},
]

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "public", "benchmark_results.json")


def save_incremental(results):
    """Save current results so the UI can load partial progress."""
    # Compute partial summary
    fit_scores = [r["score"] for r in results.get("fit", []) if r.get("score") and r["score"] < 1e6]
    opt_scores = [r["score"] for r in results.get("optimize", []) if r.get("score") and r["score"] < 1e6]
    fit_times = [r["time_seconds"] for r in results.get("fit", []) if "time_seconds" in r]
    opt_times = [r["time_seconds"] for r in results.get("optimize", []) if "time_seconds" in r]
    results["summary"] = {
        "fit_count": len(results.get("fit", [])),
        "fit_success": len(fit_scores),
        "fit_avg_score": round(sum(fit_scores) / len(fit_scores), 1) if fit_scores else None,
        "fit_best_score": round(min(fit_scores), 1) if fit_scores else None,
        "fit_worst_score": round(max(fit_scores), 1) if fit_scores else None,
        "fit_avg_time": round(sum(fit_times) / len(fit_times), 1) if fit_times else None,
        "fit_total_time": round(sum(fit_times), 1) if fit_times else 0,
        "opt_count": len(results.get("optimize", [])),
        "opt_success": len(opt_scores),
        "opt_avg_score": round(sum(opt_scores) / len(opt_scores), 1) if opt_scores else None,
        "opt_best_score": round(min(opt_scores), 1) if opt_scores else None,
        "opt_worst_score": round(max(opt_scores), 1) if opt_scores else None,
        "opt_avg_time": round(sum(opt_times) / len(opt_times), 1) if opt_times else None,
        "opt_total_time": round(sum(opt_times), 1) if opt_times else 0,
        "best_shape_name": results.get("best_shape", {}).get("shape_name") if results.get("best_shape") else None,
        "best_shape_score": results.get("best_shape", {}).get("score") if results.get("best_shape") else None,
        "best_shape_time": results.get("best_shape", {}).get("time_seconds") if results.get("best_shape") else None,
        "status": "in_progress",
    }
    results["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    log(f"  [saved partial results — {len(results.get('fit',[]))} fit, {len(results.get('optimize',[]))} opt]")


def run_mode(mode, shape_index=0, shapes=None):
    """Run a single engine mode and return result + timing."""
    if shapes is None:
        shapes = SHAPES
    payload = {
        "mode": mode,
        "shapes": shapes,
        "shape_index": shape_index,
        "center_point": CENTER,
    }
    t0 = time.time()
    result = engine.HANDLERS[mode](payload)
    elapsed = time.time() - t0
    result["time_seconds"] = round(elapsed, 2)
    result["mode"] = mode
    return result


def route_length_m(route):
    """Total route length in metres."""
    if not route or len(route) < 2:
        return 0
    total = 0
    for i in range(len(route) - 1):
        total += engine.haversine(route[i][0], route[i][1],
                                  route[i+1][0], route[i+1][1])
    return round(total)


def log(msg):
    print(msg, flush=True)


def load_existing():
    """Load previous results for resume support."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                data = json.load(f)
            log(f"[resume] Loaded existing results from {OUTPUT_FILE}")
            log(f"[resume]   fit: {len(data.get('fit', []))}/{len(SHAPES)},  "
                f"optimize: {len(data.get('optimize', []))}/{len(SHAPES)},  "
                f"best_shape: {'done' if data.get('best_shape') else 'pending'}")
            return data
        except (json.JSONDecodeError, KeyError):
            log("[resume] Existing file corrupt — starting fresh")
    return None


def main():
    t_total = time.time()

    # ── Resume support: load previous results if available ──
    prev = load_existing()
    if prev:
        results = {
            "location": prev.get("location", "Reading, UK"),
            "center": prev.get("center", CENTER),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fit": prev.get("fit", []),
            "optimize": prev.get("optimize", []),
            "best_shape": prev.get("best_shape"),
            "summary": {},
        }
    else:
        results = {
            "location": "Reading, UK",
            "center": CENTER,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fit": [],
            "optimize": [],
            "best_shape": None,
            "summary": {},
        }

    n = len(SHAPES)
    fit_done = len(results["fit"])
    opt_done = len(results["optimize"])
    bs_done = results["best_shape"] is not None

    # ── Phase 1: Quick Fit (all 20 shapes) ───────────────
    log(f"\n{'='*60}")
    log(f"  PHASE 1: Quick Fit — {n} shapes on Reading, UK")
    if fit_done:
        log(f"  (resuming — {fit_done}/{n} already done)")
    log(f"{'='*60}\n")

    for i in range(fit_done, n):
        shape = SHAPES[i]
        log(f"[fit {i+1}/{n}] {shape['name']}...")
        # Send only the single shape (engine expects index 0)
        r = run_mode("fit", shape_index=0, shapes=[shape])
        r["global_shape_index"] = i
        if r.get("route"):
            r["route_length_m"] = route_length_m(r["route"])
            r["route_points"] = len(r["route"])
            log(f"  → score={r.get('score','?')}m  "
                f"rot={r.get('rotation',0):.0f}°  "
                f"len={r['route_length_m']}m  "
                f"time={r['time_seconds']}s")
        else:
            log(f"  → FAILED: {r.get('error', 'unknown')}")
        results["fit"].append(r)
        save_incremental(results)

    # ── Phase 2: Auto-Optimize (all 20 shapes) ───────────
    log(f"\n{'='*60}")
    log(f"  PHASE 2: Auto-Optimize — {n} shapes")
    if opt_done:
        log(f"  (resuming — {opt_done}/{n} already done)")
    log(f"{'='*60}\n")

    for i in range(opt_done, n):
        shape = SHAPES[i]
        log(f"[optimize {i+1}/{n}] {shape['name']}...")
        r = run_mode("optimize", shape_index=0, shapes=[shape])
        r["global_shape_index"] = i
        if r.get("route"):
            r["route_length_m"] = route_length_m(r["route"])
            r["route_points"] = len(r["route"])
            log(f"  → score={r.get('score','?')}m  "
                f"rot={r.get('rotation',0):.0f}°  "
                f"len={r['route_length_m']}m  "
                f"time={r['time_seconds']}s")
        else:
            log(f"  → FAILED: {r.get('error', 'unknown')}")
        results["optimize"].append(r)
        save_incremental(results)

    # ── Phase 3: Find Best Shape ──────────────────────────
    log(f"\n{'='*60}")
    log(f"  PHASE 3: Find Best Shape (all {n} shapes)")
    log(f"{'='*60}\n")

    if bs_done:
        log("  (already done — skipping)")
        r = results["best_shape"]
    else:
        r = run_mode("best_shape", shape_index=0, shapes=SHAPES)
        if r.get("route"):
            r["route_length_m"] = route_length_m(r["route"])
            r["route_points"] = len(r["route"])
            log(f"  → Best: {r.get('shape_name','?')} (idx={r.get('shape_index','?')})")
            log(f"    score={r.get('score','?')}m  rot={r.get('rotation',0):.0f}°  "
                f"len={r['route_length_m']}m  time={r['time_seconds']}s")
        else:
            log(f"  → FAILED: {r.get('error', 'unknown')}")
        results["best_shape"] = r

    # ── Summary ───────────────────────────────────────────
    total_time = round(time.time() - t_total, 1)

    fit_scores = [r["score"] for r in results["fit"] if r.get("score") and r["score"] < 1e6]
    opt_scores = [r["score"] for r in results["optimize"] if r.get("score") and r["score"] < 1e6]
    fit_times = [r["time_seconds"] for r in results["fit"]]
    opt_times = [r["time_seconds"] for r in results["optimize"]]

    results["summary"] = {
        "total_time_seconds": total_time,
        "fit_count": len(results["fit"]),
        "fit_success": len(fit_scores),
        "fit_avg_score": round(sum(fit_scores) / len(fit_scores), 1) if fit_scores else None,
        "fit_best_score": round(min(fit_scores), 1) if fit_scores else None,
        "fit_worst_score": round(max(fit_scores), 1) if fit_scores else None,
        "fit_avg_time": round(sum(fit_times) / len(fit_times), 1) if fit_times else None,
        "fit_total_time": round(sum(fit_times), 1),
        "opt_count": len(results["optimize"]),
        "opt_success": len(opt_scores),
        "opt_avg_score": round(sum(opt_scores) / len(opt_scores), 1) if opt_scores else None,
        "opt_best_score": round(min(opt_scores), 1) if opt_scores else None,
        "opt_worst_score": round(max(opt_scores), 1) if opt_scores else None,
        "opt_avg_time": round(sum(opt_times) / len(opt_times), 1) if opt_times else None,
        "opt_total_time": round(sum(opt_times), 1),
        "best_shape_name": results["best_shape"].get("shape_name"),
        "best_shape_score": results["best_shape"].get("score"),
        "best_shape_time": results["best_shape"].get("time_seconds"),
    }

    # ── Save ──────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    log(f"\n{'='*60}")
    log(f"  BENCHMARK COMPLETE")
    log(f"{'='*60}")
    log(f"  Total time:  {total_time}s")
    log(f"  Quick Fit:   {len(fit_scores)}/{n} succeeded, avg score {results['summary']['fit_avg_score']}m, avg time {results['summary']['fit_avg_time']}s")
    log(f"  Optimize:    {len(opt_scores)}/{n} succeeded, avg score {results['summary']['opt_avg_score']}m, avg time {results['summary']['opt_avg_time']}s")
    log(f"  Best Shape:  {results['summary']['best_shape_name']} ({results['summary']['best_shape_score']}m)")
    log(f"  Saved to:    {OUTPUT_FILE}")
    log("")


if __name__ == "__main__":
    main()
