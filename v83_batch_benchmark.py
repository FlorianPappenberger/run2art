"""
v83_batch_benchmark.py — Large-Scale v8.3 Experiment Runner
=============================================================
Runs 20-50 test configurations on the heart shape in Reading, UK.
Generates routes for each configuration and aggregates results into
an interactive comparison HTML and CSV for visual scoring.

Usage:
    python v83_batch_benchmark.py
    python v83_batch_benchmark.py --resume
    python v83_batch_benchmark.py --config custom_tests.json
"""

import json
import math
import os
import sys
import time
import argparse
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Constants ──
CENTER = [51.4543, -0.9781]
# v8.3 improved heart: deeper top indent, more control points, parametric curve
HEART_PTS = [
    [0.500, 0.350],   # top center indent (deep V)
    [0.562, 0.126],   # right lobe inner slope
    [0.677, 0.053],   # right lobe rising
    [0.825, 0.056],   # right lobe
    [0.951, 0.147],   # right lobe peak
    [1.000, 0.291],   # right descending
    [0.951, 0.445],   # right upper side
    [0.825, 0.588],   # right mid
    [0.677, 0.716],   # right lower
    [0.562, 0.830],   # approaching cusp right
    [0.509, 0.915],   # near cusp right
    [0.500, 0.947],   # bottom cusp
    [0.491, 0.915],   # near cusp left
    [0.438, 0.830],   # leaving cusp left
    [0.323, 0.716],   # left lower
    [0.175, 0.588],   # left mid
    [0.049, 0.445],   # left upper side
    [0.000, 0.291],   # left descending
    [0.049, 0.147],   # left lobe peak
    [0.175, 0.056],   # left lobe
    [0.323, 0.053],   # left lobe rising
    [0.437, 0.126],   # left lobe inner slope
    [0.500, 0.350],   # close
]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "v83_batch_results.json")
SCORES_CSV = os.path.join(RESULTS_DIR, "visual_scores.csv")
COMPARISON_HTML = os.path.join(RESULTS_DIR, "v83_comparison.html")


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT TEST CONFIGURATIONS (30 tests)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_TESTS = [
    # ── Baseline: new shape, no enhancements ──
    {"id": "v82_baseline_fit", "mode": "fit", "flags": {}, "label": "v8.2 Baseline Fit"},
    {"id": "v82_baseline_opt", "mode": "optimize", "flags": {}, "label": "v8.2 Baseline Opt"},

    # ── Indent enforcement (core fix) ──
    {"id": "indent_fit", "mode": "fit", "flags": {"indent_enforce": True}, "label": "Indent Enforce Fit"},
    {"id": "indent_opt", "mode": "optimize", "flags": {"indent_enforce": True}, "label": "Indent Enforce Opt"},

    # ── Indent + symmetry (user's preferred combo) ──
    {"id": "indent_sym05_fit", "mode": "fit", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "Indent+Sym0.5 Fit"},
    {"id": "indent_sym05_opt", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "Indent+Sym0.5 Opt"},

    # ── Indent + symmetry + close ──
    {"id": "indent_sym05_close_fit", "mode": "fit", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "force_close": True}, "label": "Indent+Sym0.5+Close Fit"},
    {"id": "indent_sym05_close_opt", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "force_close": True}, "label": "Indent+Sym0.5+Close Opt"},

    # ── Indent + symmetry + penalty variations ──
    {"id": "indent_sym05_pen15", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "penalty_factor": 1.5}, "label": "Indent+Sym+Pen1.5"},
    {"id": "indent_sym05_pen20", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "penalty_factor": 2.0}, "label": "Indent+Sym+Pen2.0"},
    {"id": "indent_sym05_pen30", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "penalty_factor": 3.0}, "label": "Indent+Sym+Pen3.0"},

    # ── Indent + symmetry + dynamic densify ──
    {"id": "indent_sym05_dyn", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "dynamic_densify": True}, "label": "Indent+Sym+Dynamic"},
    {"id": "indent_sym05_dyn_close", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "dynamic_densify": True, "force_close": True}, "label": "Indent+Sym+Dyn+Close"},

    # ── Indent + symmetry + spline ──
    {"id": "indent_sym05_spline3", "mode": "fit", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "spline_k": 3}, "label": "Indent+Sym+Spline3"},

    # ── Indent-only variations (no symmetry) ──
    {"id": "indent_pen20_opt", "mode": "optimize", "flags": {"indent_enforce": True, "indent_weight": 0.8, "penalty_factor": 2.0}, "label": "Indent+Pen2.0 Opt"},
    {"id": "indent_close_opt", "mode": "optimize", "flags": {"indent_enforce": True, "indent_weight": 0.8, "force_close": True}, "label": "Indent+Close Opt"},

    # ── Best combos from v8.3 round 1 (now with new shape) ──
    {"id": "sym05_opt", "mode": "optimize", "flags": {"symmetry_weight": 0.5}, "label": "Sym0.5 Opt (new shape)"},
    {"id": "pen30_opt", "mode": "optimize", "flags": {"penalty_factor": 3.0}, "label": "Pen3.0 Opt (new shape)"},
    {"id": "v6prox02_opt", "mode": "optimize", "flags": {"v6_proximity_weight": 0.2}, "label": "v6 Prox 0.2 (new shape)"},

    # ── Kitchen sink v8.3: all features ──
    {"id": "full_heart_fit", "mode": "fit", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "force_close": True, "dynamic_densify": True, "spline_k": 3}, "label": "Full Heart Fit"},
    {"id": "full_heart_opt", "mode": "optimize", "flags": {"indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "force_close": True, "dynamic_densify": True}, "label": "Full Heart Opt"},

    # ═══════════════════════════════════════════════════════════════════
    #  v8.4 PERCEPTUAL & ROAD-AWARE LAYER
    # ═══════════════════════════════════════════════════════════════════

    # ── Road-Density Auto-Scaling (alone) ──
    {"id": "v84_density_fit", "mode": "fit", "flags": {"density_auto_scale": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.4 Density+Indent Fit"},
    {"id": "v84_density_opt", "mode": "optimize", "flags": {"density_auto_scale": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.4 Density+Indent Opt"},

    # ── Road-Hierarchy Bonus (alone) ──
    {"id": "v84_hierarchy_fit", "mode": "fit", "flags": {"use_road_hierarchy": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.4 Hierarchy+Indent Fit"},
    {"id": "v84_hierarchy_opt", "mode": "optimize", "flags": {"use_road_hierarchy": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.4 Hierarchy+Indent Opt"},

    # ── Skeleton Score (alone) ──
    {"id": "v84_skel_fit", "mode": "fit", "flags": {"use_skeleton_score": True, "skeleton_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Skeleton Fit"},
    {"id": "v84_skel_opt", "mode": "optimize", "flags": {"use_skeleton_score": True, "skeleton_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Skeleton Opt"},

    # ── Persistent Homology (alone) ──
    {"id": "v84_ph_fit", "mode": "fit", "flags": {"use_ph_topology": True, "ph_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 PH Topology Fit"},
    {"id": "v84_ph_opt", "mode": "optimize", "flags": {"use_ph_topology": True, "ph_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 PH Topology Opt"},

    # ── FGW Score (alone) ──
    {"id": "v84_fgw_opt", "mode": "optimize", "flags": {"use_fgw": True, "fgw_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 FGW Opt"},

    # ── Perceptual Loss (alone) ──
    {"id": "v84_percep_opt", "mode": "optimize", "flags": {"use_perceptual_loss": True, "perceptual_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Perceptual Opt"},

    # ── Combo: Skeleton + Hierarchy ──
    {"id": "v84_skel_hier_fit", "mode": "fit", "flags": {"use_skeleton_score": True, "skeleton_weight": 0.3, "use_road_hierarchy": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Skel+Hier Fit"},
    {"id": "v84_skel_hier_opt", "mode": "optimize", "flags": {"use_skeleton_score": True, "skeleton_weight": 0.3, "use_road_hierarchy": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Skel+Hier Opt"},

    # ── Combo: PH + Skeleton ──
    {"id": "v84_ph_skel_opt", "mode": "optimize", "flags": {"use_ph_topology": True, "ph_weight": 0.2, "use_skeleton_score": True, "skeleton_weight": 0.2, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 PH+Skel Opt"},

    # ── Combo: Density + Hierarchy + Skeleton ──
    {"id": "v84_dens_hier_skel", "mode": "optimize", "flags": {"density_auto_scale": True, "use_road_hierarchy": True, "use_skeleton_score": True, "skeleton_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Dens+Hier+Skel"},

    # ── Combo: Perceptual + PH ──
    {"id": "v84_percep_ph_opt", "mode": "optimize", "flags": {"use_perceptual_loss": True, "perceptual_weight": 0.2, "use_ph_topology": True, "ph_weight": 0.2, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Percep+PH Opt"},

    # ── Combo: Hierarchy + Dynamic Densify ──
    {"id": "v84_hier_dyn_opt", "mode": "optimize", "flags": {"use_road_hierarchy": True, "dynamic_densify": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.4 Hier+Dynamic Opt"},

    # ── Combo: Skeleton + Spline smoothing ──
    {"id": "v84_skel_spline_fit", "mode": "fit", "flags": {"use_skeleton_score": True, "skeleton_weight": 0.3, "spline_k": 3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.4 Skel+Spline Fit"},

    # ── v8.4 blend sweep ──
    {"id": "v84_skel_b20", "mode": "optimize", "flags": {"use_skeleton_score": True, "skeleton_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.2}, "label": "v8.4 Skel blend=0.2"},
    {"id": "v84_skel_b40", "mode": "optimize", "flags": {"use_skeleton_score": True, "skeleton_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.4}, "label": "v8.4 Skel blend=0.4"},

    # ── Penalty sweep combos with v8.4 ──
    {"id": "v84_hier_pen20", "mode": "optimize", "flags": {"use_road_hierarchy": True, "penalty_factor": 2.0, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.4 Hier+Pen2.0"},
    {"id": "v84_hier_pen30", "mode": "optimize", "flags": {"use_road_hierarchy": True, "penalty_factor": 3.0, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.4 Hier+Pen3.0"},

    # ── Full v8.4 kitchen sink ──
    {"id": "v84_full_fit", "mode": "fit", "flags": {"density_auto_scale": True, "use_road_hierarchy": True, "use_skeleton_score": True, "skeleton_weight": 0.2, "use_ph_topology": True, "ph_weight": 0.2, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "force_close": True, "v84_blend": 0.3}, "label": "v8.4 Full Fit"},
    {"id": "v84_full_opt", "mode": "optimize", "flags": {"density_auto_scale": True, "use_road_hierarchy": True, "use_skeleton_score": True, "skeleton_weight": 0.2, "use_ph_topology": True, "ph_weight": 0.2, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "force_close": True, "dynamic_densify": True, "v84_blend": 0.3}, "label": "v8.4 Full Opt"},

    # ═══════════════════════════════════════════════════════════════════
    #  v8.5 WIDE-AREA MULTI-SCALE SEARCH
    # ═══════════════════════════════════════════════════════════════════

    # ── Wide search: default 4km range, standard features ──
    {"id": "v85_wide_default", "mode": "optimize", "flags": {"wide_search": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.5 Wide Default"},

    # ── Wide search: with hierarchy bonus ──
    {"id": "v85_wide_hier", "mode": "optimize", "flags": {"wide_search": True, "use_road_hierarchy": True, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.5 Wide+Hierarchy"},

    # ── Wide search: with penalty 2.0 ──
    {"id": "v85_wide_pen20", "mode": "optimize", "flags": {"wide_search": True, "penalty_factor": 2.0, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.5 Wide+Pen2.0"},

    # ── Wide search: with hierarchy + penalty 3.0 (best v8.4 combo) ──
    {"id": "v85_wide_hier_pen30", "mode": "optimize", "flags": {"wide_search": True, "use_road_hierarchy": True, "penalty_factor": 3.0, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.5 Wide+Hier+Pen3.0"},

    # ── Wide search: smaller km range (2km) for faster run ──
    {"id": "v85_wide_2km", "mode": "optimize", "flags": {"wide_search": True, "wide_km_range": 2.0, "wide_offset_steps": 3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.5 Wide 2km"},

    # ── Wide search: extended 6km range, more candidates ──
    {"id": "v85_wide_6km", "mode": "optimize", "flags": {"wide_search": True, "wide_km_range": 6.0, "wide_offset_steps": 5, "wide_n_coarse": 300, "wide_n_refine": 40, "wide_n_fine": 15, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5}, "label": "v8.5 Wide 6km"},

    # ── Wide search: hierarchy + skeleton scoring ──
    {"id": "v85_wide_hier_skel", "mode": "optimize", "flags": {"wide_search": True, "use_road_hierarchy": True, "use_skeleton_score": True, "skeleton_weight": 0.3, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.5 Wide+Hier+Skel"},

    # ── Wide search: full features ──
    {"id": "v85_wide_full", "mode": "optimize", "flags": {"wide_search": True, "use_road_hierarchy": True, "use_skeleton_score": True, "skeleton_weight": 0.2, "use_ph_topology": True, "ph_weight": 0.2, "indent_enforce": True, "symmetry_weight": 0.5, "indent_weight": 0.5, "v84_blend": 0.3}, "label": "v8.5 Wide Full"},
]


def run_single_test(test_config, G, kd):
    """Run a single test configuration and return results."""
    from engine import HANDLERS, fit_and_score, make_result
    from v83_enhancements import fit_and_score_v83, hybrid_v6_coarse_search
    from scoring_v8 import score_v8
    from geometry import shape_to_latlngs, adaptive_densify

    test_id = test_config["id"]
    mode = test_config["mode"]
    flags = test_config.get("flags", {})
    label = test_config.get("label", test_id)

    log(f"\n{'─'*60}")
    log(f"[{test_id}] {label}")
    log(f"  Mode: {mode}  Flags: {flags}")
    log(f"{'─'*60}")

    t0 = time.time()

    # Handle abstract mode separately
    if mode in ("abstract_fit", "abstract_optimize"):
        payload = {
            "mode": mode,
            "shapes": [{"name": "Heart", "pts": HEART_PTS}],
            "shape_index": 0,
            "center_point": CENTER,
        }
        try:
            result = HANDLERS[mode](payload)
            elapsed = time.time() - t0
            return {
                "id": test_id,
                "label": label,
                "mode": mode,
                "flags": flags,
                "score": result.get("score"),
                "route": result.get("route", []),
                "rotation": result.get("rotation"),
                "scale": result.get("scale"),
                "center": result.get("center", CENTER),
                "route_length_m": result.get("route_length_m"),
                "route_points": len(result.get("route", [])),
                "time_seconds": round(elapsed, 1),
                "error": result.get("error"),
            }
        except Exception as e:
            return {
                "id": test_id, "label": label, "mode": mode, "flags": flags,
                "error": str(e), "time_seconds": round(time.time() - t0, 1),
                "score": None, "route": [],
            }

    # v8.3/v8.4 enhanced pipeline or standard v8.2
    # ── v8.5 wide search mode ──
    use_wide = flags.get('wide_search', False)
    if use_wide:
        from v85_wide_search import wide_search_pipeline
        best_score, best_route, best_params = wide_search_pipeline(
            G, HEART_PTS, CENTER, kdtree_data=kd, config=flags)
        elapsed = time.time() - t0

        # Compute route length
        length_m = 0
        if best_route and len(best_route) >= 2:
            from geometry import haversine
            for i in range(len(best_route) - 1):
                length_m += haversine(best_route[i][0], best_route[i][1],
                                      best_route[i+1][0], best_route[i+1][1])

        # Human recognizability score
        hr_score, hr_explain = None, None
        if best_route and len(best_route) >= 10:
            from v85_wide_search import route_heart_recognizability
            hr_score, hr_explain = route_heart_recognizability(best_route)

        return {
            "id": test_id, "label": label, "mode": mode, "flags": flags,
            "score": round(best_score, 1) if best_score < 1e8 else None,
            "route": best_route or [],
            "rotation": best_params.get("rotation"),
            "scale": best_params.get("scale"),
            "center": best_params.get("center", CENTER),
            "route_length_m": round(length_m),
            "route_points": len(best_route) if best_route else 0,
            "time_seconds": round(elapsed, 1),
            "heart_recognizability": hr_score,
            "hr_explanation": hr_explain,
            "error": None if best_route else "No route found",
        }

    V83_V84_BOOL_FLAGS = [
        'dynamic_densify', 'spline_k', 'multi_res', 'force_close', 'indent_enforce',
        'density_auto_scale', 'use_road_hierarchy', 'use_skeleton_score',
        'use_fgw', 'use_perceptual_loss', 'use_ph_topology', 'wide_search',
    ]
    has_v83_flags = any(flags.get(k) for k in V83_V84_BOOL_FLAGS) \
        or flags.get('symmetry_weight', 0) > 0 \
        or flags.get('penalty_factor', 1.0) != 1.0 \
        or flags.get('v6_proximity_weight', 0) > 0 \
        or flags.get('indent_weight', 0) > 0 \
        or flags.get('skeleton_weight', 0) > 0 \
        or flags.get('fgw_weight', 0) > 0 \
        or flags.get('perceptual_weight', 0) > 0 \
        or flags.get('ph_weight', 0) > 0

    use_hybrid = flags.get('hybrid_v6', False)

    # For pure v8.2 baselines (no v8.3 flags), delegate to engine HANDLERS directly
    if not has_v83_flags and not use_hybrid:
        payload = {
            "mode": mode,
            "shapes": [{"name": "Heart", "pts": HEART_PTS}],
            "shape_index": 0,
            "center_point": CENTER,
        }
        if mode in HANDLERS:
            result = HANDLERS[mode](payload)
            elapsed = time.time() - t0
            rt = result.get("route", [])
            hr_score, hr_explain = None, None
            if rt and len(rt) >= 10:
                from v85_wide_search import route_heart_recognizability
                hr_score, hr_explain = route_heart_recognizability(rt)
            return {
                "id": test_id, "label": label, "mode": mode, "flags": flags,
                "score": result.get("score"),
                "route": rt,
                "rotation": result.get("rotation"),
                "scale": result.get("scale"),
                "center": result.get("center", CENTER),
                "route_length_m": result.get("route_length_m"),
                "route_points": len(rt),
                "time_seconds": round(elapsed, 1),
                "heart_recognizability": hr_score,
                "hr_explanation": hr_explain,
                "error": result.get("error"),
            }

    # v8.3 enhanced: coarse scan + v83 fine routing
    from engine import coarse_grid_search, _cusp_align_candidates, make_offsets

    if mode == "fit":
        # FIT MODE: lighter coarse scan
        rotations = list(range(0, 360, 45))
        scales = [0.010, 0.015, 0.020, 0.028]
        offsets = make_offsets(km_range=1.0, steps=2)

        if use_hybrid:
            coarse = hybrid_v6_coarse_search(G, HEART_PTS, CENTER, kdtree_data=kd,
                                             penalty_factor=flags.get('penalty_factor', 1.5))
        else:
            coarse = coarse_grid_search(G, HEART_PTS, CENTER, rotations, scales,
                                        offsets, densify_spacing=150, kdtree_data=kd)

        # Top-3 candidates, reduced variation grid
        seen, kept = set(), []
        for _, rot, sc, c in coarse:
            key = (round(rot, 0), round(sc, 4), round(c[0], 4), round(c[1], 4))
            if key not in seen:
                seen.add(key)
                kept.append((rot, sc, c))
            if len(kept) >= 3:
                break

        best_score, best_route, best_params = 1e9, None, {}
        for rot, sc, c in kept:
            for dr in [0, -15, 15]:
                for sf in [1.0, 0.90, 1.10]:
                    r2, s2 = rot + dr, sc * sf
                    score, route = fit_and_score_v83(
                        G, HEART_PTS, r2, s2, c,
                        config=flags, kdtree_data=kd)
                    if route and score < best_score:
                        best_score = score
                        best_route = route
                        best_params = {"rotation": r2, "scale": s2, "center": c}

    elif mode == "optimize":
        # OPTIMIZE MODE: broader coarse + top-5 fine
        rotations = list(range(0, 360, 15))
        scales = [0.010, 0.014, 0.018, 0.022, 0.027, 0.033, 0.040]
        offsets = make_offsets(km_range=2.0, steps=3)

        if use_hybrid:
            coarse = hybrid_v6_coarse_search(G, HEART_PTS, CENTER, kdtree_data=kd,
                                             penalty_factor=flags.get('penalty_factor', 1.5))
        else:
            coarse = coarse_grid_search(G, HEART_PTS, CENTER, rotations, scales,
                                        offsets, densify_spacing=200, kdtree_data=kd)

        cusp_aligned = _cusp_align_candidates(HEART_PTS, coarse[:20], kd)
        merged = coarse[:50] + cusp_aligned

        seen, kept = set(), []
        for _, rot, sc, c in merged:
            key = (round(rot, 0), round(sc, 4), round(c[0], 4), round(c[1], 4))
            if key not in seen:
                seen.add(key)
                kept.append((rot, sc, c))
            if len(kept) >= 5:
                break

        best_score, best_route, best_params = 1e9, None, {}
        for rot, sc, c in kept:
            for dr in [0, -7, 7, -15, 15]:
                for sf in [1.0, 0.88, 1.12]:
                    r2, s2 = rot + dr, sc * sf
                    score, route = fit_and_score_v83(
                        G, HEART_PTS, r2, s2, c,
                        config=flags, kdtree_data=kd)
                    if route and score < best_score:
                        best_score = score
                        best_route = route
                        best_params = {"rotation": r2, "scale": s2, "center": c}

    else:
        return {
            "id": test_id, "label": label, "mode": mode, "flags": flags,
            "error": f"Unknown mode: {mode}", "time_seconds": 0,
            "score": None, "route": [],
        }

    elapsed = time.time() - t0

    # Compute route length
    length_m = 0
    if best_route and len(best_route) >= 2:
        from geometry import haversine
        for i in range(len(best_route) - 1):
            length_m += haversine(best_route[i][0], best_route[i][1],
                                  best_route[i+1][0], best_route[i+1][1])

    # Human recognizability score
    hr_score, hr_explain = None, None
    if best_route and len(best_route) >= 10:
        from v85_wide_search import route_heart_recognizability
        hr_score, hr_explain = route_heart_recognizability(best_route)

    return {
        "id": test_id,
        "label": label,
        "mode": mode,
        "flags": flags,
        "score": round(best_score, 1) if best_score < 1e8 else None,
        "route": best_route or [],
        "rotation": best_params.get("rotation"),
        "scale": best_params.get("scale"),
        "center": best_params.get("center", CENTER),
        "route_length_m": round(length_m),
        "route_points": len(best_route) if best_route else 0,
        "time_seconds": round(elapsed, 1),
        "heart_recognizability": hr_score,
        "hr_explanation": hr_explain,
        "error": None if best_route else "No route found",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  HTML COMPARISON MAP GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_comparison_html(results, output_path):
    """Generate an interactive Leaflet map with all test routes."""

    # Assign colors from a palette
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
        "#8bc34a", "#ff5722", "#607d8b", "#795548", "#cddc39",
        "#ff9800", "#673ab7", "#009688", "#f44336", "#2196f3",
        "#4caf50", "#ffc107", "#9c27b0", "#00e5ff", "#76ff03",
        "#ff6f00", "#283593", "#00695c", "#d50000", "#304ffe",
    ]

    routes_data = {}
    config_data = {}
    for i, r in enumerate(results):
        rid = r["id"]
        routes_data[rid] = {
            "route": r.get("route", []),
            "score": r.get("score"),
            "time_seconds": r.get("time_seconds"),
            "route_length_m": r.get("route_length_m"),
            "rotation": r.get("rotation"),
            "scale": r.get("scale"),
            "route_points": r.get("route_points", 0),
            "mode": r.get("mode"),
            "flags": r.get("flags", {}),
        }
        color = palette[i % len(palette)]
        is_baseline = "baseline" in rid
        config_data[rid] = {
            "color": color,
            "weight": 5 if is_baseline else 3,
            "dash": "" if "fit" in r.get("mode", "") else "8,5",
            "label": r.get("label", rid),
        }

    routes_json = json.dumps(routes_data)
    config_json = json.dumps(config_data)

    # Ideal heart overlay coords (using typical best params)
    from geometry import shape_to_latlngs, rotate_shape
    ideal_coords = shape_to_latlngs(
        [[.50,.14],[.66,-.04],[.90,-.06],[1.0,.18],[.94,.46],
         [.76,.70],[.50,1.0],[.24,.70],[.06,.46],[0,.18],
         [.10,-.06],[.34,-.04],[.50,.14]],
        CENTER, 0.01, 0)
    ideal_json = json.dumps(ideal_coords)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>v8.3 Experiment Comparison — Heart Shape, Reading UK</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;display:flex}}
  #map{{flex:1;height:100vh}}
  .panel{{width:380px;height:100vh;overflow-y:auto;background:#fafafa;border-left:1px solid #ddd;padding:12px;font-size:13px}}
  .panel h3{{margin:0 0 8px;font-size:16px}}
  .sub{{color:#888;font-size:11px;margin-bottom:10px}}
  .section{{margin:12px 0 4px;font-weight:700;font-size:11px;color:#555;border-top:1px solid #eee;padding-top:6px;text-transform:uppercase}}
  .row{{display:flex;align-items:center;padding:3px 2px;cursor:pointer}}
  .row:hover{{background:#f0f0f0;border-radius:3px}}
  .row input{{margin-right:6px}}
  .swatch{{display:inline-block;width:24px;height:0;margin-right:8px;flex-shrink:0}}
  .lbl{{flex:1;font-size:12px}}
  .sc{{font-size:11px;color:#666;font-weight:700;margin-left:4px;white-space:nowrap}}
  .btns{{margin:8px 0}}
  .btns button{{margin:2px 4px 2px 0;padding:3px 10px;font-size:11px;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#fff}}
  .btns button:hover{{background:#eee}}
  table.scores{{width:100%;border-collapse:collapse;margin-top:8px;font-size:11px}}
  table.scores th,table.scores td{{border:1px solid #ddd;padding:3px 5px;text-align:left}}
  table.scores th{{background:#f5f5f5}}
  .best{{background:#e8f5e9;font-weight:bold}}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h3>v8.3 Experiments</h3>
  <div class="sub">Heart Shape &bull; Reading, UK &bull; {len(results)} tests</div>
  <div class="btns">
    <button onclick="toggleAll(true)">Show All</button>
    <button onclick="toggleAll(false)">Hide All</button>
    <button onclick="showOnly('baseline')">Baselines</button>
    <button onclick="showOnly('fit')">Fit Only</button>
    <button onclick="showOnly('opt')">Optimize Only</button>
  </div>
  <div id="legend"></div>
  <div class="section">Score Summary</div>
  <table class="scores" id="scoreTable">
    <tr><th>Test</th><th>Score</th><th>Time</th><th>Pts</th></tr>
  </table>
</div>
<script>
const map=L.map('map').setView([51.4543,-0.9781],14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{maxZoom:19}}).addTo(map);
const routes={routes_json};
const config={config_json};
const ideal={ideal_json};
L.polyline(ideal,{{color:'#ff0066',weight:2,opacity:0.4,dashArray:'6,4'}}).addTo(map);
const layers={{}};const allBounds=[];
Object.keys(routes).forEach(k=>{{
  const d=routes[k],cfg=config[k];
  if(!cfg||!d.route||d.route.length<2)return;
  const ll=d.route.map(p=>[p[0],p[1]]);
  const poly=L.polyline(ll,{{color:cfg.color,weight:cfg.weight,opacity:0.85,dashArray:cfg.dash||null}});
  poly.bindPopup('<b>'+cfg.label+'</b><br>Score: '+(d.score!=null?d.score.toFixed(1)+'m':'?')+'<br>Time: '+(d.time_seconds||'?')+'s<br>Mode: '+d.mode+'<br>Points: '+d.route.length+'<br>Flags: '+JSON.stringify(d.flags));
  poly.addTo(map);layers[k]=poly;allBounds.push(...ll);
}});
if(allBounds.length)map.fitBounds(L.latLngBounds(allBounds),{{padding:[50,50]}});

// Legend
const legend=document.getElementById('legend');
const sorted=Object.keys(routes).sort((a,b)=>(routes[a].score||1e9)-(routes[b].score||1e9));
let bestScore=1e9;
sorted.forEach(k=>{{
  const d=routes[k],cfg=config[k];if(!cfg)return;
  const row=document.createElement('label');row.className='row';row.dataset.key=k;
  const cb=document.createElement('input');cb.type='checkbox';cb.checked=!!d.route?.length;cb.dataset.key=k;
  cb.addEventListener('change',()=>{{if(layers[k]){{if(cb.checked)map.addLayer(layers[k]);else map.removeLayer(layers[k])}}}});
  const sw=document.createElement('span');sw.className='swatch';
  sw.style.borderTop=cfg.weight+'px '+(cfg.dash?'dashed':'solid')+' '+cfg.color;
  const lb=document.createElement('span');lb.className='lbl';lb.textContent=cfg.label;
  const sc=document.createElement('span');sc.className='sc';
  sc.textContent=d.score!=null?d.score.toFixed(1)+'m':'—';
  if(!d.route?.length){{cb.disabled=true;row.style.opacity='0.4'}}
  row.appendChild(cb);row.appendChild(sw);row.appendChild(lb);row.appendChild(sc);
  legend.appendChild(row);
}});

// Score table
const tbl=document.getElementById('scoreTable');
sorted.forEach(k=>{{
  const d=routes[k],cfg=config[k];if(!cfg)return;
  const tr=document.createElement('tr');
  if(d.score!=null&&d.score<bestScore){{bestScore=d.score;tr.className='best'}}
  tr.innerHTML='<td>'+cfg.label+'</td><td>'+(d.score!=null?d.score.toFixed(1):'—')+'</td><td>'+(d.time_seconds||'—')+'s</td><td>'+(d.route?.length||0)+'</td>';
  tbl.appendChild(tr);
}});

function toggleAll(on){{
  document.querySelectorAll('.row input').forEach(cb=>{{cb.checked=on;if(layers[cb.dataset.key]){{if(on)map.addLayer(layers[cb.dataset.key]);else map.removeLayer(layers[cb.dataset.key])}}}});
}}
function showOnly(filter){{
  document.querySelectorAll('.row input').forEach(cb=>{{
    const k=cb.dataset.key;
    const show=filter==='baseline'?k.includes('baseline'):filter==='fit'?routes[k]?.mode==='fit':routes[k]?.mode==='optimize';
    cb.checked=show;if(layers[k]){{if(show)map.addLayer(layers[k]);else map.removeLayer(layers[k])}}
  }});
}}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"[HTML] Saved comparison map → {output_path}")


def generate_visual_scores_csv(results, output_path):
    """Generate a CSV template for manual visual scoring."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_id", "label", "mode", "flags",
            "auto_score", "time_s", "route_pts", "route_length_m",
            "recognizability_1to10", "jaggedness_1to10_low_is_good",
            "symmetry_1to10", "closure_1to10", "proportions_1to10",
            "notes"
        ])
        for r in sorted(results, key=lambda x: x.get("score") or 1e9):
            writer.writerow([
                r["id"], r.get("label", ""), r.get("mode", ""),
                json.dumps(r.get("flags", {})),
                r.get("score", ""), r.get("time_seconds", ""),
                r.get("route_points", ""), r.get("route_length_m", ""),
                "", "", "", "", "",  # blank for manual scoring
                ""
            ])
    log(f"[CSV] Saved visual scoring template → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v8.3 Batch Benchmark Runner")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--config", type=str, help="Custom test config JSON file")
    parser.add_argument("--test", type=str, help="Run only a specific test ID")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test configurations
    if args.config:
        with open(args.config) as f:
            tests = json.load(f)
    else:
        tests = DEFAULT_TESTS

    # Resume support — load from full results (with routes)
    completed = {}
    full_path = os.path.join(RESULTS_DIR, "v83_full_results.json")
    if args.resume and os.path.exists(full_path):
        with open(full_path) as f:
            prev = json.load(f)
        for r in prev.get("results", []):
            if r.get("route") and len(r.get("route", [])) > 1:
                completed[r["id"]] = r
        log(f"[resume] Loaded {len(completed)} completed tests (with routes)")

    # Filter to single test if requested
    if args.test:
        tests = [t for t in tests if t["id"] == args.test]
        if not tests:
            log(f"[ERROR] Test '{args.test}' not found")
            return

    # Fetch graph once (shared across all tests)
    log("Loading road network graph...")
    from routing import fetch_graph, build_kdtree
    G = fetch_graph(CENTER, dist=4000)
    if G is None:
        log("[ERROR] Could not fetch graph. Aborting.")
        return
    kd = build_kdtree(G)
    log(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Run tests
    results = []
    total = len(tests)
    for i, test in enumerate(tests, 1):
        tid = test["id"]

        if tid in completed and args.resume:
            log(f"[{i}/{total}] Skipping {tid} (already done)")
            results.append(completed[tid])
            continue

        log(f"\n{'═'*60}")
        log(f"  TEST {i}/{total}: {test.get('label', tid)}")
        log(f"{'═'*60}")

        try:
            result = run_single_test(test, G, kd)
        except Exception as e:
            log(f"[ERROR] Test {tid} crashed: {e}")
            result = {
                "id": tid, "label": test.get("label", tid),
                "mode": test.get("mode"), "flags": test.get("flags", {}),
                "error": str(e), "score": None, "route": [],
                "time_seconds": 0,
            }

        results.append(result)

        # Log result summary
        if result.get("score") is not None:
            log(f"  → Score: {result['score']}  Time: {result['time_seconds']}s  "
                f"Points: {result.get('route_points', 0)}")
        else:
            log(f"  → FAILED: {result.get('error', 'unknown')}")

        # Save incrementally (slim — no routes, just scores)
        slim_results = []
        for r in results:
            slim = dict(r)
            slim["route"] = []
            slim_results.append(slim)
        with open(RESULTS_FILE, "w") as f:
            json.dump({
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "location": "Reading, UK", "center": CENTER,
                "total_tests": total, "completed": len(results),
                "results": slim_results,
            }, f, indent=2)

        # Also save full results with routes incrementally
        full_path = os.path.join(RESULTS_DIR, "v83_full_results.json")
        with open(full_path, "w") as f:
            json.dump({
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "location": "Reading, UK", "center": CENTER,
                "total_tests": total, "completed": len(results),
                "results": results,
            }, f)

    # Final outputs
    log(f"\n{'═'*60}")
    log(f"  ALL {total} TESTS COMPLETE")
    log(f"{'═'*60}")

    # Save full results with routes
    full_save = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Reading, UK",
        "center": CENTER,
        "total_tests": total,
        "completed": len(results),
        "results": results,
    }
    full_path = os.path.join(RESULTS_DIR, "v83_full_results.json")
    with open(full_path, "w") as f:
        json.dump(full_save, f)
    log(f"Full results saved → {full_path}")

    # Generate comparison HTML
    generate_comparison_html(results, COMPARISON_HTML)

    # Generate visual scoring CSV template
    generate_visual_scores_csv(results, SCORES_CSV)

    # Print score summary table
    log(f"\n{'─'*80}")
    log(f"{'Test ID':<30} {'Score':>8} {'HR':>5} {'Time':>8} {'Pts':>6}")
    log(f"{'─'*80}")
    for r in sorted(results, key=lambda x: x.get("score") or 1e9):
        sc = f"{r['score']:.1f}" if r.get("score") else "FAIL"
        tm = f"{r['time_seconds']:.0f}s" if r.get("time_seconds") else "—"
        pts = str(r.get("route_points", 0))
        hr = f"{r['heart_recognizability']:.1f}" if r.get("heart_recognizability") is not None else "—"
        log(f"{r['id']:<30} {sc:>8} {hr:>5} {tm:>8} {pts:>6}")


if __name__ == "__main__":
    main()
