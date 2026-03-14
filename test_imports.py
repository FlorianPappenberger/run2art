"""Quick import test to verify all changes compile."""
try:
    from engine import mode_optimize, mode_fit, _cma_refine, _cusp_align_candidates
    print("engine.py OK")
except Exception as e:
    print(f"engine.py FAIL: {e}")

try:
    from scoring_v8 import score_v8, fourier_descriptor_score, _hausdorff_distance
    print("scoring_v8.py OK")
except Exception as e:
    print(f"scoring_v8.py FAIL: {e}")

try:
    from core_router import CoreRouter, precompute_edge_weights
    print("core_router.py OK")
except Exception as e:
    print(f"core_router.py FAIL: {e}")

try:
    from geometry import generate_heart_variants
    print("geometry.py OK")
except Exception as e:
    print(f"geometry.py FAIL: {e}")

try:
    from scoring import coarse_proximity_score
    print("scoring.py OK")
except Exception as e:
    print(f"scoring.py FAIL: {e}")

print("All done")
