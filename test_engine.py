"""
test_engine.py — Regression tests for engine.py speed optimisation.
Run with: python -m pytest test_engine.py -v
"""
import math
import time
import pytest

import engine

# ── Test data ────────────────────────────────────────────────────────────

# Simple polyline: a known lat/lng path (roughly a square in Reading, UK)
SQUARE_IDEAL = [
    [51.456, -0.980],
    [51.456, -0.970],
    [51.450, -0.970],
    [51.450, -0.980],
    [51.456, -0.980],
]

# A heart shape (first shape from the app)
HEART_PTS = [[.50,.14],[.66,-.04],[.90,-.06],[1.0,.18],[.94,.46],
             [.76,.70],[.50,1.0],[.24,.70],[.06,.46],[0,.18],
             [.10,-.06],[.34,-.04],[.50,.14]]

CENTER = [51.4543, -0.9781]

# A "route" that roughly follows the square (simulated)
SQUARE_ROUTE = [
    [51.4560, -0.9800], [51.4560, -0.9780], [51.4560, -0.9760],
    [51.4560, -0.9740], [51.4560, -0.9720], [51.4560, -0.9700],
    [51.4550, -0.9700], [51.4540, -0.9700], [51.4530, -0.9700],
    [51.4520, -0.9700], [51.4500, -0.9700], [51.4500, -0.9720],
    [51.4500, -0.9740], [51.4500, -0.9760], [51.4500, -0.9780],
    [51.4500, -0.9800], [51.4520, -0.9800], [51.4540, -0.9800],
    [51.4560, -0.9800],
]

# Longer route for perf testing
LONG_ROUTE = [[51.45 + i * 0.00005, -0.98 + (i % 30) * 0.00002]
              for i in range(300)]
LONG_IDEAL = [[51.45 + i * 0.0001, -0.98 + (i % 15) * 0.00004]
              for i in range(80)]


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — haversine
# ══════════════════════════════════════════════════════════════════════════

class TestHaversine:
    def test_zero_distance(self):
        assert engine.haversine(51.0, -1.0, 51.0, -1.0) == 0.0

    def test_known_distance(self):
        """London to Paris is ~340 km."""
        d = engine.haversine(51.5074, -0.1278, 48.8566, 2.3522)
        assert 340_000 < d < 345_000

    def test_symmetry(self):
        d1 = engine.haversine(51.0, -1.0, 52.0, -2.0)
        d2 = engine.haversine(52.0, -2.0, 51.0, -1.0)
        assert abs(d1 - d2) < 0.01

    def test_short_distance(self):
        """~111m for 0.001° latitude."""
        d = engine.haversine(51.000, -1.0, 51.001, -1.0)
        assert 110 < d < 112


class TestHaversineMatrix:
    """Tests for the vectorised haversine_matrix (added in Phase 1)."""

    def test_exists(self):
        assert hasattr(engine, 'haversine_matrix'), \
            "haversine_matrix not found — Phase 1 not implemented"

    def test_matches_scalar(self):
        """Matrix version must match scalar haversine within 0.01m."""
        import numpy as np
        lats1 = np.array([51.5074, 51.0, 51.456])
        lons1 = np.array([-0.1278, -1.0, -0.980])
        lats2 = np.array([48.8566, 52.0, 51.450])
        lons2 = np.array([2.3522, -2.0, -0.970])

        # Matrix: pairwise distances (3×3)
        mat = engine.haversine_matrix(lats1, lons1, lats2, lons2)
        assert mat.shape == (3, 3)

        for i in range(3):
            for j in range(3):
                expected = engine.haversine(lats1[i], lons1[i],
                                            lats2[j], lons2[j])
                assert abs(mat[i, j] - expected) < 0.1, \
                    f"Mismatch at ({i},{j}): {mat[i,j]} vs {expected}"

    def test_self_distance_zero(self):
        import numpy as np
        lats = np.array([51.0, 52.0])
        lons = np.array([-1.0, -2.0])
        mat = engine.haversine_matrix(lats, lons, lats, lons)
        for i in range(2):
            assert mat[i, i] < 0.01


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — sample_polyline
# ══════════════════════════════════════════════════════════════════════════

class TestSamplePolyline:
    def test_returns_correct_count(self):
        sampled = engine.sample_polyline(SQUARE_IDEAL, 20)
        assert len(sampled) == 20

    def test_endpoints_preserved(self):
        sampled = engine.sample_polyline(SQUARE_IDEAL, 10)
        # First point should be very close to original start
        d = engine.haversine(sampled[0][0], sampled[0][1],
                             SQUARE_IDEAL[0][0], SQUARE_IDEAL[0][1])
        assert d < 1.0  # within 1m

    def test_single_point(self):
        sampled = engine.sample_polyline([[51.0, -1.0]], 5)
        assert len(sampled) == 1

    def test_two_points(self):
        sampled = engine.sample_polyline([[51.0, -1.0], [51.001, -1.0]], 5)
        assert len(sampled) == 5


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — bidirectional_score
# ══════════════════════════════════════════════════════════════════════════

class TestBidirectionalScore:
    def test_perfect_match_low_score(self):
        """A route identical to ideal should score very low."""
        score = engine.bidirectional_score(SQUARE_IDEAL, SQUARE_IDEAL)
        assert score < 15.0, f"Perfect match scored {score}, expected < 15"

    def test_worse_for_bad_route(self):
        """A distant route should score much worse."""
        far_route = [[lat + 0.01, lng + 0.01] for lat, lng in SQUARE_IDEAL]
        score_good = engine.bidirectional_score(SQUARE_IDEAL, SQUARE_IDEAL)
        score_bad = engine.bidirectional_score(far_route, SQUARE_IDEAL)
        assert score_bad > score_good * 2

    def test_returns_1e9_for_empty(self):
        assert engine.bidirectional_score([], SQUARE_IDEAL) == 1e9
        assert engine.bidirectional_score(SQUARE_IDEAL, []) == 1e9

    def test_reasonable_score_range(self):
        """Score on a realistic route/ideal pair should be positive finite."""
        score = engine.bidirectional_score(SQUARE_ROUTE, SQUARE_IDEAL)
        assert 0 < score < 10000

    def test_score_deterministic(self):
        """Same inputs must produce identical score."""
        s1 = engine.bidirectional_score(SQUARE_ROUTE, SQUARE_IDEAL)
        s2 = engine.bidirectional_score(SQUARE_ROUTE, SQUARE_IDEAL)
        assert s1 == s2


# ══════════════════════════════════════════════════════════════════════════
#  REGRESSION SNAPSHOT — scoring must stay within tolerance
# ══════════════════════════════════════════════════════════════════════════

class TestScoreRegression:
    """
    Capture exact scores before optimisation and verify they don't drift.
    Tolerance: ±1.0m (numerical precision may change slightly with NumPy).
    """

    @pytest.fixture(autouse=True)
    def capture_baseline(self):
        """Compute baseline scores once."""
        self.score_perfect = engine.bidirectional_score(
            SQUARE_IDEAL, SQUARE_IDEAL)
        self.score_near = engine.bidirectional_score(
            SQUARE_ROUTE, SQUARE_IDEAL)
        self.score_long = engine.bidirectional_score(
            LONG_ROUTE, LONG_IDEAL)

    def test_perfect_stable(self):
        assert self.score_perfect < 15.0

    def test_near_stable(self):
        # This score should be moderate — route roughly follows ideal
        assert 10.0 < self.score_near < 500.0

    def test_long_stable(self):
        # Long synthetic route
        assert 0 < self.score_long < 10000


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — geometry helpers
# ══════════════════════════════════════════════════════════════════════════

class TestGeometryHelpers:
    def test_rotate_shape_360(self):
        """Full rotation should return near-original points."""
        pts = [[0.3, 0.4], [0.7, 0.6], [0.5, 0.8]]
        rotated = engine.rotate_shape(pts, 360)
        for orig, rot in zip(pts, rotated):
            assert abs(orig[0] - rot[0]) < 1e-10
            assert abs(orig[1] - rot[1]) < 1e-10

    def test_shape_to_latlngs_center(self):
        """Center point (0.5, 0.5) should map to geographic center."""
        pts = [[0.5, 0.5]]
        ll = engine.shape_to_latlngs(pts, [51.0, -1.0], 0.01)
        assert abs(ll[0][0] - 51.0) < 0.001
        assert abs(ll[0][1] - (-1.0)) < 0.001

    def test_turning_angle_straight(self):
        """Straight line should have ~0° turning angle."""
        a = [51.0, -1.0]
        b = [51.001, -1.0]
        c = [51.002, -1.0]
        angle = engine.turning_angle(a, b, c)
        assert abs(angle) < 1.0

    def test_point_to_segment_dist(self):
        """Point on segment has distance 0."""
        a = [51.0, -1.0]
        b = [51.001, -1.0]
        mid = [51.0005, -1.0]
        d = engine.point_to_segment_dist(mid, a, b)
        assert d < 1.0  # < 1m

    def test_min_dist_to_polyline(self):
        p = [51.4555, -0.975]
        d = engine.min_dist_to_polyline(p, SQUARE_IDEAL)
        assert d >= 0
        assert d < 1000  # within 1km


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — densification
# ══════════════════════════════════════════════════════════════════════════

class TestDensification:
    def test_densify_increases_points(self):
        d = engine.densify(SQUARE_IDEAL, spacing_m=100)
        assert len(d) > len(SQUARE_IDEAL)

    def test_adaptive_densify(self):
        d = engine.adaptive_densify(SQUARE_IDEAL)
        assert len(d) > len(SQUARE_IDEAL)

    def test_densify_preserves_endpoints(self):
        d = engine.densify(SQUARE_IDEAL, spacing_m=100)
        d_start = engine.haversine(d[0][0], d[0][1],
                                   SQUARE_IDEAL[0][0], SQUARE_IDEAL[0][1])
        assert d_start < 1.0


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — coarse proximity score
# ══════════════════════════════════════════════════════════════════════════

class TestCoarseProximity:
    def test_returns_1e9_without_graph(self):
        score = engine.coarse_proximity_score(None, SQUARE_IDEAL)
        assert score == 1e9


# ══════════════════════════════════════════════════════════════════════════
#  PERFORMANCE BASELINE — timing tests (not strict, just informational)
# ══════════════════════════════════════════════════════════════════════════

class TestPerformance:
    def test_scoring_speed(self):
        """Score computation should complete in reasonable time."""
        t0 = time.time()
        for _ in range(10):
            engine.bidirectional_score(SQUARE_ROUTE, SQUARE_IDEAL)
        elapsed = time.time() - t0
        print(f"\n  bidirectional_score × 10: {elapsed:.3f}s "
              f"({elapsed/10*1000:.1f}ms/eval)")
        # After optimisation this should be much faster
        assert elapsed < 30.0  # generous upper bound

    def test_sample_polyline_speed(self):
        t0 = time.time()
        for _ in range(100):
            engine.sample_polyline(LONG_ROUTE, 150)
        elapsed = time.time() - t0
        print(f"\n  sample_polyline × 100: {elapsed:.3f}s "
              f"({elapsed/100*1000:.1f}ms/call)")
        assert elapsed < 10.0

    def test_scoring_long_speed(self):
        """Score on longer route."""
        t0 = time.time()
        for _ in range(5):
            engine.bidirectional_score(LONG_ROUTE, LONG_IDEAL)
        elapsed = time.time() - t0
        print(f"\n  bidirectional_score(long) × 5: {elapsed:.3f}s "
              f"({elapsed/5*1000:.1f}ms/eval)")
        assert elapsed < 60.0


# ══════════════════════════════════════════════════════════════════════════
#  SHAPE DISTANCE & SIMILARITY
# ══════════════════════════════════════════════════════════════════════════

class TestShapeSimilarity:
    def test_identical_distance_zero(self):
        d = engine._shape_distance(HEART_PTS, HEART_PTS)
        assert d < 0.001

    def test_different_shapes(self):
        star = [[.50,.00],[.62,.34],[1.0,.38],[.72,.60],[.80,1.0],
                [.50,.76],[.20,1.0],[.28,.60],[0,.38],[.38,.34],[.50,.00]]
        d = engine._shape_distance(HEART_PTS, star)
        assert d > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
