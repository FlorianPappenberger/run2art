"""Quick validation test for v8.0 engine on heart shape, Reading UK."""
import sys, time, math, json
sys.path.insert(0, '.')

from geometry import shape_to_latlngs, adaptive_densify
from routing import fetch_graph, build_kdtree, log
from core_router import CoreRouter
from scoring_v8 import score_v8, frechet_score
from scoring import bidirectional_score

# Heart shape (parametric)
N = 50
pts = []
for i in range(N):
    t = 2 * math.pi * i / N
    x = 16 * math.sin(t) ** 3
    y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
    pts.append([(x + 16) / 32, (y + 20) / 40])
pts.append(pts[0])  # close

center = [51.4543, -0.9781]  # Reading, UK
scale = 0.018
rotation = 0

print("=" * 60)
print("Run2Art v8.0 — Quick Validation: Heart @ Reading")
print("=" * 60)

# Fetch graph
t0 = time.time()
G = fetch_graph(center, dist=2500)
if not G:
    print("ERROR: Could not fetch graph")
    sys.exit(1)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges [{time.time()-t0:.1f}s]")

# Generate ideal line
wps = shape_to_latlngs(pts, center, scale, rotation)
ideal = adaptive_densify(wps, base_spacing=35, curve_spacing=15)
print(f"Ideal line: {len(ideal)} points")

# --- v8 CoreRouter ---
print("\n--- v8.0 CoreRouter ---")
t1 = time.time()
router = CoreRouter(G, ideal)
route_v8 = router.route()
t_route = time.time() - t1

if route_v8:
    s_v8 = score_v8(route_v8, ideal)
    s_v7 = bidirectional_score(route_v8, ideal)
    fd = frechet_score(route_v8, ideal)
    print(f"  Route: {len(route_v8)} points [{t_route:.1f}s]")
    print(f"  v8 score (Frechet-primary): {s_v8:.1f}m")
    print(f"  v7 score (legacy):          {s_v7:.1f}m")
    print(f"  Frechet distance:           {fd:.1f}m")
    
    # Compute route length
    from geometry import haversine
    length = sum(haversine(route_v8[i][0], route_v8[i][1],
                           route_v8[i+1][0], route_v8[i+1][1])
                 for i in range(len(route_v8)-1))
    print(f"  Route length:               {length:.0f}m")
else:
    print(f"  FAILED — no route found [{t_route:.1f}s]")

# --- v7 pipeline for comparison ---
print("\n--- v7.0 Pipeline (comparison) ---")
from engine import _fit_and_score_v7
t2 = time.time()
kd = build_kdtree(G)
s7, route_v7 = _fit_and_score_v7(G, pts, rotation, scale, center, kdtree_data=kd)
t_v7 = time.time() - t2

if route_v7:
    fd7 = frechet_score(route_v7, ideal)
    print(f"  Route: {len(route_v7)} points [{t_v7:.1f}s]")
    print(f"  v8 score (Frechet-primary): {score_v8(route_v7, ideal):.1f}m")
    print(f"  v7 score (legacy):          {s7:.1f}m")
    print(f"  Frechet distance:           {fd7:.1f}m")
    
    length7 = sum(haversine(route_v7[i][0], route_v7[i][1],
                            route_v7[i+1][0], route_v7[i+1][1])
                  for i in range(len(route_v7)-1))
    print(f"  Route length:               {length7:.0f}m")
else:
    print(f"  FAILED [{t_v7:.1f}s]")

# Save results
results = {}
if route_v8:
    results['v8'] = {
        'route': route_v8, 'score_v8': round(s_v8, 1),
        'frechet': round(fd, 1), 'points': len(route_v8),
        'length_m': round(length, 0), 'time_s': round(t_route, 1)
    }
if route_v7:
    results['v7'] = {
        'route': route_v7, 'score_v7': round(s7, 1),
        'frechet': round(fd7, 1), 'points': len(route_v7),
        'length_m': round(length7, 0), 'time_s': round(t_v7, 1)
    }

with open('public/v8_validation.json', 'w') as f:
    json.dump(results, f)
print(f"\nResults saved to public/v8_validation.json")

print("\n" + "=" * 60)
print("Validation complete")
print("=" * 60)
