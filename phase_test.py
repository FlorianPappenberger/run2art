"""
phase_test.py — Per-phase benchmark + visualization for run2art improvements.

Usage:
    python phase_test.py [phase_number] [phase_description]
    e.g.  python phase_test.py 0 "Baseline"
          python phase_test.py 1 "n_fine=40 + reweighted scoring"

Runs heart-only benchmark (fit + optimize), captures score breakdowns,
renders folium maps + matplotlib comparison PNGs, appends to cumulative table.
"""

import json, math, os, sys, time, webbrowser
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import fit_and_score, make_result, coarse_grid_search, make_offsets, HANDLERS
from geometry import shape_to_latlngs, densify, adaptive_densify, haversine, sample_polyline
from scoring import coarse_proximity_score, bidirectional_score
from scoring_v8 import score_v8, frechet_score, frechet_normalized
from routing import log, fetch_graph, build_kdtree

# ── Constants ──
CENTER = [51.4543, -0.9781]
HEART_PTS = [
    [.50,.14],[.66,-.04],[.90,-.06],[1.0,.18],[.94,.46],
    [.76,.70],[.50,1.0],[.24,.70],[.06,.46],[0,.18],
    [.10,-.06],[.34,-.04],[.50,.14]
]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "results")
CUMULATIVE_FILE = os.path.join(RESULTS_DIR, "cumulative_results.json")


def score_breakdown(route, ideal_pts):
    """Return dict with individual score components."""
    if not route or len(route) < 2:
        return {"total": 1e9, "frechet": 1e9, "coverage": 1e9,
                "perpendicular": 1e9, "heading": 1e9, "length_ratio": 1e9,
                "hausdorff": 1e9, "fourier": 1e9, "bidir_total": 1e9}

    from scoring_v8 import (frechet_score, _coverage_penalty,
                            _perpendicular_mean, _heading_fidelity,
                            _length_ratio_penalty, _hausdorff_distance,
                            fourier_descriptor_score, score_v8)

    fd = frechet_score(route, ideal_pts)
    haus = _hausdorff_distance(route, ideal_pts)
    cov = _coverage_penalty(route, ideal_pts)
    perp = _perpendicular_mean(route, ideal_pts)
    head = _heading_fidelity(route, ideal_pts)
    lr = _length_ratio_penalty(route, ideal_pts)
    fourier = fourier_descriptor_score(route, ideal_pts)
    total = score_v8(route, ideal_pts)  # uses current weights

    bidir = bidirectional_score(route, ideal_pts)

    return {
        "total": round(total, 1),
        "frechet": round(fd, 1),
        "coverage": round(cov, 1),
        "perpendicular": round(perp, 1),
        "heading": round(head, 1),
        "length_ratio": round(lr, 1),
        "hausdorff": round(haus, 1),
        "fourier": round(fourier, 1),
        "bidir_total": round(bidir, 1),
    }


def run_heart_benchmark(mode="optimize"):
    """Run heart benchmark and return result dict + all candidates."""
    log(f"[phase_test] Running heart {mode}...")

    payload = {
        "mode": mode,
        "shapes": [{"name": "Heart", "pts": HEART_PTS}],
        "shape_index": 0,
        "center_point": CENTER,
    }
    t0 = time.time()
    result = HANDLERS[mode](payload)
    elapsed = time.time() - t0
    result["time_seconds"] = round(elapsed, 1)
    return result, elapsed


def render_routes_to_html(routes, heart_template_coords, output_path, phase_name):
    """Render interactive folium map of all candidate routes."""
    try:
        import folium
    except ImportError:
        log("[phase_test] folium not installed, skipping HTML map")
        return

    if not routes:
        return

    centre_lat = sum(r['coords'][0][0] for r in routes if r['coords']) / max(len(routes), 1)
    centre_lon = sum(r['coords'][0][1] for r in routes if r['coords']) / max(len(routes), 1)
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=15,
                   tiles='CartoDB positron')

    # Draw ideal heart ghost outline
    folium.PolyLine(
        locations=heart_template_coords + [heart_template_coords[0]],
        color='red', weight=2, opacity=0.6, dash_array='8 4',
        tooltip='Ideal heart template'
    ).add_to(m)

    for route in routes:
        if not route.get('coords'):
            continue
        color = '#00cc44' if route.get('is_best') else '#aaaaaa'
        weight = 4 if route.get('is_best') else 1.5
        opacity = 1.0 if route.get('is_best') else 0.4
        folium.PolyLine(
            locations=route['coords'],
            color=color, weight=weight, opacity=opacity,
            tooltip=f"{route.get('label','')} | score={route.get('score', 0):.1f}"
        ).add_to(m)
        if route.get('is_best'):
            folium.Marker(
                location=route['coords'][0],
                icon=folium.Icon(color='green', icon='play'),
                tooltip='Start / Finish'
            ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;top:10px;right:10px;z-index:9999;background:white;
                padding:12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.3);
                font-family:monospace;font-size:12px;max-width:280px">
        <b>{phase_name}</b><br>
        <span style="color:#00cc44">━━</span> Best route &nbsp;
        <span style="color:#aaa">━━</span> Other candidates<br>
        <span style="color:red">╌╌</span> Ideal heart template<br><br>
        {"<br>".join(
            f"{'* ' if r.get('is_best') else '  '}{r.get('label','')}: <b>{r.get('score',0):.1f}</b>"
            for r in sorted(routes, key=lambda x: x.get('score', 1e9))
        )}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_path)
    log(f"  [MAP] Saved -> {output_path}")


def render_comparison_png(prev_route, curr_route, heart_template, G,
                          output_path, phase_name):
    """Side-by-side comparison PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log("[phase_test] matplotlib not installed, skipping PNG")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#1a1a2e')
    titles = [f"Before {phase_name}", f"After {phase_name}"]
    routes = [prev_route, curr_route]

    for ax, route, title in zip(axes, routes, titles):
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title, color='white', fontsize=13, pad=8)

        # Draw road network edges
        if G:
            for u, v, data in G.edges(data=True):
                xs = [G.nodes[u]['x'], G.nodes[v]['x']]
                ys = [G.nodes[u]['y'], G.nodes[v]['y']]
                ax.plot(xs, ys, color='#444466', linewidth=0.4, alpha=0.6)

        # Ideal heart outline
        hx = [p[1] for p in heart_template] + [heart_template[0][1]]
        hy = [p[0] for p in heart_template] + [heart_template[0][0]]
        ax.plot(hx, hy, color='#ff4466', linewidth=1.5, linestyle='--',
                alpha=0.7, label='Ideal heart')

        # Actual route
        if route and route.get('coords'):
            rx = [p[1] for p in route['coords']]
            ry = [p[0] for p in route['coords']]
            ax.plot(rx, ry, color='#00ff88', linewidth=2.5,
                    label=f"Score: {route.get('score', 0):.1f}")
            ax.plot(rx[0], ry[0], 'o', color='#ffdd00', markersize=8)
        else:
            ax.text(0.5, 0.5, 'No route', color='white', fontsize=16,
                    ha='center', va='center', transform=ax.transAxes)

        ax.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=9)
        ax.tick_params(colors='#666688')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    plt.suptitle(f"run2art - Heart Route Quality: {phase_name}",
                 color='white', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    log(f"  [PNG] Saved -> {output_path}")


def load_cumulative():
    """Load cumulative results from disk."""
    if os.path.exists(CUMULATIVE_FILE):
        with open(CUMULATIVE_FILE, 'r') as f:
            return json.load(f)
    return {"phases": []}


def save_cumulative(data):
    with open(CUMULATIVE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def print_cumulative_table(phases):
    """Print markdown-style cumulative comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Phase':<8} {'Description':<35} {'Total':>8} {'Frechet':>8} {'Haus':>8} "
          f"{'Cover':>8} {'Turn':>8} {'Delta':>8} {'Time':>8}")
    print("-" * 100)
    baseline_score = phases[0]["breakdown"]["total"] if phases else 0
    for p in phases:
        bd = p["breakdown"]
        delta = bd["total"] - baseline_score
        delta_str = f"{delta:+.1f}" if p["phase"] > 0 else "---"
        print(f"P{p['phase']:<7} {p['description']:<35} {bd['total']:>8.1f} "
              f"{bd['frechet']:>8.1f} {bd['hausdorff']:>8.1f} "
              f"{bd['coverage']:>8.1f} {bd['heading']:>8.1f} "
              f"{delta_str:>8} {p['time']:>7.1f}s")
    print("=" * 100 + "\n")


def generate_dashboard(phases):
    """Generate master dashboard HTML."""
    table_rows = ""
    baseline = phases[0]["breakdown"]["total"] if phases else 0
    for p in phases:
        bd = p["breakdown"]
        delta = bd["total"] - baseline
        delta_class = "improvement" if delta < 0 else ("regression" if delta > 0 else "")
        delta_str = f"{delta:+.1f}" if p["phase"] > 0 else "---"
        table_rows += f"""      <tr>
        <td>P{p['phase']}</td><td>{p['description']}</td>
        <td>{bd['total']:.1f}</td><td>{bd['hausdorff']:.1f}</td>
        <td>{bd['coverage']:.1f}</td><td>{bd['heading']:.1f}</td>
        <td class="{delta_class}">{delta_str}</td><td>{p['time']:.1f}s</td>
      </tr>\n"""

    phase_sections = ""
    for p in phases:
        png_name = f"phase_{p['phase']}_comparison.png"
        map_name = f"phase_{p['phase']}_routes.html"
        phase_sections += f"""
    <h3>Phase {p['phase']}: {p['description']}</h3>
    <img class="phase-img" src="{png_name}" alt="Phase {p['phase']}">
    <a class="map-link" href="{map_name}">Interactive Map</a>
"""

    labels = json.dumps([f"P{p['phase']}" for p in phases])
    scores = json.dumps([p["breakdown"]["total"] for p in phases])

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>run2art - Heart Route Improvement Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ background: #1a1a2e; color: #eee; font-family: monospace; padding: 24px; }}
    h1 {{ color: #00ff88; }}
    h2 {{ color: #aaaaff; border-bottom: 1px solid #333; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
    th {{ background: #2a2a4e; color: #00ff88; padding: 8px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #2a2a3e; }}
    .improvement {{ color: #00ff88; font-weight: bold; }}
    .regression  {{ color: #ff4466; font-weight: bold; }}
    .phase-img   {{ width: 100%; border-radius: 8px; margin-bottom: 16px;
                   border: 1px solid #333; }}
    .map-link    {{ display: inline-block; margin: 6px 8px 6px 0;
                   padding: 6px 14px; background: #2a2a4e;
                   color: #00aaff; border-radius: 4px; text-decoration: none; }}
    .map-link:hover {{ background: #3a3a6e; }}
    canvas       {{ max-height: 300px; margin-bottom: 40px; }}
  </style>
</head>
<body>
  <h1>run2art - Heart Route Improvement Dashboard</h1>

  <h2>Score Over Time</h2>
  <canvas id="scoreChart"></canvas>

  <h2>Phase Results</h2>
  <table id="resultsTable">
    <thead>
      <tr>
        <th>Phase</th><th>Description</th><th>Total Score</th>
        <th>Hausdorff</th><th>Coverage</th><th>Turning</th>
        <th>Delta</th><th>Time</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>

  <h2>Visual Timeline</h2>
  {phase_sections}

  <script>
    new Chart(document.getElementById('scoreChart'), {{
      type: 'line',
      data: {{
        labels: {labels},
        datasets: [{{
          label: 'Heart Shape Score (lower = better)',
          data: {scores},
          borderColor: '#00ff88',
          backgroundColor: 'rgba(0,255,136,0.1)',
          tension: 0.3,
          pointRadius: 6,
          pointBackgroundColor: '#00ff88'
        }}]
      }},
      options: {{
        plugins: {{
          legend: {{ labels: {{ color: '#eee' }} }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#2a2a3e' }} }},
          y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#2a2a3e' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""
    path = os.path.join(RESULTS_DIR, "dashboard.html")
    with open(path, 'w') as f:
        f.write(html)
    log(f"[dashboard] Saved -> {path}")
    return path


def run_phase(phase_num, description, extra_routes=None):
    """Main entry: run heart benchmark, score, visualize, record."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cumulative = load_cumulative()

    # Run fit + optimize
    fit_result, fit_time = run_heart_benchmark("fit")
    opt_result, opt_time = run_heart_benchmark("optimize")

    # Use the better of fit/optimize
    best_result = opt_result if opt_result.get("score", 1e9) < fit_result.get("score", 1e9) else fit_result
    best_time = opt_time if opt_result.get("score", 1e9) < fit_result.get("score", 1e9) else fit_time
    route = best_result.get("route")

    # Score breakdown
    if route:
        wps = shape_to_latlngs(HEART_PTS, best_result.get("center", CENTER),
                               best_result.get("scale", 0.012),
                               best_result.get("rotation", 0))
        ideal = adaptive_densify(wps, base_spacing=35, curve_spacing=15)
        bd = score_breakdown(route, ideal)
    else:
        bd = score_breakdown(None, None)

    # Build phase record
    phase_record = {
        "phase": phase_num,
        "description": description,
        "breakdown": bd,
        "time": round(best_time, 1),
        "best_result": {
            "score": best_result.get("score"),
            "rotation": best_result.get("rotation"),
            "scale": best_result.get("scale"),
            "center": best_result.get("center"),
            "route_points": len(route) if route else 0,
        },
        "fit_score": fit_result.get("score"),
        "opt_score": opt_result.get("score"),
        "fit_time": round(fit_time, 1),
        "opt_time": round(opt_time, 1),
    }

    # Build route objects for visualization
    all_routes = []
    for tag, res, is_best in [("fit", fit_result, False), ("optimize", opt_result, True)]:
        r = res.get("route")
        if r:
            all_routes.append({
                "coords": r,
                "score": res.get("score", 1e9),
                "label": f"{tag} rot={res.get('rotation',0):.0f} sc={res.get('scale',0):.4f}",
                "is_best": is_best and res is best_result,
            })
    if extra_routes:
        all_routes.extend(extra_routes)

    # Ideal heart template coords for visualization
    wps_vis = shape_to_latlngs(HEART_PTS, best_result.get("center", CENTER),
                                best_result.get("scale", 0.012),
                                best_result.get("rotation", 0))
    heart_vis = [[p[0], p[1]] for p in wps_vis]

    # Folium map
    map_path = os.path.join(RESULTS_DIR, f"phase_{phase_num}_routes.html")
    render_routes_to_html(all_routes, heart_vis, map_path,
                          f"Phase {phase_num}: {description}")

    # Comparison PNG
    prev_phases = cumulative.get("phases", [])
    prev_route = None
    if prev_phases:
        prev_best = prev_phases[-1].get("best_result", {})
        # We'll just show the current route vs itself if no previous
        prev_route = {"coords": None, "score": prev_phases[-1]["breakdown"]["total"]}

    curr_route_obj = {"coords": route, "score": bd["total"]}

    # Try to render PNG with graph
    try:
        G = fetch_graph(CENTER, dist=2500)
    except Exception:
        G = None

    png_path = os.path.join(RESULTS_DIR, f"phase_{phase_num}_comparison.png")
    render_comparison_png(
        prev_route or {"coords": None, "score": 0},
        curr_route_obj, heart_vis, G, png_path,
        f"Phase {phase_num}: {description}"
    )

    # Append to cumulative
    # Remove existing entry for this phase number
    cumulative["phases"] = [p for p in cumulative.get("phases", [])
                           if p["phase"] != phase_num]
    cumulative["phases"].append(phase_record)
    cumulative["phases"].sort(key=lambda x: x["phase"])
    save_cumulative(cumulative)

    # Print table
    print_cumulative_table(cumulative["phases"])

    # Generate dashboard
    dashboard_path = generate_dashboard(cumulative["phases"])

    return phase_record


if __name__ == "__main__":
    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    desc = sys.argv[2] if len(sys.argv) > 2 else "Baseline"
    run_phase(phase, desc)
