"""
all_methods_map.py — Generate a single interactive HTML map comparing ALL engine
versions & methods tried so far (v5.1 → v8.2).

Loads pre-existing routes from full_comparison.json (v5.1–v8.1-Abstract),
runs the current engine (v8.2) in fit mode to get the latest route,
and renders everything on a Leaflet map with toggleable layers.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Constants ──
CENTER = [51.4543, -0.9781]
HEART_PTS = [
    [.50, .14], [.66, -.04], [.90, -.06], [1.0, .18], [.94, .46],
    [.76, .70], [.50, 1.0], [.24, .70], [.06, .46], [0, .18],
    [.10, -.06], [.34, -.04], [.50, .14]
]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "results")
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "all_methods_comparison.html")


def parametric_heart(n=80):
    """Generate a smooth parametric heart for the ideal overlay."""
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x = 16.0 * math.sin(t) ** 3
        y = (13.0 * math.cos(t) - 5.0 * math.cos(2 * t)
             - 2.0 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((x, y))
    xs, ys = zip(*pts)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    norm = [[round((x - xmin) / (xmax - xmin), 4),
             round(1.0 - (y - ymin) / (ymax - ymin), 4)]
            for x, y in pts]
    norm.append(norm[0])
    return norm


def shape_to_latlngs(pts, center, scale_deg, rotation_deg=0):
    """Convert normalised [0-1] shape to [lat, lng] coords."""
    if rotation_deg:
        rad = math.radians(rotation_deg)
        cx, cy = 0.5, 0.5
        pts = [[(x - cx) * math.cos(rad) - (y - cy) * math.sin(rad) + cx,
                (x - cx) * math.sin(rad) + (y - cy) * math.cos(rad) + cy]
               for x, y in pts]
    cx, cy = 0.5, 0.5
    return [[center[0] - (y - cy) * scale_deg,
             center[1] + (x - cx) * scale_deg * 1.4] for x, y in pts]


def load_existing_routes():
    """Load v5.1–v8.1 routes from full_comparison.json."""
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "public", "full_comparison.json")
    if not os.path.exists(fpath):
        print(f"[WARN] {fpath} not found — only v8.2 will be shown")
        return {}
    with open(fpath, "r") as f:
        data = json.load(f)
    return data.get("routes", {})


def run_v82_fit():
    """Run the current engine (v8.2 with all improvements) in fit mode."""
    from engine import HANDLERS
    payload = {
        "mode": "fit",
        "shapes": [{"name": "Heart", "pts": HEART_PTS}],
        "shape_index": 0,
        "center_point": CENTER,
    }
    print("[v8.2] Running fit mode...")
    t0 = time.time()
    result = HANDLERS["fit"](payload)
    elapsed = time.time() - t0
    print(f"[v8.2] Fit done in {elapsed:.1f}s — score={result.get('score', '?')}")
    return {
        "version": "v8.2",
        "mode": "fit",
        "score": result.get("score"),
        "route_length_m": result.get("route_length_m"),
        "time_seconds": round(elapsed, 1),
        "rotation": result.get("rotation"),
        "scale": result.get("scale"),
        "route": result.get("route", []),
    }


def run_v82_optimize():
    """Run the current engine (v8.2) in optimize mode."""
    from engine import HANDLERS
    payload = {
        "mode": "optimize",
        "shapes": [{"name": "Heart", "pts": HEART_PTS}],
        "shape_index": 0,
        "center_point": CENTER,
    }
    print("[v8.2] Running optimize mode...")
    t0 = time.time()
    result = HANDLERS["optimize"](payload)
    elapsed = time.time() - t0
    print(f"[v8.2] Optimize done in {elapsed:.1f}s — score={result.get('score', '?')}")
    return {
        "version": "v8.2",
        "mode": "optimize",
        "score": result.get("score"),
        "route_length_m": result.get("route_length_m"),
        "time_seconds": round(elapsed, 1),
        "rotation": result.get("rotation"),
        "scale": result.get("scale"),
        "route": result.get("route", []),
    }


def build_html(all_routes, ideal_coords):
    """Build a standalone Leaflet HTML with all routes."""

    # Route config: color, dash pattern, weight, friendly label
    config = {
        "v5.1_fit":               {"color": "#e74c3c", "dash": "",        "weight": 2, "label": "v5.1 Fit"},
        "v5.1_optimize":          {"color": "#c0392b", "dash": "10, 6",   "weight": 2, "label": "v5.1 Optimize"},
        "v6.0_fit":               {"color": "#e67e22", "dash": "",        "weight": 2, "label": "v6.0 Fit"},
        "v6.0_optimize":          {"color": "#d35400", "dash": "10, 6",   "weight": 2, "label": "v6.0 Optimize"},
        "v7.0_fit":               {"color": "#f1c40f", "dash": "",        "weight": 2, "label": "v7.0 Fit"},
        "v7.0_optimize":          {"color": "#d4ac0d", "dash": "10, 6",   "weight": 2, "label": "v7.0 Optimize"},
        "v8.0_fit":               {"color": "#3498db", "dash": "",        "weight": 3, "label": "v8.0 Fit"},
        "v8.0_optimize":          {"color": "#2471a3", "dash": "10, 6",   "weight": 3, "label": "v8.0 Optimize"},
        "v8.1_fit":               {"color": "#2ecc71", "dash": "",        "weight": 3, "label": "v8.1 Fit (Flow-Aware)"},
        "v8.1_optimize":          {"color": "#27ae60", "dash": "10, 6",   "weight": 3, "label": "v8.1 Optimize (Flow-Aware)"},
        "v8.1_abstract_fit":      {"color": "#9b59b6", "dash": "",        "weight": 3, "label": "v8.1-Abstract Fit"},
        "v8.1_abstract_optimize": {"color": "#8e44ad", "dash": "10, 6",   "weight": 3, "label": "v8.1-Abstract Optimize"},
        "v8.2_fit":               {"color": "#00e5ff", "dash": "",        "weight": 5, "label": "v8.2 Fit (All Improvements)"},
        "v8.2_optimize":          {"color": "#00b8d4", "dash": "10, 6",   "weight": 5, "label": "v8.2 Optimize (All Improvements)"},
    }

    routes_json = json.dumps(all_routes)
    config_json = json.dumps(config)
    ideal_json = json.dumps(ideal_coords)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPS Art — All Methods Comparison Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  #map {{ height: 100vh; width: 100%; }}
  .panel {{
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: rgba(255,255,255,0.96); padding: 16px 14px;
    border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    max-width: 320px; max-height: 90vh; overflow-y: auto;
    font-size: 13px;
  }}
  .panel h3 {{ margin: 0 0 6px 0; font-size: 15px; }}
  .panel .sub {{ color: #888; font-size: 11px; margin-bottom: 10px; }}
  .section {{ margin: 10px 0 4px 0; font-weight: 700; font-size: 11px;
              color: #555; border-top: 1px solid #eee; padding-top: 6px;
              text-transform: uppercase; letter-spacing: 0.5px; }}
  .row {{
    display: flex; align-items: center; padding: 2px 0;
    cursor: pointer; user-select: none;
  }}
  .row:hover {{ background: #f5f5f5; border-radius: 3px; }}
  .row input {{ margin-right: 6px; }}
  .swatch {{
    display: inline-block; width: 28px; height: 0;
    margin-right: 8px; flex-shrink: 0;
  }}
  .lbl {{ flex: 1; }}
  .sc {{ font-size: 11px; color: #666; font-weight: 700; margin-left: 4px; }}
  #btnAll, #btnNone, #btnLatest {{
    margin: 2px 4px 0 0; padding: 3px 10px; font-size: 11px;
    border: 1px solid #ccc; border-radius: 4px; cursor: pointer;
    background: #fafafa;
  }}
  #btnAll:hover, #btnNone:hover, #btnLatest:hover {{ background: #eee; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel" id="panel">
  <h3>All Methods Comparison</h3>
  <div class="sub">Heart Shape &bull; Reading, UK &bull; {len(all_routes)} routes</div>
  <div>
    <button id="btnAll">Show All</button>
    <button id="btnNone">Hide All</button>
    <button id="btnLatest">Latest Only</button>
  </div>
  <div id="legend"></div>
</div>

<script>
const map = L.map('map').setView([51.4543, -0.9781], 14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap &copy; CARTO'
}}).addTo(map);

const routes = {routes_json};
const config = {config_json};
const ideal = {ideal_json};

// Draw ideal heart outline
L.polyline(ideal, {{
  color: '#ff0066', weight: 2, opacity: 0.5, dashArray: '6,4',
  interactive: false
}}).addTo(map).bindTooltip('Ideal heart template');

const layers = {{}};
const allBounds = [];

// Group routes by generation
const groups = {{
  'Generation 1 (Legacy)': ['v5.1_fit','v5.1_optimize','v6.0_fit','v6.0_optimize'],
  'Generation 2 (Bidirectional)': ['v7.0_fit','v7.0_optimize'],
  'Generation 3 (CoreRouter)': ['v8.0_fit','v8.0_optimize','v8.1_fit','v8.1_optimize',
                                  'v8.1_abstract_fit','v8.1_abstract_optimize'],
  'Generation 4 (v8.2 — All Improvements)': ['v8.2_fit','v8.2_optimize'],
}};

// Plot all routes
Object.keys(routes).forEach(key => {{
  const data = routes[key];
  const cfg = config[key];
  if (!cfg || !data.route || data.route.length === 0) return;
  const latlngs = data.route.map(pt => [pt[0], pt[1]]);
  const poly = L.polyline(latlngs, {{
    color: cfg.color, weight: cfg.weight, opacity: 0.85,
    dashArray: cfg.dash || null, smoothFactor: 1
  }});
  poly.bindPopup(
    '<b>' + cfg.label + '</b><br>' +
    'Score: ' + (data.score != null ? data.score.toFixed(1) + 'm' : '?') + '<br>' +
    'Route: ' + (data.route_length_m ? data.route_length_m.toFixed(0) + 'm' : 'N/A') + '<br>' +
    'Time: ' + (data.time_seconds != null ? data.time_seconds.toFixed(1) + 's' : '?') + '<br>' +
    'Rotation: ' + (data.rotation != null ? data.rotation + '&deg;' : 'N/A') + '<br>' +
    'Scale: ' + (data.scale != null ? data.scale.toFixed(4) : 'N/A') + '<br>' +
    'Points: ' + data.route.length
  );
  poly.addTo(map);
  layers[key] = poly;
  allBounds.push(...latlngs);
}});

if (allBounds.length > 0) map.fitBounds(L.latLngBounds(allBounds), {{ padding: [50, 50] }});

// Build legend
const legend = document.getElementById('legend');
Object.entries(groups).forEach(([group, keys]) => {{
  const sec = document.createElement('div');
  sec.className = 'section';
  sec.textContent = group;
  legend.appendChild(sec);
  keys.forEach(key => {{
    const cfg = config[key];
    const data = routes[key];
    if (!cfg) return;
    const row = document.createElement('label');
    row.className = 'row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!data;
    cb.dataset.key = key;
    cb.addEventListener('change', () => {{
      if (layers[key]) {{
        if (cb.checked) map.addLayer(layers[key]);
        else map.removeLayer(layers[key]);
      }}
    }});
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.borderTop = cfg.weight + 'px ' + (cfg.dash ? 'dashed' : 'solid') + ' ' + cfg.color;
    const lbl = document.createElement('span');
    lbl.className = 'lbl';
    lbl.textContent = cfg.label;
    const sc = document.createElement('span');
    sc.className = 'sc';
    sc.textContent = data ? data.score.toFixed(1) + 'm' : '—';
    if (!data) {{ cb.disabled = true; row.style.opacity = '0.4'; }}
    row.appendChild(cb);
    row.appendChild(swatch);
    row.appendChild(lbl);
    row.appendChild(sc);
    legend.appendChild(row);
  }});
}});

// Button controls
document.getElementById('btnAll').addEventListener('click', () => {{
  document.querySelectorAll('.row input').forEach(cb => {{
    cb.checked = true;
    if (layers[cb.dataset.key]) map.addLayer(layers[cb.dataset.key]);
  }});
}});
document.getElementById('btnNone').addEventListener('click', () => {{
  document.querySelectorAll('.row input').forEach(cb => {{
    cb.checked = false;
    if (layers[cb.dataset.key]) map.removeLayer(layers[cb.dataset.key]);
  }});
}});
document.getElementById('btnLatest').addEventListener('click', () => {{
  document.querySelectorAll('.row input').forEach(cb => {{
    const isLatest = cb.dataset.key.startsWith('v8.2');
    cb.checked = isLatest;
    if (layers[cb.dataset.key]) {{
      if (isLatest) map.addLayer(layers[cb.dataset.key]);
      else map.removeLayer(layers[cb.dataset.key]);
    }}
  }});
}});
</script>
</body>
</html>"""
    return html


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load existing routes (v5.1–v8.1)
    all_routes = load_existing_routes()
    print(f"Loaded {len(all_routes)} existing routes from full_comparison.json")

    # 2. Run v8.2 (current engine with all improvements)
    v82_fit = run_v82_fit()
    all_routes["v8.2_fit"] = v82_fit

    try:
        v82_opt = run_v82_optimize()
        all_routes["v8.2_optimize"] = v82_opt
    except Exception as e:
        print(f"[WARN] v8.2 optimize failed: {e}")

    # 3. Build ideal heart overlay at best v8.2 params
    best_v82 = v82_fit
    best_rot = best_v82.get("rotation", 0) or 0
    best_scale = best_v82.get("scale", 0.01) or 0.01
    ideal_pts = parametric_heart(80)
    ideal_coords = shape_to_latlngs(ideal_pts, CENTER, best_scale, best_rot)

    # 4. Generate HTML
    html = build_html(all_routes, ideal_coords)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"Map saved to: {OUTPUT_HTML}")
    print(f"{'='*60}")
    print(f"\nRoutes on map:")
    for key in sorted(all_routes.keys()):
        r = all_routes[key]
        pts = len(r.get("route", []))
        sc = r.get("score", "?")
        print(f"  {key:30s}  score={sc:<8}  points={pts}")


if __name__ == "__main__":
    main()
