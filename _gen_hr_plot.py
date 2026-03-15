"""Generate an HTML page showing all routes sorted by Heart Recognizability score.

Recalculates HR using the latest scorer (fixed symmetry) and embeds a manual
rating widget (1-10 stars) per route that persists to localStorage and can be
exported/imported as JSON.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v85_wide_search import route_heart_recognizability

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "results")

data = json.load(open(os.path.join(RESULTS_DIR, "v83_full_results.json")))
results = data["results"]

# Recalculate HR scores with fixed symmetry scorer
print("Recalculating HR scores with fixed symmetry...")
for r in results:
    route = r.get("route")
    if route and len(route) >= 4:
        hr, explain = route_heart_recognizability(route, n_sample=100)
        r["heart_recognizability"] = round(hr, 1)
        r["hr_explanation"] = explain

# Also update the full results file
with open(os.path.join(RESULTS_DIR, "v83_full_results.json"), "w") as f:
    json.dump(data, f)
print("Updated v83_full_results.json with recalculated HR scores.")

# Filter to routes with HR and valid geometry
results_with_hr = [r for r in results if r.get("heart_recognizability") is not None and r.get("route") and len(r.get("route", [])) >= 2]
results_with_hr.sort(key=lambda r: -r["heart_recognizability"])

# Build routes JSON
routes_json = []
for r in results_with_hr:
    routes_json.append({
        "id": r["id"],
        "label": r.get("label", r["id"]),
        "route": r["route"],
        "score": r.get("score"),
        "hr": r.get("heart_recognizability"),
        "hr_explain": r.get("hr_explanation", ""),
        "mode": r.get("mode"),
        "time_seconds": r.get("time_seconds"),
        "route_points": r.get("route_points", len(r.get("route", []))),
        "route_length_m": r.get("route_length_m"),
    })

routes_data = json.dumps(routes_json)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Heart Routes — HR Ranking + Manual Ratings</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }}
  .header {{
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white; padding: 20px 30px; text-align: center;
  }}
  .header h1 {{ font-size: 24px; margin-bottom: 4px; }}
  .header .sub {{ font-size: 13px; opacity: 0.85; }}
  .controls {{
    background: white; padding: 12px 30px; border-bottom: 1px solid #ddd;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }}
  .controls label {{ font-size: 12px; color: #555; }}
  .controls select, .controls button {{
    font-size: 12px; padding: 4px 10px; border: 1px solid #ccc;
    border-radius: 4px; background: white; cursor: pointer;
  }}
  .controls button:hover {{ background: #eee; }}
  .controls .btn-export {{ background: #2196F3; color: white; border-color: #1976D2; }}
  .controls .btn-export:hover {{ background: #1976D2; }}
  .controls .btn-import {{ background: #4CAF50; color: white; border-color: #388E3C; }}
  .controls .btn-import:hover {{ background: #388E3C; }}
  .controls .rating-count {{ font-size: 11px; color: #888; margin-left: auto; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 16px; padding: 20px 30px; max-width: 1600px; margin: 0 auto;
  }}
  .card {{
    background: white; border-radius: 8px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
  .card.rated {{ border: 2px solid #FFD700; }}
  .card-map {{ height: 260px; background: #e8e8e8; }}
  .card-info {{ padding: 12px 16px; }}
  .card-title {{ font-size: 14px; font-weight: 700; margin-bottom: 6px; display: flex; justify-content: space-between; }}
  .card-title .rank {{ color: white; background: #e74c3c; border-radius: 12px; padding: 1px 8px; font-size: 11px; }}
  .hr-bar-container {{
    background: #f0f0f0; border-radius: 4px; height: 22px; margin: 6px 0;
    position: relative; overflow: hidden;
  }}
  .hr-bar {{
    height: 100%; border-radius: 4px; transition: width 0.5s;
    display: flex; align-items: center; padding-left: 8px;
    font-size: 11px; font-weight: 700; color: white;
  }}
  /* Manual rating stars */
  .manual-rating {{
    margin: 8px 0; display: flex; align-items: center; gap: 6px;
    border: 1px dashed #ddd; border-radius: 6px; padding: 6px 10px;
    background: #fafafa;
  }}
  .manual-rating .label {{ font-size: 11px; color: #666; white-space: nowrap; }}
  .stars {{ display: flex; gap: 1px; }}
  .star {{
    cursor: pointer; font-size: 20px; color: #ddd;
    transition: color 0.1s; user-select: none; line-height: 1;
  }}
  .star:hover, .star.hover {{ color: #FFC107; }}
  .star.active {{ color: #FFD700; }}
  .rating-value {{ font-size: 13px; font-weight: 700; color: #333; min-width: 28px; text-align: center; }}
  .clear-rating {{
    font-size: 10px; color: #999; cursor: pointer; margin-left: 4px;
    text-decoration: underline;
  }}
  .clear-rating:hover {{ color: #e74c3c; }}
  .hr-breakdown {{
    display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap;
  }}
  .hr-pill {{
    font-size: 10px; padding: 2px 6px; border-radius: 3px;
    background: #f0f0f0; color: #555;
  }}
  .hr-pill.good {{ background: #e8f5e9; color: #2e7d32; }}
  .hr-pill.medium {{ background: #fff3e0; color: #e65100; }}
  .hr-pill.bad {{ background: #ffebee; color: #c62828; }}
  .meta {{ font-size: 11px; color: #888; margin-top: 6px; }}
  .meta span {{ margin-right: 12px; }}
  .big-map-section {{ padding: 20px 30px; max-width: 1600px; margin: 0 auto; }}
  .big-map-section h2 {{ font-size: 18px; margin-bottom: 10px; }}
  #bigMap {{ height: 500px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  #fileInput {{ display: none; }}
</style>
</head>
<body>

<div class="header">
  <h1>Heart Routes &mdash; HR Ranking + Manual Ratings</h1>
  <div class="sub">{len(results_with_hr)} routes &bull; Reading, UK &bull; Click stars to rate how much each looks like a heart</div>
</div>

<div class="controls">
  <label>Sort by:</label>
  <select id="sortBy" onchange="reSort()">
    <option value="hr" selected>Auto HR Score (highest)</option>
    <option value="manual">Your Rating (highest)</option>
    <option value="score">Routing Score (lowest)</option>
    <option value="combined">Combined (HR/score)</option>
    <option value="diff">Biggest disagreement (|yours - auto|)</option>
  </select>
  <label>Filter:</label>
  <select id="filterBy" onchange="reSort()">
    <option value="all" selected>All</option>
    <option value="rated">Rated by you</option>
    <option value="unrated">Not yet rated</option>
    <option value="v85">v8.5 Wide Search</option>
    <option value="v84">v8.4 Perceptual</option>
    <option value="v83">v8.3 Indent/Sym</option>
    <option value="v82">v8.2 Baseline</option>
  </select>
  <button class="btn-export" onclick="exportRatings()">Export Ratings (JSON)</button>
  <button class="btn-import" onclick="document.getElementById('fileInput').click()">Import Ratings</button>
  <input type="file" id="fileInput" accept=".json" onchange="importRatings(event)">
  <span class="rating-count" id="ratingCount">0 / {len(results_with_hr)} rated</span>
</div>

<div class="big-map-section">
  <h2>All Routes Overlay</h2>
  <div id="bigMap"></div>
</div>

<div class="grid" id="cardGrid"></div>

<script>
const allRoutes = {routes_data};
const STORAGE_KEY = 'gpsart_heart_manual_ratings';

// In-memory ratings store (primary) + localStorage (backup)
// This fixes file:// and VS Code Simple Browser where localStorage is unreliable
let _ratings = {{}};
try {{ _ratings = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{}}; }} catch(e) {{}}

function loadRatings() {{ return _ratings; }}
function saveRatings(ratings) {{
  _ratings = ratings;
  try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(ratings)); }} catch(e) {{}}
  updateRatingCount();
}}
function getRating(id) {{ return _ratings[id] || null; }}
function setRating(id, val) {{
  if (val === null) delete _ratings[id]; else _ratings[id] = val;
  saveRatings(_ratings);
}}
function updateRatingCount() {{
  const n = Object.keys(_ratings).length;
  const el = document.getElementById('ratingCount');
  if (el) el.textContent = n + ' / {len(results_with_hr)} rated';
}}

// Export ratings as JSON download
function exportRatings() {{
  const r = _ratings;
  const exportData = {{
    exported: new Date().toISOString(),
    ratings: r,
    summary: Object.entries(r).sort((a,b) => b[1]-a[1]).map(([id,v]) => ({{id, rating: v}}))
  }};
  const blob = new Blob([JSON.stringify(exportData, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'heart_manual_ratings.json';
  a.click(); URL.revokeObjectURL(url);
}}

// Import ratings from JSON file
function importRatings(event) {{
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    try {{
      const data = JSON.parse(e.target.result);
      const ratings = data.ratings || data;
      if (typeof ratings !== 'object') throw new Error('Invalid format');
      const existing = loadRatings();
      Object.assign(existing, ratings);
      saveRatings(existing);
      reSort();
      alert('Imported ' + Object.keys(ratings).length + ' ratings!');
    }} catch(err) {{
      alert('Failed to import: ' + err.message);
    }}
  }};
  reader.readAsText(file);
  event.target.value = '';
}}

// Color scale
function hrColor(hr) {{
  const t = Math.max(0, Math.min(1, (hr - 3) / 5));
  const r = Math.round(255 * (1 - t));
  const g = Math.round(180 * t);
  return `rgb(${{r}},${{g}},60)`;
}}

function pillClass(val) {{
  if (val >= 0.7) return 'good';
  if (val >= 0.4) return 'medium';
  return 'bad';
}}

function parseBreakdown(explain) {{
  if (!explain) return {{}};
  const parts = {{}};
  const re = /(\\w+)=([\\d.]+)/g;
  let m;
  while ((m = re.exec(explain)) !== null) parts[m[1]] = parseFloat(m[2]);
  return parts;
}}

// Component display order and labels for pills
const PILL_KEYS = ['lobes','indent','cusp','sym','dev','smooth','closure'];
const PILL_LABELS = {{lobes:'Lobes',indent:'Indent',cusp:'Cusp',sym:'Symmetry',dev:'Template',smooth:'Smooth',closure:'Closed'}};

// Big overview map
const bigMap = L.map('bigMap').setView([51.4543, -0.9781], 13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{maxZoom: 19}}).addTo(bigMap);

const allBounds = [];
allRoutes.forEach(r => {{
  if (!r.route || r.route.length < 2) return;
  const ll = r.route.map(p => [p[0], p[1]]);
  allBounds.push(...ll);
  const color = hrColor(r.hr || 0);
  const poly = L.polyline(ll, {{ color, weight: 2.5, opacity: 0.7 }});
  poly.bindPopup(`<b>${{r.label}}</b><br>HR: ${{r.hr}}/10<br>Score: ${{r.score}}m`);
  poly.addTo(bigMap);
}});
if (allBounds.length) bigMap.fitBounds(L.latLngBounds(allBounds), {{padding: [30, 30]}});

// Card grid
const cardMaps = {{}};

function buildStarsHTML(routeId) {{
  const current = getRating(routeId);
  let html = '<div class="stars" data-route="' + routeId + '">';
  for (let i = 1; i <= 10; i++) {{
    const active = current && i <= current ? ' active' : '';
    html += '<span class="star' + active + '" data-val="' + i + '">&#9733;</span>';
  }}
  html += '</div>';
  html += '<span class="rating-value" id="rv-' + routeId + '">' + (current || '—') + '</span>';
  if (current) html += '<span class="clear-rating" data-route="' + routeId + '">clear</span>';
  return html;
}}

function setupStarEvents(card, routeId) {{
  const stars = card.querySelectorAll('.stars[data-route="' + routeId + '"] .star');
  stars.forEach(star => {{
    star.addEventListener('mouseenter', function() {{
      const val = parseInt(this.dataset.val);
      stars.forEach((s, i) => s.classList.toggle('hover', i < val));
    }});
    star.addEventListener('mouseleave', function() {{
      stars.forEach(s => s.classList.remove('hover'));
    }});
    star.addEventListener('click', function() {{
      const val = parseInt(this.dataset.val);
      setRating(routeId, val);
      stars.forEach((s, i) => s.classList.toggle('active', i < val));
      const rv = document.getElementById('rv-' + routeId);
      if (rv) rv.textContent = val;
      // Add clear button if not present
      const container = this.closest('.manual-rating');
      if (!container.querySelector('.clear-rating')) {{
        const clr = document.createElement('span');
        clr.className = 'clear-rating';
        clr.dataset.route = routeId;
        clr.textContent = 'clear';
        clr.addEventListener('click', () => clearRating(routeId));
        container.appendChild(clr);
      }}
      // Mark card as rated
      card.classList.add('rated');
    }});
  }});
  const clr = card.querySelector('.clear-rating[data-route="' + routeId + '"]');
  if (clr) clr.addEventListener('click', () => clearRating(routeId));
}}

function clearRating(routeId) {{
  setRating(routeId, null);
  reSort();
}}

function renderCards(routes) {{
  const grid = document.getElementById('cardGrid');
  grid.innerHTML = '';
  Object.values(cardMaps).forEach(m => m.remove());
  Object.keys(cardMaps).forEach(k => delete cardMaps[k]);

  routes.forEach((r, idx) => {{
    const hr = r.hr || 0;
    const color = hrColor(hr);
    const bd = parseBreakdown(r.hr_explain);
    const manualRating = getRating(r.id);

    const card = document.createElement('div');
    card.className = 'card' + (manualRating ? ' rated' : '');
    card.innerHTML = `
      <div class="card-map" id="map-${{r.id}}"></div>
      <div class="card-info">
        <div class="card-title">
          <span>${{r.label}}</span>
          <span class="rank">#${{idx + 1}}</span>
        </div>
        <div class="hr-bar-container">
          <div class="hr-bar" style="width:${{hr * 10}}%;background:${{color}}">
            HR ${{hr}}/10${{bd.raw != null ? ' (raw:'+bd.raw+' abs:'+bd.abstract+')' : ''}}
          </div>
        </div>
        <div class="manual-rating">
          <span class="label">Your rating:</span>
          ${{buildStarsHTML(r.id)}}
        </div>
        <div class="hr-breakdown">
          ${{PILL_KEYS.filter(k => bd[k] != null).map(k =>
            '<span class="hr-pill ' + pillClass(bd[k]) + '">' + (PILL_LABELS[k]||k) + ': ' + bd[k].toFixed(2) + '</span>'
          ).join('')}}
        </div>
        <div class="meta">
          <span>Score: ${{r.score != null ? r.score + 'm' : '\\u2014'}}</span>
          <span>Mode: ${{r.mode}}</span>
          <span>Time: ${{r.time_seconds || '\\u2014'}}s</span>
          <span>Pts: ${{r.route_points || r.route?.length || 0}}</span>
        </div>
      </div>
    `;
    grid.appendChild(card);
    setupStarEvents(card, r.id);

    setTimeout(() => {{
      const mapDiv = document.getElementById('map-' + r.id);
      if (!mapDiv) return;
      const m = L.map(mapDiv, {{
        zoomControl: false, attributionControl: false,
        dragging: false, scrollWheelZoom: false,
        doubleClickZoom: false, touchZoom: false,
      }}).setView([51.4543, -0.9781], 14);
      L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{maxZoom: 19}}).addTo(m);
      if (r.route && r.route.length >= 2) {{
        const ll = r.route.map(p => [p[0], p[1]]);
        L.polyline(ll, {{ color, weight: 4, opacity: 0.9 }}).addTo(m);
        m.fitBounds(L.latLngBounds(ll), {{ padding: [20, 20] }});
      }}
      cardMaps[r.id] = m;
    }}, idx * 50);
  }});
}}

function reSort() {{
  const sortBy = document.getElementById('sortBy').value;
  const filterBy = document.getElementById('filterBy').value;
  const ratings = loadRatings();

  let filtered = allRoutes.filter(r => {{
    if (filterBy === 'all') return true;
    if (filterBy === 'rated') return ratings[r.id] != null;
    if (filterBy === 'unrated') return ratings[r.id] == null;
    if (filterBy === 'v85') return r.id.startsWith('v85');
    if (filterBy === 'v84') return r.id.startsWith('v84');
    if (filterBy === 'v83') return r.id.startsWith('indent') || r.id.startsWith('sym') || r.id.startsWith('pen') || r.id.startsWith('full_heart') || r.id.startsWith('v6prox');
    if (filterBy === 'v82') return r.id.startsWith('v82');
    return true;
  }});

  if (sortBy === 'hr') {{
    filtered.sort((a, b) => (b.hr || 0) - (a.hr || 0));
  }} else if (sortBy === 'manual') {{
    filtered.sort((a, b) => {{
      const ra = ratings[a.id] || 0;
      const rb = ratings[b.id] || 0;
      return rb - ra || (b.hr || 0) - (a.hr || 0);
    }});
  }} else if (sortBy === 'score') {{
    filtered.sort((a, b) => (a.score || 1e9) - (b.score || 1e9));
  }} else if (sortBy === 'diff') {{
    filtered.sort((a, b) => {{
      const da = ratings[a.id] != null ? Math.abs(ratings[a.id] - (a.hr||0)) : -1;
      const db = ratings[b.id] != null ? Math.abs(ratings[b.id] - (b.hr||0)) : -1;
      return db - da;
    }});
  }} else {{
    filtered.sort((a, b) => {{
      const ra = (a.hr || 0) / Math.max(a.score || 1, 1);
      const rb = (b.hr || 0) / Math.max(b.score || 1, 1);
      return rb - ra;
    }});
  }}

  renderCards(filtered);
}}

updateRatingCount();
reSort();
</script>
</body>
</html>"""

output_path = os.path.join(RESULTS_DIR, "hr_ranking.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Generated: {output_path}")
print(f"Routes plotted: {len(results_with_hr)}")
print(f"\nTop 10 by HR:")
for i, r in enumerate(results_with_hr[:10]):
    print(f"  {i+1:2d}. HR={r['heart_recognizability']:.1f}  score={r.get('score','?'):>7}  {r['id']}")
