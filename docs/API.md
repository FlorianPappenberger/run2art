# API Reference

> HTTP API for the GPS Art server (`server.js`).

## Base URL

```
http://localhost:3000
```

## Endpoints

### `GET /` — Frontend

Serves the single-page application from `public/index.html`.

### `GET /{path}` — Static Files

Serves static files from the `public/` directory. Supported MIME types: `.html`, `.css`, `.js`, `.json`, `.png`, `.svg`, `.ico`.

---

### `POST /api/match` — Shape Matching

The core API endpoint. Sends a shape-fitting request to the Python engine and returns the matched route.

#### Request

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |

#### Request Body

```json
{
  "mode": "fit",
  "shapes": [
    {
      "name": "Heart",
      "pts": [[0.5, 0], [0.75, -0.18], [1, 0.15], [0.85, 0.55], [0.5, 1], [0.15, 0.55], [0, 0.15], [0.25, -0.18], [0.5, 0]]
    }
  ],
  "shape_index": 0,
  "center_point": [51.505, -0.09],
  "rotation_deg": 0,
  "scale": 0.012,
  "zoom_level": 15,
  "bbox": "-0.1,51.49,-0.08,51.52"
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `mode` | string | No | `"fit"` | One of `"fit"`, `"optimize"`, `"best_shape"` |
| `shapes` | array | Yes | — | Array of shape objects, each with `name` (string) and `pts` (array of `[x, y]` normalised to `[0, 1]`) |
| `shape_index` | integer | No | `0` | Index into the `shapes` array for `fit` and `optimize` modes |
| `center_point` | `[lat, lng]` | No | `[51.505, -0.09]` | Geographic centre point for shape placement |
| `rotation_deg` | number | No | `0` | Rotation angle in degrees (only used in `fit` mode) |
| `scale` | number | No | `0.012` | Scale in degrees (~0.012° ≈ 1.3 km; only used in `fit` mode) |
| `zoom_level` | number | No | — | Current map zoom (informational, not used by engine) |
| `bbox` | string | No | — | Current map bounding box (informational, not used by engine) |

#### Response — Success

```json
{
  "route": [[51.5048, -0.0912], [51.5049, -0.0911], ...],
  "score": 42.3,
  "rotation": 15.0,
  "scale": 0.01200,
  "center": [51.505000, -0.090000],
  "shape_index": 0,
  "shape_name": "Heart"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `route` | `[[lat, lng], ...]` | Ordered list of geographic coordinates forming the matched route |
| `score` | number | Quality score in metres (lower = better). Composite of 6 metrics. |
| `rotation` | number | Optimised rotation angle in degrees |
| `scale` | number | Optimised scale in degrees |
| `center` | `[lat, lng]` | Optimised centre point |
| `shape_index` | integer | Index of the matched shape (may differ from input for `best_shape` mode) |
| `shape_name` | string | Human-readable name of the matched shape |

#### Response — Error

```json
{
  "error": "Could not trace shape on road network"
}
```

Common error messages:
- `"Invalid shape index"` — `shape_index` out of range
- `"No shapes provided"` — Empty `shapes` array (for `best_shape` mode)
- `"Could not fetch road network (osmnx required)."` — osmnx not installed (for `best_shape` mode)
- `"Could not trace shape on road network"` — No route could be found
- `"Could not fit shape. Try a different area."` — Optimiser found no workable candidate
- `"Insufficient road density for GPS art here."` — Area lacks enough roads

#### Typical Response Times

| Mode | Time | Notes |
|------|------|-------|
| `fit` | 30–120s | Includes OSM graph download (~10s first time) |
| `optimize` | 5–15 min | 4,200 coarse + ~600 fine evaluations |
| `best_shape` | 15–30 min | 6,480 coarse + ~225 fine evaluations across all shapes |

#### Example (PowerShell)

```powershell
$body = @{
  mode = "fit"
  shapes = @(@{
    name = "Heart"
    pts = @(@(0.5,0),@(0.75,-0.18),@(1,0.15),@(0.85,0.55),@(0.5,1),@(0.15,0.55),@(0,0.15),@(0.25,-0.18),@(0.5,0))
  })
  shape_index = 0
  center_point = @(51.505, -0.09)
  rotation_deg = 0
  scale = 0.012
} | ConvertTo-Json -Depth 5

$result = Invoke-RestMethod -Uri "http://localhost:3000/api/match" `
  -Method Post -Body $body -ContentType "application/json" -TimeoutSec 180
```

#### Example (cURL)

```bash
curl -X POST http://localhost:3000/api/match \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "fit",
    "shapes": [{"name":"Heart","pts":[[0.5,0],[0.75,-0.18],[1,0.15],[0.85,0.55],[0.5,1],[0.15,0.55],[0,0.15],[0.25,-0.18],[0.5,0]]}],
    "shape_index": 0,
    "center_point": [51.505, -0.09],
    "rotation_deg": 0,
    "scale": 0.012
  }'
```
