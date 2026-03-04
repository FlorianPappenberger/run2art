# Shape Template Design Guide

> How the 20 built-in shape templates are designed, why they work, and how to create new ones.

## Shape Format

Each shape is defined as normalised `[x, y]` coordinates in a `[0, 1]` bounding box:

```javascript
{ name: 'Heart', pts: [[0.5, 0], [0.75, -0.18], [1, 0.15], ...] }
```

- **x** increases left → right (maps to longitude)
- **y** increases top → bottom (maps to latitude, inverted)
- Coordinates may exceed `[0, 1]` slightly (e.g., Heart uses `y = -0.18` for the upper lobes)
- The first and last point should be identical to close the shape
- Points are connected in order to form the outline

## Design Principles for GPS Art Shapes

These principles are derived from research across the GPS art ecosystem (Waschk & Krüger 2018, Li & Fu 2026, community feedback from GPS Art App, Strava art creators):

### 1. Recognisability at Low Resolution
Road networks have limited resolution. A shape must be recognisable even when rendered with only 15–30 road segments. This means:
- **Emphasise distinctive features** — the dip in a heart, the points of a star
- **Avoid fine detail** — intricate patterns collapse into noise on roads
- **Use smooth curves with clear direction changes** rather than many small wiggles

### 2. Optimal Vertex Count
- **Too few vertices** (< 6): Shape is ambiguous, router has too much freedom
- **Too many vertices** (> 25): Engine spends excessive time routing, and dense vertices on straight sections add no value
- **Sweet spot: 8–20 vertices** for most shapes
- Place vertices at **direction changes**, not evenly around the perimeter
- The adaptive densification engine adds intermediate points automatically

### 3. Smooth Curves over Sharp Angles
- Very sharp angles (< 30°) force the router into U-turns, which rarely exist in road networks
- Prefer angles of 60°–150° at vertices
- Use multiple vertices to approximate curves rather than single sharp corners

### 4. Closed Paths (Return to Start)
- All shapes should close (first point = last point)
- This ensures the runner returns to the starting point
- The engine's scoring assumes closed shapes

### 5. Avoid Self-Intersection
- Self-crossing paths confuse the router, which may take shortcuts through the intersection
- Exception: figure-8 patterns (like Butterfly, Peace) work if the crossing is at a clearly defined vertex

### 6. Balanced Proportions
- Very elongated shapes (aspect ratio > 3:1) are hard to fit because roads are distributed roughly uniformly
- Roughly square bounding boxes (1:1 to 1.5:1) work best
- The engine's `×1.4` longitude stretch compensates for latitude distortion

### 7. Road-Network Awareness
- **Large, simple shapes work better** on sparse road networks (suburbs, rural)
- **Detailed shapes work better** in dense urban grids
- Shapes with long straight segments align naturally with grid-like streets
- Shapes with many curves perform better in organic European-style street layouts

## Shape Library (v4)

The 20 included shapes are optimised for GPS art with these criteria:

| # | Shape | Vertices | Category | Notes |
|---|-------|----------|----------|-------|
| 1 | Heart | 13 | Romance | Quintessential GPS art shape. Dual-lobe top with pronounced dip, smooth curves. |
| 2 | Star | 11 | Classic | 5-pointed, well-separated inner/outer points. Inner radius ~40% of outer. |
| 3 | Smiley | 15 | Emoji | Circle with inward smile curve at bottom. Clean single-stroke design. |
| 4 | Peace Sign | 16 | Classic | Circle + vertical line + two diagonals. Efficient Euler-path routing. |
| 5 | Cat | 17 | Animals | Cat face with prominent triangular ears, V-bridge, rounded chin. |
| 6 | Bone | 19 | Animals | Dog bone with bulbous ends. Popular with pet lovers. |
| 7 | Fish | 13 | Animals | Classic fish profile with tail fork. |
| 8 | Diamond | 11 | Classic | Gemstone with faceted crown, girdle, and sharp pavilion point. |
| 9 | Butterfly | 17 | Nature | Symmetrical wings with defined body centre. |
| 10 | Flower | 17 | Nature | 6-petal rosette pattern, good for dense areas. |
| 11 | Christmas Tree | 16 | Seasonal | Zigzag silhouette with trunk. Iconic seasonal shape. |
| 12 | Crescent Moon | 14 | Nature | Curved crescent. Works well in organic street layouts. |
| 13 | Airplane | 15 | Transport | Top-down view with swept wings and tail. |
| 14 | Sailboat | 11 | Transport | Triangular sail above flat hull. |
| 15 | House | 10 | Objects | Pentagon — roof triangle + square base + chimney. Excellent road alignment. |
| 16 | Crown | 10 | Classic | 3-pointed crown. Simple, regal, fits grid streets well. |
| 17 | Thumbs Up | 19 | Emoji | Upward thumb with curled fist. Clean silhouette. |
| 18 | Arrow | 8 | Classic | Chevron arrowhead + rectangular shaft. Few vertices, reliable. |
| 19 | Lightning | 8 | Classic | Zigzag bolt. Few vertices, works in sparse areas. |
| 20 | Music Note | 13 | Culture | Eighth note with stem and flag. Distinctive silhouette. |

## Creating New Shapes

### Method 1: Manual Coordinate Entry

1. Sketch the shape on graph paper or a digital canvas
2. Mark key vertices at direction changes
3. Normalise coordinates to `[0, 1]` bounding box
4. Test in the app with Quick Fit
5. Iterate: add vertices at corners that get cut, remove unnecessary ones on straights

### Method 2: SVG Trace

1. Find or create an SVG outline of the shape
2. Extract the path coordinates
3. Normalise to `[0, 1]`
4. Simplify to 8–20 key vertices
5. Ensure the path is closed

### Validation Checklist

- [ ] First point = last point (closed path)
- [ ] 8–20 vertices
- [ ] No angles sharper than 30° (unless intentional like a star)
- [ ] Roughly square bounding box
- [ ] Recognisable when drawn at 80×80 pixels (thumbnail test)
- [ ] Quick Fit produces a recognisable result at default scale
