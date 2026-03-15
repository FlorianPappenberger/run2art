"""
v87_llm_planner.py — v8.7 AI-Driven Route Planning via Ollama
==============================================================
Uses a local Ollama LLM server to suggest heart-shaped GPS art routes.
The LLM acts as a high-level route planner: it generates waypoints that
are subsequently snapped to the OSM road/footpath network and stitched
together using A*-weighted shortest paths.

Features:
  - Structured GPS-art prompt with parametric heart guidance
  - Parses primary + alternative route suggestions from LLM JSON output
  - Validates that returned coordinates are within a UK bounding box
  - Snaps each waypoint to the nearest graph node (with max-snap-distance guard)
  - Routes between consecutive waypoints using networkx shortest_path
  - Module-level response cache — both v87 test variants share one API call
  - Safe fallback: returns empty route on any LLM / network error

Standalone smoke-test:
    python v87_llm_planner.py
"""

import base64
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# ── project imports ──
from geometry import haversine, sample_polyline

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "https://llms.ecmwf-puserbr.compute.cci1.ecmwf.int/api/generate",
)
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder-next:latest")

# Basic-auth credentials (override via env vars for CI / shared machines)
OLLAMA_USER = os.environ.get("OLLAMA_USER", "florian")
OLLAMA_PASS = os.environ.get("OLLAMA_PASS", "tuvalu")

CENTER = [51.4543, -0.9781]
LOCATION_NAME = "Reading, UK"

# Maximum distance (metres) to allow when snapping a waypoint to a graph node.
# Waypoints further than this are skipped.
MAX_SNAP_M = 600

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an expert GPS art route planner specializing in creating visually \
recognizable shapes like hearts for running or cycling activities. Your goal \
is to generate routes that are safe, runnable ({length_km}km total length), \
and highly symmetric with clear features: two rounded upper lobes, a pointed \
bottom, and smooth curves. Prioritize using a mix of roads, footways, \
cycleways, and open spaces (e.g., parks or trails) to avoid jaggedness and \
improve flow—do not stick to roads only if open areas allow better shape fidelity.

Input constraints:
- Location: Center at {location} (latitude, longitude) in {location_name}.
- Shape: A classic symmetric heart (use parametric inspiration like \
x=16 sin\u00b3(t), y=13 cos(t)-5 cos(2t)-2 cos(3t)-cos(4t) for t=0 to 2\u03c0, \
but adapt to real geography).
- Scale: Start with {scale} degrees diameter to fit the area, but suggest a \
variant if possible without distortion.
- Avoid: Highways, steep elevations, dead-ends; ensure the route is a closed loop.
- Evaluation: Aim for high human recognizability (score 8-10/10 based on \
symmetry, lobes, closure, smoothness, and low deviation from ideal heart).

Output strictly as valid JSON with EXACTLY this structure \
(no markdown fences, no explanation outside the JSON):
{{
  "primary": {{
    "description": "A brief verbal summary of the route, including start/end \
point, key landmarks, and why it forms a good heart.",
    "points": [[lat1, lon1], [lat2, lon2]],
    "length_km": 6.5,
    "hr_score": "8/10 \u2014 explanation",
    "improvements": "suggestions for refinement"
  }},
  "alternative": {{
    "description": "Alternative variant with more open spaces or different \
landmark anchoring.",
    "points": [[lat1, lon1], [lat2, lon2]],
    "length_km": 6.5,
    "hr_score": "8/10 \u2014 explanation",
    "improvements": "suggestions for refinement"
  }}
}}

Each "points" array must contain 60-120 GPS coordinates (latitude first, \
longitude second) forming the complete closed heart loop. Use real streets, \
paths, parks, and landmarks from {location_name} centered near {location}. \
The first and last point should be the same location to close the loop.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL LLM RESPONSE CACHE
# ─────────────────────────────────────────────────────────────────────────────

_llm_response_cache: dict[str, tuple[dict, dict]] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  OLLAMA HTTP CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _make_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that accepts the ECMWF self-signed certificate."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    url: str = OLLAMA_URL,
    timeout: int = 180,
    user: str = OLLAMA_USER,
    password: str = OLLAMA_PASS,
) -> str:
    """POST to Ollama /api/generate endpoint and return the response text.

    Supports HTTPS with Basic auth for remote Ollama servers.
    Raises RuntimeError if the server is unreachable or returns an error.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.35,
            "num_predict": 4096,
            "top_p": 0.9,
        },
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if user and password:
        creds = base64.b64encode(f"{user}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    ssl_ctx = _make_ssl_context() if url.startswith("https") else None

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama server unreachable at {url}: {exc}\n"
            "Check URL, credentials, and network connectivity."
        ) from exc

    try:
        data = json.loads(raw)
        return data.get("response", "")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON from Ollama: {exc}\nRaw (first 500 chars): {raw[:500]}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
#  JSON EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_from_text(text: str) -> dict | None:
    """Extract the first valid JSON object from arbitrary LLM output.

    LLMs often wrap JSON in markdown code fences or add preamble text;
    this function strips common wrappers and scans for a brace-balanced block.
    """
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Locate the outermost { ... } block
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[brace_start:], start=brace_start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start: i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass  # may be malformed; keep scanning

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  UK COORDINATE VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def _valid_uk_point(p: Any) -> bool:
    """Return True if *p* is a [lat, lon] pair within a generous UK bbox."""
    return (
        isinstance(p, (list, tuple))
        and len(p) == 2
        and 49.0 < float(p[0]) < 62.0
        and -10.0 < float(p[1]) < 5.0
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LLM PLAN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def llm_plan_heart_route(
    center: list,
    length_km: float = 6.5,
    scale: float = 0.018,
    location_name: str = LOCATION_NAME,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout: int = 180,
) -> tuple[dict, dict]:
    """Call Ollama to plan a heart-shaped GPS art route.

    Returns (primary_plan, alternative_plan) dicts, each with keys:
        description  — LLM narrative of the route
        points       — list of validated [lat, lon] pairs
        length_km    — LLM-estimated distance
        hr_score     — LLM self-assessed recognizability string
        improvements — LLM suggestions

    Returns ({}, {}) on any failure (server down, bad JSON, no valid points).
    Results are cached by (center, length_km, scale, model) to avoid duplicate
    API calls when both the primary and alternative test variants run.
    """
    cache_key = f"{center[0]:.4f},{center[1]:.4f}|{length_km}|{scale}|{model}"
    if cache_key in _llm_response_cache:
        log("[v8.7-llm] Using cached LLM response")
        return _llm_response_cache[cache_key]

    location = f"{center[0]:.4f}, {center[1]:.4f}"
    prompt = PROMPT_TEMPLATE.format(
        location=location,
        location_name=location_name,
        length_km=length_km,
        scale=scale,
    )

    log(f"[v8.7-llm] Querying Ollama model='{model}' at {ollama_url} ...")
    log(f"[v8.7-llm] Prompt length: {len(prompt)} chars")
    t0 = time.time()

    try:
        response_text = call_ollama(prompt, model=model, url=ollama_url, timeout=timeout)
    except RuntimeError as exc:
        log(f"[v8.7-llm] ERROR contacting Ollama: {exc}")
        result: tuple[dict, dict] = ({}, {})
        _llm_response_cache[cache_key] = result
        return result

    elapsed = time.time() - t0
    log(f"[v8.7-llm] Response received in {elapsed:.1f}s ({len(response_text)} chars)")
    log(f"[v8.7-llm] Response preview: {response_text[:400]!r}")

    parsed = _extract_json_from_text(response_text)
    if not parsed:
        log("[v8.7-llm] Could not extract JSON from LLM response")
        result = ({}, {})
        _llm_response_cache[cache_key] = result
        return result

    primary: dict = parsed.get("primary", {})
    alternative: dict = parsed.get("alternative", {})

    # Validate and sanitise each plan
    for plan_name, plan in [("primary", primary), ("alternative", alternative)]:
        raw_pts = plan.get("points", [])
        if not isinstance(raw_pts, list):
            plan["points"] = []
            continue
        valid_pts = [
            [float(p[0]), float(p[1])]
            for p in raw_pts
            if _valid_uk_point(p)
        ]
        plan["points"] = valid_pts
        log(
            f"[v8.7-llm] {plan_name}: {len(valid_pts)}/{len(raw_pts)} valid UK points | "
            f"hr_score={plan.get('hr_score', '?')!r}"
        )
        if valid_pts:
            log(f"[v8.7-llm] {plan_name} description: {plan.get('description', '')[:120]}")

    result = (primary, alternative)
    _llm_response_cache[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  WAYPOINT → ROUTED PATH
# ─────────────────────────────────────────────────────────────────────────────

def route_from_llm_waypoints(G, waypoints: list) -> list:
    """Snap LLM waypoints to the road graph and stitch a routed path.

    Algorithm:
        1. Build cKDTree over all graph nodes.
        2. Snap each waypoint to its nearest node within MAX_SNAP_M; skip
           waypoints that are further away (hallucinated coordinates).
        3. For each consecutive pair of snapped nodes, compute
           nx.shortest_path(..., weight='length').
        4. Concatenate node GPS coords, deduplicating at junctions.
        5. Close the loop if the last point != first point.

    Returns a list of [lat, lon] GPS coordinates, or [] on failure.
    """
    import networkx as nx

    if not waypoints or len(waypoints) < 3:
        return []

    # Step 1: Build spatial index over graph nodes
    node_ids = list(G.nodes())
    coords = np.array(
        [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in node_ids],
        dtype=np.float64,
    )
    tree = cKDTree(coords)

    def snap(lat: float, lon: float) -> int | None:
        """Return nearest node id, or None if beyond MAX_SNAP_M."""
        _, idx = tree.query([lat, lon], k=1)
        node = node_ids[idx]
        dist_m = haversine(lat, lon, G.nodes[node]["y"], G.nodes[node]["x"])
        if dist_m > MAX_SNAP_M:
            log(f"[v8.7-llm] Waypoint ({lat:.4f},{lon:.4f}) too far from graph "
                f"({dist_m:.0f}m > {MAX_SNAP_M}m) — skipping")
            return None
        return node

    # Step 2: Snap waypoints
    snapped: list[int] = []
    for p in waypoints:
        node = snap(p[0], p[1])
        if node is not None:
            snapped.append(node)

    if len(snapped) < 3:
        log(f"[v8.7-llm] Only {len(snapped)} waypoints snapped to graph — aborting")
        return []

    # Close the loop
    if snapped[-1] != snapped[0]:
        snapped.append(snapped[0])

    # Step 3 & 4: Route between consecutive snapped nodes
    full_route: list = []
    failed = 0
    for i in range(len(snapped) - 1):
        src, dst = snapped[i], snapped[i + 1]
        if src == dst:
            continue
        try:
            path_nodes = nx.shortest_path(G, src, dst, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            failed += 1
            # Fallback: include src node coordinate and jump to dst
            nd = G.nodes[src]
            full_route.append([nd["y"], nd["x"]])
            continue

        for node in path_nodes[:-1]:  # exclude last to avoid duplicates at junctions
            nd = G.nodes[node]
            full_route.append([nd["y"], nd["x"]])

    if failed:
        log(f"[v8.7-llm] {failed}/{len(snapped)-1} segments had no path — used direct jump")

    # Step 5: Close the loop
    if full_route and full_route[0] != full_route[-1]:
        full_route.append(full_route[0])

    return full_route


# ─────────────────────────────────────────────────────────────────────────────
#  HIGH-LEVEL TEST ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_planned_test(
    G,
    kdtree_data,
    center: list,
    config: dict,
) -> tuple[float, list, dict]:
    """Full pipeline: LLM plan → waypoints → routed path → (score, route, params).

    config keys (all optional):
        llm_variant   — "primary" or "alternative" (default "primary")
        llm_model     — Ollama model name (default DEFAULT_MODEL)
        ollama_url    — Ollama server URL (default OLLAMA_URL)
        llm_length_km — Target route length hint given to LLM (default 6.5)
        llm_scale     — Scale hint in degrees (default 0.018)
        llm_timeout   — API timeout in seconds (default 180)

    Returns:
        (score, route, params)
        score  — approximate shape-quality proxy (lower = better)
                 For LLM routes this is computed as mean nearest-distance
                 from route samples to ideal heart in metres; 1e9 on failure.
        route  — list of [lat, lon] GPS coords
        params — metadata dict (llm_description, llm_hr_score, etc.)
    """
    variant = config.get("llm_variant", "primary")
    model = config.get("llm_model", DEFAULT_MODEL)
    ollama_url = config.get("ollama_url", OLLAMA_URL)
    length_km = config.get("llm_length_km", 6.5)
    scale = config.get("llm_scale", 0.018)
    timeout = config.get("llm_timeout", 180)

    primary, alternative = llm_plan_heart_route(
        center=center,
        length_km=length_km,
        scale=scale,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
    )

    plan = primary if variant == "primary" else alternative

    if not plan or not plan.get("points"):
        log(f"[v8.7-llm] No valid '{variant}' plan returned by LLM")
        return 1e9, [], {}

    waypoints = plan["points"]
    log(f"[v8.7-llm] Routing '{variant}' through {len(waypoints)} LLM waypoints ...")

    route = route_from_llm_waypoints(G, waypoints)
    if not route or len(route) < 10:
        log(f"[v8.7-llm] '{variant}': post-routing produced too few points ({len(route)})")
        return 1e9, [], {}

    # Approximate shape-quality score: mean distance (metres) from sampled
    # route points to the nearest point on the ideal parametric heart, scaled
    # into metres.  Lower is better, comparable to the Fréchet score.
    score = _approx_heart_score(route, center, scale)

    params = {
        "rotation": 0.0,
        "scale": scale,
        "center": center,
        "llm_description": plan.get("description", ""),
        "llm_hr_score": plan.get("hr_score", ""),
        "llm_improvements": plan.get("improvements", ""),
        "llm_variant": variant,
        "llm_waypoints": len(waypoints),
    }

    log(
        f"[v8.7-llm] '{variant}': {len(route)} routed points, approx_score={score:.1f}m"
    )
    log(f"[v8.7-llm] LLM description: {plan.get('description', '')[:140]}")
    return score, route, params


# ─────────────────────────────────────────────────────────────────────────────
#  SHAPE QUALITY PROXY SCORER
# ─────────────────────────────────────────────────────────────────────────────

def _approx_heart_score(route: list, center: list, scale: float) -> float:
    """Estimate Fréchet-like distance (metres) between *route* and ideal heart.

    Generates the parametric valentine heart at the given center/scale,
    samples both curves to 80 points, computes mean nearest-neighbour
    distance in metres from route samples to ideal samples.
    """
    try:
        from v86_open_experiments import get_heart_blueprint
        from geometry import shape_to_latlngs

        ideal_norm = get_heart_blueprint("valentine", n=80)
        ideal_gps = shape_to_latlngs(ideal_norm, center, scale, rotation=0)

        n_sample = 80
        route_s = sample_polyline(route, n_sample)
        ideal_s = sample_polyline(ideal_gps, n_sample)

        ra = np.array(route_s, dtype=np.float64)
        ia = np.array(ideal_s, dtype=np.float64)

        # Build KD-tree in approximate metres (multiply degrees by ~111000)
        ra_m = ra * np.array([111000.0, 111000.0 * math.cos(math.radians(center[0]))])
        ia_m = ia * np.array([111000.0, 111000.0 * math.cos(math.radians(center[0]))])

        tree = cKDTree(ia_m)
        dists, _ = tree.query(ra_m, k=1)
        return float(np.mean(dists))
    except Exception as exc:
        log(f"[v8.7-llm] _approx_heart_score failed ({exc}), returning 999")
        return 999.0


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 60)
    log("v8.7 LLM Planner — Smoke Test")
    log("=" * 60)

    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    url = sys.argv[2] if len(sys.argv) > 2 else OLLAMA_URL

    log(f"Model : {model}")
    log(f"Server: {url}")
    log("")

    primary, alt = llm_plan_heart_route(
        center=CENTER,
        model=model,
        ollama_url=url,
    )

    for name, plan in [("PRIMARY", primary), ("ALTERNATIVE", alt)]:
        log(f"\n--- {name} ---")
        if not plan:
            log("  (empty — LLM call failed)")
            continue
        pts = plan.get("points", [])
        log(f"  Points     : {len(pts)}")
        log(f"  Length km  : {plan.get('length_km', '?')}")
        log(f"  HR score   : {plan.get('hr_score', '?')}")
        log(f"  Description: {plan.get('description', '')[:200]}")
        log(f"  First 3 pts: {pts[:3]}")
        log(f"  Improvements: {plan.get('improvements', '')[:160]}")
