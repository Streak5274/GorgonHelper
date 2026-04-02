"""
Map calibration script: detect landmark markers on wiki marked maps,
cross-reference with landmarks.json world coordinates, solve for precise bounds.
Uses scipy for fast connected-component labeling.
"""

import json
import os
import sys
import math
import warnings
import numpy as np
from PIL import Image
from scipy import ndimage

BASE_DIR = "C:/Users/Janne/AppData/LocalLow/Elder Game/Project Gorgon/Reports"
MARKED_DIR = os.path.join(BASE_DIR, "Maps/marked")
MAPS_DIR = os.path.join(BASE_DIR, "Maps")
LANDMARKS_FILE = os.path.join(BASE_DIR, "pg_landmarks.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "Maps/calibration_from_wiki.json")

# Current rough MAP_CONFIGS (x_left=worldMinX, x_right=worldMaxX, z_top=worldMaxZ, z_bottom=worldMinZ)
ROUGH_CONFIGS = {
    "AreaSerbule":      dict(x_left=193.2,   x_right=2575.85, z_top=2761.56, z_bottom=273.06,  rotate180=False),
    "AreaSerbule2":     dict(x_left=105,      x_right=2429,    z_top=3021,    z_bottom=238,     rotate180=False),
    "AreaEltibule":     dict(x_left=227,      x_right=2762,    z_top=3136,    z_bottom=60,      rotate180=True),
    "AreaKurMountains": dict(x_left=3847.5,   x_right=253.0,   z_top=3847.7,  z_bottom=251.7,   rotate180=False),
    "AreaSunVale":      dict(x_left=-1858,    x_right=1911,    z_top=1899,    z_bottom=-1738,   rotate180=False),
    "AreaDesert1":      dict(x_left=-1504,    x_right=712,     z_top=792,     z_bottom=-1861,   rotate180=False),
    "AreaRahu":         dict(x_left=-9,       x_right=1705,    z_top=637,     z_bottom=-1268,   rotate180=False),
    "AreaGazluk":       dict(x_left=-2437,    x_right=1984,    z_top=2635,    z_bottom=-2285,   rotate180=False),
    "AreaPovus":        dict(x_left=55,       x_right=2748,    z_top=2653,    z_bottom=110,     rotate180=False),
    "AreaFaeRealm1":    dict(x_left=330,      x_right=2697,    z_top=3075,    z_bottom=56,      rotate180=False),
    "AreaNewbieIsland": dict(x_left=145,      x_right=310,     z_top=400,     z_bottom=318,     rotate180=False),
    "AreaStatehelm":    dict(x_left=-144,     x_right=1146,    z_top=1652,    z_bottom=142,     rotate180=False),
}

LANDMARK_TYPES = {"Portal", "TeleportationPlatform", "MeditationPillar"}


def load_landmarks():
    with open(LANDMARKS_FILE) as f:
        return json.load(f)


def world_to_pixel_rough(wx, wz, cfg, img_w, img_h):
    """Project world coord to pixel using rough config."""
    x_left, x_right = cfg["x_left"], cfg["x_right"]
    z_top, z_bottom = cfg["z_top"], cfg["z_bottom"]

    if cfg.get("rotate180", False):
        # Map is rotated 180: to match wiki orientation (north up), we need to check
        # For rotate180, the wiki map is standard (north up), but game map was flipped.
        # The rough config was designed for the flipped game map.
        # For wiki map calibration, treat as un-rotated but with inverted axes:
        u = (x_left - wx) / (x_left - x_right)
        v = (wz - z_bottom) / (z_top - z_bottom)
    else:
        u = (wx - x_left) / (x_right - x_left)
        v = (z_top - wz) / (z_top - z_bottom)

    px = u * img_w
    py = v * img_h
    return px, py


def compute_hsv_masks(pixels):
    """Compute color masks from RGB pixel array. Returns dict of masks."""
    R = pixels[:, :, 0].astype(np.float32)
    G = pixels[:, :, 1].astype(np.float32)
    B = pixels[:, :, 2].astype(np.float32)

    maxRGB = np.maximum(np.maximum(R, G), B)
    minRGB = np.minimum(np.minimum(R, G), B)
    delta = maxRGB - minRGB

    V = maxRGB / 255.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.where(maxRGB > 0, delta / maxRGB, 0.0)

    H = np.zeros_like(R)
    eps = 1e-6
    mask_r = (maxRGB == R) & (delta > 0)
    mask_g = (maxRGB == G) & (delta > 0)
    mask_b = (maxRGB == B) & (delta > 0)
    H[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / (delta[mask_r] + eps))) % 360.0
    H[mask_g] = 60.0 * ((B[mask_g] - R[mask_g]) / (delta[mask_g] + eps)) + 120.0
    H[mask_b] = 60.0 * ((R[mask_b] - G[mask_b]) / (delta[mask_b] + eps)) + 240.0

    return {
        # Green (portals/teleports on many wiki maps)
        "green": (H >= 80) & (H <= 160) & (S > 0.35) & (V > 0.35),
        # Blue/indigo (meditation pillars)
        "blue": (H >= 200) & (H <= 270) & (S > 0.30) & (V > 0.25),
        # White (teleport platforms)
        "white": (V > 0.85) & (S < 0.20),
        # Orange/red
        "orange": ((H <= 40) | (H >= 340)) & (S > 0.40) & (V > 0.40),
        # Yellow
        "yellow": (H >= 40) & (H < 80) & (S > 0.40) & (V > 0.40),
        # Cyan/teal
        "cyan": (H >= 165) & (H < 200) & (S > 0.30) & (V > 0.30),
        # Purple
        "purple": (H >= 270) & (H <= 330) & (S > 0.30) & (V > 0.25),
    }, H, S, V


def find_clusters_scipy(mask, min_pixels=25, dilation_radius=8):
    """
    Find clusters of True pixels using scipy connected components.
    Applies morphological dilation first to merge nearby pixels.
    Returns list of (cx, cy, size).
    """
    if not np.any(mask):
        return []

    # Dilate to merge nearby pixels into blobs
    struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(mask, structure=struct, iterations=dilation_radius)

    # Label connected components
    labeled, num_features = ndimage.label(dilated)

    clusters = []
    for label_id in range(1, num_features + 1):
        component = labeled == label_id
        # Count original (undilated) pixels in this component
        original_pixels = np.sum(mask & component)
        if original_pixels >= min_pixels:
            ys, xs = np.where(mask & component)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            clusters.append((cx, cy, int(original_pixels)))

    return clusters


def detect_markers(img_path, min_cluster_pixels=20):
    """
    Detect colored marker clusters on wiki marked map.
    Returns list of (cx, cy, color_type, size) and image dimensions.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    pixels = np.array(img)

    masks, H, S, V = compute_hsv_masks(pixels)

    results = []
    for color_type, mask in masks.items():
        clusters = find_clusters_scipy(mask, min_pixels=min_cluster_pixels, dilation_radius=10)
        for cx, cy, size in clusters:
            results.append((cx, cy, color_type, size))

    # Deduplicate: if two clusters of different colors are within 30px, keep larger one
    deduplicated = []
    used = [False] * len(results)
    # Sort by size descending
    results_sorted = sorted(enumerate(results), key=lambda x: -x[1][3])
    for orig_i, (cx, cy, ct, sz) in results_sorted:
        if used[orig_i]:
            continue
        deduplicated.append((cx, cy, ct, sz))
        used[orig_i] = True
        # Mark nearby clusters as used
        for orig_j, (cx2, cy2, ct2, sz2) in results_sorted:
            if used[orig_j]:
                continue
            if math.sqrt((cx2-cx)**2 + (cy2-cy)**2) < 30:
                used[orig_j] = True

    return deduplicated, w, h


def solve_linear(pairs):
    """
    Solve u = A*coord + B for list of (u, coord) pairs.
    Returns (A, B) or (None, None).
    """
    if len(pairs) < 2:
        return None, None
    coords = np.array([p[1] for p in pairs], dtype=float)
    norms = np.array([p[0] for p in pairs], dtype=float)
    A_mat = np.column_stack([coords, np.ones_like(coords)])
    result = np.linalg.lstsq(A_mat, norms, rcond=None)
    A, B = result[0]
    return float(A), float(B)


def match_clusters_to_landmarks(clusters, area_landmarks, cfg, wiki_w, wiki_h, max_dist=150):
    """
    Match detected clusters to landmarks using rough projection.
    Returns list of match dicts.
    """
    matched = []
    used_clusters = set()

    # For each landmark, find nearest unmatched cluster within max_dist
    for li, lm in enumerate(area_landmarks):
        wx, wz = lm["x"], lm["z"]
        rough_px, rough_py = world_to_pixel_rough(wx, wz, cfg, wiki_w, wiki_h)

        # Skip if rough projection is wildly out of bounds
        margin = max(wiki_w, wiki_h) * 0.5
        if rough_px < -margin or rough_px > wiki_w + margin:
            continue
        if rough_py < -margin or rough_py > wiki_h + margin:
            continue

        best_dist = max_dist
        best_ci = -1
        for ci, (cx, cy, ct, sz) in enumerate(clusters):
            if ci in used_clusters:
                continue
            dist = math.sqrt((cx - rough_px)**2 + (cy - rough_py)**2)
            if dist < best_dist:
                best_dist = dist
                best_ci = ci

        if best_ci >= 0:
            cx, cy, ct, sz = clusters[best_ci]
            matched.append({
                "px": cx, "py": cy,
                "wx": wx, "wz": wz,
                "name": lm["name"],
                "type": lm["type"],
                "color": ct,
                "rough_px": rough_px, "rough_py": rough_py,
                "match_dist": best_dist,
            })
            used_clusters.add(best_ci)

    return matched


def calibrate_area(area_key, landmarks_data):
    """Run calibration for one area. Returns calibration result dict."""
    cfg = ROUGH_CONFIGS.get(area_key)
    if cfg is None:
        return {"error": f"No rough config for {area_key}"}

    marked_path = os.path.join(MARKED_DIR, f"{area_key}_MarkedMap.jpg")
    if not os.path.exists(marked_path):
        return {"error": f"No marked map: {marked_path}"}

    area_landmarks = [lm for lm in landmarks_data.get(area_key, [])
                      if lm.get("type") in LANDMARK_TYPES]
    n_landmarks = len(area_landmarks)

    print(f"\n{'='*60}")
    print(f"  {area_key}  (landmarks: {n_landmarks})")

    # Detect marker clusters on wiki map
    clusters, wiki_w, wiki_h = detect_markers(marked_path, min_cluster_pixels=20)

    print(f"  Wiki map size: {wiki_w}x{wiki_h}")
    print(f"  Detected clusters: {len(clusters)}")
    # Show top 20 by size
    for cx, cy, ct, sz in sorted(clusters, key=lambda x: -x[3])[:20]:
        print(f"    {ct:8s} at ({cx:.0f},{cy:.0f})  sz={sz}")

    base_result = {
        "area": area_key,
        "n_landmarks": n_landmarks,
        "n_clusters": len(clusters),
        "wiki_w": wiki_w,
        "wiki_h": wiki_h,
    }

    # Get game map dims
    game_map_path = None
    for name in [f"Map_{area_key}.jpg", f"Map_{area_key}.webp", f"Map_{area_key}.png",
                 "Map_Povus.jpg"]:
        p = os.path.join(MAPS_DIR, name)
        if os.path.exists(p):
            game_map_path = p
            break
    if game_map_path:
        gimg = Image.open(game_map_path)
        base_result["game_w"], base_result["game_h"] = gimg.size

    if n_landmarks == 0:
        base_result.update({"n_matched": 0, "error": "No landmarks"})
        return base_result

    if len(clusters) == 0:
        base_result.update({"n_matched": 0, "error": "No clusters detected"})
        return base_result

    matched = match_clusters_to_landmarks(clusters, area_landmarks, cfg, wiki_w, wiki_h, max_dist=150)
    n_matched = len(matched)
    print(f"  Matched: {n_matched}")
    for m in matched:
        print(f"    '{m['name']}' ({m['type']}) world=({m['wx']:.1f},{m['wz']:.1f}) "
              f"pix=({m['px']:.0f},{m['py']:.0f}) dist={m['match_dist']:.0f} [{m['color']}]")

    base_result["n_matched"] = n_matched

    if n_matched < 2:
        base_result["error"] = "Too few matches"
        base_result["matched_points"] = matched
        base_result["rough_config"] = dict(cfg)
        return base_result

    # Solve regression
    x_pairs = [(m["px"] / wiki_w, m["wx"]) for m in matched]
    z_pairs = [(m["py"] / wiki_h, m["wz"]) for m in matched]

    A_x, B_x = solve_linear(x_pairs)
    A_z, B_z = solve_linear(z_pairs)

    if A_x is None or abs(A_x) < 1e-12 or A_z is None or abs(A_z) < 1e-12:
        base_result["error"] = "Regression failed (singular)"
        return base_result

    # Recover bounds
    x_left_cal  = -B_x / A_x
    x_right_cal = (1.0 - B_x) / A_x
    z_top_cal    = -B_z / A_z
    z_bottom_cal = (1.0 - B_z) / A_z

    # Compute residuals
    residuals = []
    for m in matched:
        u_p = A_x * m["wx"] + B_x
        v_p = A_z * m["wz"] + B_z
        u_a = m["px"] / wiki_w
        v_a = m["py"] / wiki_h
        res = math.sqrt(((u_p - u_a) * wiki_w)**2 + ((v_p - v_a) * wiki_h)**2)
        residuals.append(res)

    mean_res = sum(residuals) / len(residuals)
    print(f"  Calibrated: x_left={x_left_cal:.1f}, x_right={x_right_cal:.1f}, "
          f"z_top={z_top_cal:.1f}, z_bottom={z_bottom_cal:.1f}")
    print(f"  Mean residual: {mean_res:.1f}px")

    base_result.update({
        "x_left": round(x_left_cal, 1),
        "x_right": round(x_right_cal, 1),
        "z_top": round(z_top_cal, 1),
        "z_bottom": round(z_bottom_cal, 1),
        "mean_residual_px": round(mean_res, 1),
        "matched_points": matched,
        "rough_config": dict(cfg),
        "A_x": A_x, "B_x": B_x,
        "A_z": A_z, "B_z": B_z,
    })
    return base_result


def main():
    print("Loading landmarks...")
    landmarks_data = load_landmarks()

    areas = sorted(ROUGH_CONFIGS.keys())
    results = {}

    for area_key in areas:
        result = calibrate_area(area_key, landmarks_data)
        results[area_key] = result

    # Save JSON
    def to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(OUTPUT_FILE, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")

    # Summary table
    print("\n" + "="*110)
    print("CALIBRATION SUMMARY")
    print("="*110)
    hdr = f"{'Area':<20} {'LM':>4} {'Clus':>5} {'Mtch':>5}  {'x_left':>10} {'x_right':>10} {'z_top':>10} {'z_bot':>10}  {'Res(px)':>8}  Notes"
    print(hdr)
    print("-"*110)

    for area_key in areas:
        r = results[area_key]
        nm = r.get("n_matched", 0)
        flag = " ***LOW***" if nm < 3 else ""
        err = r.get("error", "")
        if "x_left" not in r:
            print(f"{area_key:<20} {r.get('n_landmarks',0):>4} {r.get('n_clusters',0):>5} {nm:>5}  "
                  f"{'ERROR':>43}  {err}")
        else:
            print(f"{area_key:<20} {r.get('n_landmarks',0):>4} {r.get('n_clusters',0):>5} {nm:>5}  "
                  f"{r['x_left']:>10.1f} {r['x_right']:>10.1f} {r['z_top']:>10.1f} {r['z_bottom']:>10.1f}  "
                  f"{r.get('mean_residual_px',0):>8.1f}  {flag}")

    # JavaScript MAP_CONFIGS snippet
    print("\n" + "="*80)
    print("MAP_CONFIGS (JavaScript)")
    print("="*80)

    for area_key in areas:
        r = results[area_key]
        rc = ROUGH_CONFIGS[area_key]

        if "x_left" not in r:
            # Fall back to rough values
            print(f"  // {area_key}: CALIBRATION FAILED ({r.get('error','?')}) - using rough values")
            print(f"  {area_key}: {{ worldMinX:{rc['x_left']:.1f}, worldMaxX:{rc['x_right']:.1f}, "
                  f"worldMinZ:{rc['z_bottom']:.1f}, worldMaxZ:{rc['z_top']:.1f} }},")
            continue

        xl = r["x_left"]
        xr = r["x_right"]
        zt = r["z_top"]
        zb = r["z_bottom"]
        nm = r.get("n_matched", 0)
        res = r.get("mean_residual_px", 0)

        note = f"// n={nm}, res={res:.0f}px"
        if nm < 3:
            note += " *** LOW ***"
        print(f"  {note}")
        print(f"  {area_key}: {{ worldMinX:{xl:.1f}, worldMaxX:{xr:.1f}, worldMinZ:{zb:.1f}, worldMaxZ:{zt:.1f} }},")

    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    for area_key, expected in [
        ("AreaSerbule",      dict(x_left=193.2,  x_right=2575.85, z_top=2761.56, z_bottom=273.06)),
        ("AreaKurMountains", dict(x_left=3847.5, x_right=253.0,   z_top=3847.7,  z_bottom=251.7)),
    ]:
        r = results.get(area_key, {})
        print(f"\n{area_key}:")
        print(f"  Expected:   x_left={expected['x_left']:.1f}  x_right={expected['x_right']:.1f}  "
              f"z_top={expected['z_top']:.1f}  z_bottom={expected['z_bottom']:.1f}")
        if "x_left" in r:
            print(f"  Calibrated: x_left={r['x_left']:.1f}  x_right={r['x_right']:.1f}  "
                  f"z_top={r['z_top']:.1f}  z_bottom={r['z_bottom']:.1f}")
            print(f"  Delta:      Δxl={abs(r['x_left']-expected['x_left']):.1f}  "
                  f"Δxr={abs(r['x_right']-expected['x_right']):.1f}  "
                  f"Δzt={abs(r['z_top']-expected['z_top']):.1f}  "
                  f"Δzb={abs(r['z_bottom']-expected['z_bottom']):.1f}")
        else:
            print(f"  FAILED: {r.get('error','?')}")


if __name__ == "__main__":
    main()
