from Polify.polify import polify
from Geometry.segment import segment_by_color
from Geometry.contours import (
    extract_all_contours_with_hierarchy,
    extract_all_contours,
    contour_to_points,
    contour_to_polygon_string,
)
from gis import encode_gis_v3, decode_gis_v3, encode_gis_v2
from graphIt import render_gis_v3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_verification(posterized_rgb, rendered_rgb):
    """
    Compute pixel-perfect verification metrics between the posterized
    source and the GIS round-trip rendered image.
    """
    if posterized_rgb.shape != rendered_rgb.shape:
        return {
            "match": False,
            "reason": f"Shape mismatch: {posterized_rgb.shape} vs {rendered_rgb.shape}",
        }

    diff = posterized_rgb.astype(np.int16) - rendered_rgb.astype(np.int16)
    abs_diff = np.abs(diff)
    
    total_pixels = posterized_rgb.shape[0] * posterized_rgb.shape[1]
    # A pixel matches if ALL 3 channels match exactly
    pixel_match = np.all(posterized_rgb == rendered_rgb, axis=-1)
    matched = int(np.sum(pixel_match))
    mismatched = total_pixels - matched
    match_pct = matched / total_pixels * 100

    # PSNR
    mse = float(np.mean(abs_diff.astype(np.float64) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(255.0**2 / mse)

    return {
        "match": mismatched == 0,
        "total_pixels": total_pixels,
        "matched_pixels": matched,
        "mismatched_pixels": mismatched,
        "match_pct": match_pct,
        "psnr_db": psnr,
        "max_channel_error": int(abs_diff.max()),
    }


if __name__ == "__main__":
    input_path = input("Enter the path to the image: ").strip('"')

    # Load image (BGR)
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise ValueError("Failed to load image. Check path.")
    print("Image loaded successfully.")

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pre-process: Detail-Preserving Bilateral Filter
    print("Applying Bilateral Filter...")
    img_rgb = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

    # Polify (posterize)
    print("Processing image...")
    polified = polify(img_rgb, colors=128)
    print("Polify done.")

    # Save posterized PNG (Removed for bloat reduction)
    # posterized_path = Path(input_path).with_name(f"{Path(input_path).stem}_posterized.png")
    # cv2.imwrite(str(posterized_path), cv2.cvtColor(polified, cv2.COLOR_RGB2BGR))
    # print(f"Saved posterized reference to {posterized_path}")

    # Segment by color
    print("Segmenting...")
    segments = segment_by_color(polified)
    print(f"{len(segments)} unique colors found.")

    # Detect background color (most frequent color = largest area)
    bg_color_raw = max(segments.keys(), key=lambda c: np.sum(segments[c]))
    bg_color = tuple(int(x) for x in bg_color_raw)
    print(f"Background color: {bg_color}")

    # Extract contours WITH hierarchy (GIS v3 pipeline)
    print("Extracting contours with CCOMP hierarchy...")
    color_contours = extract_all_contours_with_hierarchy(segments)

    # Convert to pure Python int tuples for JSON/struct packing safety
    safe_color_contours = {}
    total_regions = 0
    total_points = 0

    for color, (contours, hierarchy) in color_contours.items():
        color_tuple = tuple(int(x) for x in color)
        safe_color_contours[color_tuple] = (contours, hierarchy)
        total_regions += len(contours)
        for c in contours:
            total_points += len(c)

    print(f"  {total_regions} contours, {total_points} total vertices")

    # Also generate Desmos-compatible text (legacy v1)
    print("Generating Desmos equations...")
    equation_lines = []
    contours_dict_flat = extract_all_contours(segments)
    kept_contours = 0

    for color, contours in contours_dict_flat.items():
        color_tuple = tuple(int(x) for x in color)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2:
                continue
            desc = contour_to_polygon_string(contour)
            if not desc["equation"]:
                continue
            equation_lines.append(f"# color: {color_tuple}")
            equation_lines.append(desc["equation"])
            kept_contours += 1

    # equations_path = Path(input_path).with_name(f"{Path(input_path).stem}_desmos.txt")
    # v1_text = "\n".join(equation_lines)
    # equations_path.write_text(v1_text, encoding="utf-8")

    # ── Encode GIS v3 ──
    h, w = polified.shape[:2]
    gis_bytes = encode_gis_v3(w, h, safe_color_contours, bg_color=bg_color)
    gis_path = Path(input_path).with_name(f"{Path(input_path).stem}.gis")
    gis_path.write_bytes(gis_bytes)

    # ── Verification: Round-trip decode → render → compare ──
    print("\n--- Round-Trip Verification ---")
    gis_data = decode_gis_v3(gis_bytes)
    rendered = render_gis_v3(gis_data)

    # Save rendered output (Removed for bloat reduction)
    # rendered_path = Path(input_path).with_name(f"{Path(input_path).stem}_rendered.png")
    # cv2.imwrite(str(rendered_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

    metrics = compute_verification(polified, rendered)

    if metrics.get("reason"):
        print(f"  [FAIL] {metrics['reason']}")
    else:
        status = "[PERFECT MATCH]" if metrics["match"] else "[IMPERFECT]"
        print(f"  {status}")
        print(f"  Pixels matched: {metrics['matched_pixels']:,} / {metrics['total_pixels']:,} ({metrics['match_pct']:.4f}%)")
        if metrics["mismatched_pixels"] > 0:
            print(f"  Mismatched:     {metrics['mismatched_pixels']:,}")
            print(f"  Max channel err: {metrics['max_channel_error']}")
        print(f"  PSNR:           {metrics['psnr_db']:.2f} dB")

    # ── Benchmarking ──
    success, png_buf = cv2.imencode(".png", cv2.cvtColor(polified, cv2.COLOR_RGB2BGR))
    png_size = len(png_buf)
    success, jpg_buf = cv2.imencode(".jpg", cv2.cvtColor(polified, cv2.COLOR_RGB2BGR))
    jpg_size = len(jpg_buf)
    # v1_size = len(v1_text) # Removed v1 text size calculation
    v3_size = len(gis_bytes)

    print("\n--- Benchmark Metrics ---")
    print(f"| Format           |     Size |")
    print(f"|------------------|----------|")
    print(f"| PNG (posterized) | {png_size:7,}B |")
    print(f"| JPEG             | {jpg_size:7,}B |")
    print(f"| GIS v3 (binary)  | {v3_size:7,}B |")
    print(f"-------------------------")
    # print(f"Saved {kept_contours} Desmos equations to {equations_path}")
    print(f"Saved GIS v3 binary to {gis_path}")
    # print(f"Saved rendered output to {rendered_path}")

    # ── Visual Comparison ──
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(img_rgb)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(polified)
    axs[1].set_title("Posterized (Ground Truth)")
    axs[1].axis("off")

    axs[2].imshow(rendered)
    axs[2].set_title(f"GIS v3 Rendered ({metrics.get('match_pct', 0):.2f}% match)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
