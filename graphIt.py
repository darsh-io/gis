import re
import ast
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


POLYGON_RE = re.compile(r"^\s*polygon\((.*?)\)\s*$", re.IGNORECASE)


def parse_polygon(line):
    match = POLYGON_RE.match(line)
    if not match:
        raise ValueError(f"Could not parse polygon: {line!r}")
    
    points_str = match.group(1)
    
    try:
        points = ast.literal_eval(f"[{points_str}]")
        return np.array(points, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error parsing points {points_str!r}: {e}")


def load_contour_polygons(text_path):
    raw_text = Path(text_path).read_text(encoding="utf-8")
    items = []
    current_color = (0, 0, 0)
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# color:"):
            color_str = line.split(":", 1)[1].strip()
            try:
                current_color = ast.literal_eval(color_str)
            except Exception:
                pass
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        try:
            poly = parse_polygon(line)
            items.append((poly, current_color))
        except ValueError:
            pass
    return items


# ──────────────────────────────────────────────────────────────────────
# Pixel-Perfect Renderer (OpenCV drawContours with hierarchy)
# ──────────────────────────────────────────────────────────────────────

def render_gis_v3(gis_data):
    """
    Render a decoded GIS v3 file to a pixel-perfect numpy image.

    Uses cv2.drawContours with the stored CCOMP hierarchy for each color,
    which is the exact inverse of cv2.findContours — guaranteed lossless.

    Args:
        gis_data: dict from decode_gis_v3 with keys:
            "width", "height", "bg_color", "color_contours"

    Returns:
        np.ndarray: RGB image (H, W, 3) uint8
    """
    w = gis_data["width"]
    h = gis_data["height"]
    bg = gis_data["bg_color"]
    color_contours = gis_data["color_contours"]

    # Start with background color
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = bg

    # For each color, build a mask using drawContours with hierarchy,
    # then apply the color where the mask is set.
    for color, (contours, hierarchy) in color_contours.items():
        if not contours:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)

        # Wrap hierarchy in the extra dimension that drawContours expects
        # drawContours needs hierarchy shape (1, N, 4)
        hier_3d = hierarchy.reshape(1, -1, 4)

        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED, hierarchy=hier_3d)

        # Apply color where mask is set
        canvas[mask > 0] = color

    return canvas


# ──────────────────────────────────────────────────────────────────────
# Matplotlib Renderer (for display / legacy v2 / text files)
# ──────────────────────────────────────────────────────────────────────

def estimate_bounds(items):
    if not items:
        return 0.0, 1.0, 0.0, 1.0

    all_points = np.vstack([poly for poly, color in items])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)

    return (
        x_min - 0.05 * x_span,
        x_max + 0.05 * x_span,
        y_min - 0.05 * y_span,
        y_max + 0.05 * y_span,
    )


def build_canvas(items):
    x_min, x_max, y_min, y_max = estimate_bounds(items)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    for pts, color in items:
        c = [val / 255.0 for val in color]
        poly_patch = Polygon(pts, closed=True, facecolor=c, edgecolor=c, linewidth=2.0, joinstyle='round')
        ax.add_patch(poly_patch)

    ax.axis("off")
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    txt_path = input("Enter the path to the contour/GIS file: ").strip('"')
    
    if txt_path.endswith(".gis"):
        data = Path(txt_path).read_bytes()
        magic = data[:4]

        if magic == b"GIS3":
            from gis import decode_gis_v3
            gis_data = decode_gis_v3(data)

            # Pixel-perfect render
            rendered = render_gis_v3(gis_data)

            output_path = Path(txt_path).with_name(f"{Path(txt_path).stem}_graph.png")
            cv2.imwrite(str(output_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            print(f"Saved pixel-perfect render to {output_path}")
            print(f"  Dimensions: {rendered.shape[1]}x{rendered.shape[0]}")

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rendered)
            ax.set_title("GIS v3 Pixel-Perfect Render")
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        elif magic == b"GIS2":
            from gis import decode_gis_v2
            gis_data = decode_gis_v2(data)
            items = gis_data["items"]
        else:
            raise ValueError(f"Unknown GIS format: {magic!r}")
    else:
        items = load_contour_polygons(txt_path)

    if not items:
        raise ValueError("No polygons/data found in the file.")

    fig, ax = build_canvas(items)

    output_path = Path(txt_path).with_name(f"{Path(txt_path).stem}_graph.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved graph to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()