import cv2
import numpy as np


def extract_contours_with_hierarchy(mask):
    """
    Extract contours from a binary mask with full CCOMP hierarchy.
    Returns the raw OpenCV contours + hierarchy for lossless round-trip
    via cv2.drawContours.

    Args:
        mask (np.ndarray): Boolean or 0/1 mask (H, W)

    Returns:
        tuple: (contours_list, hierarchy_array) or ([], None)
            contours_list: list of Nx1x2 arrays
            hierarchy_array: Nx4 array [next, prev, first_child, parent]
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask = mask * 255

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return [], None

    return contours, hierarchy[0]  # hierarchy[0] removes the extra dimension


def extract_all_contours_with_hierarchy(segments):
    """
    Extract contours with CCOMP hierarchy for all segmented color regions.

    Args:
        segments (dict): {color_tuple: mask}

    Returns:
        dict: {color_tuple: (contours_list, hierarchy_array)}
    """
    result = {}

    for color, mask in segments.items():
        contours, hierarchy = extract_contours_with_hierarchy(mask)
        if contours:
            result[color] = (contours, hierarchy)

    return result


def contour_to_points(contour):
    """
    Convert a raw OpenCV contour to a flat list of [x, y] points.
    No simplification — preserves every vertex from CHAIN_APPROX_SIMPLE.

    Args:
        contour (np.ndarray): Nx1x2 OpenCV contour

    Returns:
        list of [x, y] pairs
    """
    pts = contour.reshape(-1, 2)
    return pts.tolist()


def contour_to_polygon_string(contour, epsilon_ratio=None):
    """
    Convert a contour into a polygon string for Desmos.
    Y coordinates are negated so the image isn't rendered upside down.

    If epsilon_ratio is None, no simplification is applied (pixel-perfect).
    If epsilon_ratio is a float, RDP simplification is applied.

    Returns:
        dict: {
            "points": [(x, y), ...],  # raw points (un-negated)
            "equation": "polygon(...)"  # Desmos-format with negated Y
        }
    """
    if epsilon_ratio is not None:
        points = simplify_contour(contour, epsilon_ratio=epsilon_ratio)
    else:
        points = contour.reshape(-1, 2)

    if len(points) < 3:
        return {"points": points.tolist(), "equation": ""}

    point_strings = []
    for x, y in points:
        point_strings.append(f"({x}, {-y})")

    equation = "polygon(" + ", ".join(point_strings) + ")"

    return {"points": points.tolist(), "equation": equation}


def simplify_contour(contour, epsilon_ratio=0.001):
    """
    Simplify a contour using RDP algorithm.
    Only used when explicit simplification is requested.

    Args:
        contour (np.ndarray): contour returned by OpenCV.
        epsilon_ratio (float): fraction of contour perimeter used for simplification.

    Returns:
        np.ndarray: simplified points shaped (N, 2).
    """
    if contour is None or len(contour) == 0:
        return np.empty((0, 2), dtype=np.int32)

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, float(epsilon_ratio) * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2)


# ──────────────────────────────────────────────────────────────────────
# Legacy functions (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────

def extract_contours(mask):
    """
    Extract contours from a binary mask (legacy, outer-only).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = mask * 255
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def extract_all_contours(segments):
    """
    Extract contours for all segmented regions (legacy, outer-only).
    """
    result = {}
    for color, mask in segments.items():
        contours = extract_contours(mask)
        if contours:
            result[color] = contours
    return result
