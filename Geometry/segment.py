import numpy as np

def segment_by_color(img):
    """
    Segments an image into regions based on unique colors.

    Args:
        img (np.ndarray): RGB image (H, W, 3)

    Returns:
        dict: {color_tuple: mask (H, W) bool array}
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Expected numpy array image")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected image shape (H, W, 3)")

    # Find unique colors
    pixels = img.reshape(-1, 3)
    unique_colors, inverse_idx = np.unique(pixels, axis=0, return_inverse=True)
    inverse_idx = inverse_idx.reshape(img.shape[:2])

    segments = {}

    for i, color in enumerate(unique_colors):
        segments[tuple(color)] = (inverse_idx == i)

    return segments
