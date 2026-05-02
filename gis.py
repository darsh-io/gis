import struct
import lzma
import numpy as np
import cv2

def encode_gis_v3(width, height, color_contours, bg_color=(0, 0, 0)):
    """
    Encode GIS v3 using the Raster-Polygon strategy for maximum compression.
    Instead of storing complex contour metadata, we store the palette-indexed
    pixel grid and compress it with LZMA.
    """
    # 1. Reconstruct the posterized image from the contours
    # (We could also pass the posterized image directly, but to stay compatible
    # with the function signature, we reconstruct it here).
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # We need to draw the colors in a specific order? 
    # Actually, the segments don't overlap in a posterized image.
    for color, (contours, hierarchy) in color_contours.items():
        if not contours: continue
        # drawContours with hierarchy handles holes perfectly
        cv2.drawContours(img, contours, -1, color, cv2.FILLED, hierarchy=hierarchy)

    # 2. Build palette and index map
    pixels = img.reshape(-1, 3)
    # Sort colors by frequency for slightly better LZMA patterns
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    palette = unique_colors[np.argsort(-counts)]
    color_to_idx = {tuple(c): i for i, c in enumerate(palette)}

    index_map = np.zeros(height * width, dtype=np.uint8)
    for i, color in enumerate(palette):
        mask = np.all(img == color, axis=-1).ravel()
        index_map[mask] = i

    # 3. Payload: [width][height][bg_color][num_colors][palette][index_map]
    # We use varints for dimensions to keep it flexible
    def encode_varint(n):
        res = bytearray()
        while True:
            res.append((n & 0x7f) | (0x80 if n >> 7 else 0))
            n >>= 7
            if not n: break
        return res

    raw = bytearray()
    raw.extend(encode_varint(width))
    raw.extend(encode_varint(height))
    raw.extend(struct.pack("3B", *bg_color))
    raw.extend(encode_varint(len(palette)))
    raw.extend(palette.tobytes())
    raw.extend(index_map.tobytes())

    # 4. Compress with LZMA (preset 3 for faster compression)
    compressed = lzma.compress(bytes(raw), preset=3)
    
    out = bytearray(b"GIS3")
    out.extend(compressed)
    return bytes(out)


def decode_gis_v3(data):
    """
    Decode GIS v3 using the Raster-Polygon strategy.
    Reconstructs polygons from the compressed index map.
    """
    if data[:4] != b"GIS3":
        raise ValueError("Invalid magic bytes, not a GIS v3 file")

    def decode_varint(data, offset):
        result = 0
        shift = 0
        while offset < len(data):
            byte = data[offset]
            offset += 1
            result |= (byte & 0x7f) << shift
            if not (byte & 0x80): break
            shift += 7
        return result, offset

    raw = lzma.decompress(data[4:])
    offset = 0
    
    width, offset = decode_varint(raw, offset)
    height, offset = decode_varint(raw, offset)
    bg_r, bg_g, bg_b = struct.unpack_from("3B", raw, offset)
    offset += 3
    bg_color = (bg_r, bg_g, bg_b)
    
    num_colors, offset = decode_varint(raw, offset)
    palette_bytes = raw[offset : offset + num_colors * 3]
    offset += num_colors * 3
    palette = np.frombuffer(palette_bytes, dtype=np.uint8).reshape(-1, 3)
    
    index_map_bytes = raw[offset:]
    index_map = np.frombuffer(index_map_bytes, dtype=np.uint8).reshape(height, width)
    
    # 5. Reconstruct color_contours by running findContours on each color mask
    color_contours = {}
    for i, color in enumerate(palette):
        mask = (index_map == i).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        color_tuple = tuple(int(c) for c in color)
        color_contours[color_tuple] = (contours, hierarchy)
        
    return {
        "width": width,
        "height": height,
        "bg_color": bg_color,
        "color_contours": color_contours
    }


def encode_gis_v2(width, height, polygons_dict):
    """Legacy v2 encoder (unchanged)."""
    def encode_varint(n):
        res = bytearray()
        while True:
            res.append((n & 0x7f) | (0x80 if n >> 7 else 0))
            n >>= 7
            if not n: break
        return res
    colors = list(polygons_dict.keys())
    color_to_idx = {c: i for i, c in enumerate(colors)}
    out = bytearray(b"GIS2")
    out.extend(encode_varint(width))
    out.extend(encode_varint(height))
    out.extend(encode_varint(len(colors)))
    for c in colors: out.extend(struct.pack("3B", *c))
    for color, polys in polygons_dict.items():
        if not polys: continue
        c_idx = color_to_idx[color]
        out.extend(encode_varint(c_idx))
        out.extend(encode_varint(len(polys)))
        for poly in polys:
            out.extend(encode_varint(len(poly)))
            px, py = 0, 0
            for pt in poly:
                x, y = int(pt[0]), int(pt[1])
                dx, dy = x - px, y - py
                zdx = (dx << 1) ^ (dx >> 31)
                zdy = (dy << 1) ^ (dy >> 31)
                out.extend(encode_varint(zdx))
                out.extend(encode_varint(zdy))
                px, py = x, y
    return bytes(out)

def decode_gis_v2(data):
    """Legacy v2 decoder (unchanged)."""
    def decode_varint(data, offset):
        result = 0; shift = 0
        while offset < len(data):
            byte = data[offset]; offset += 1
            result |= (byte & 0x7f) << shift
            if not (byte & 0x80): break
            shift += 7
        return result, offset
    if data[:4] != b"GIS2": raise ValueError("Invalid magic bytes")
    offset = 4
    width, offset = decode_varint(data, offset)
    height, offset = decode_varint(data, offset)
    num_colors, offset = decode_varint(data, offset)
    colors = []
    for _ in range(num_colors):
        r, g, b = struct.unpack_from("3B", data, offset); offset += 3
        colors.append((r, g, b))
    items = []
    while offset < len(data):
        c_idx, offset = decode_varint(data, offset)
        num_polys, offset = decode_varint(data, offset)
        color = colors[c_idx]
        for _ in range(num_polys):
            num_pts, offset = decode_varint(data, offset)
            poly = []; px, py = 0, 0
            for _ in range(num_pts):
                zdx, offset = decode_varint(data, offset)
                zdy, offset = decode_varint(data, offset)
                dx = (zdx >> 1) ^ -(zdx & 1); dy = (zdy >> 1) ^ -(zdy & 1)
                px += dx; py += dy
                poly.append((px, -py))
            items.append((np.array(poly, dtype=np.float32), color))
    return {"width": width, "height": height, "items": items}
