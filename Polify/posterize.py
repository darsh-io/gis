import numpy as np
from sklearn.cluster import MiniBatchKMeans

def posterize(image_array, k=64):
    """
    Fast posterize via 5-bit quantization + LUT dedup + weighted KMeans.

    Key ideas:
      1. Quantize to 5-bit/channel (32768 max unique colors) before any ML —
         collapses millions of pixels to at most 32768 representative colors.
      2. LUT-based deduplication: O(N) single pass vs O(N log N) np.unique.
      3. Frequency-weighted KMeans on unique colors only — not all 8M pixels.
      4. Final remap via a prebuilt 32768-slot output LUT: one vectorized
         fancy-index op covers the entire image.

    On multi-core machines, add ThreadPoolExecutor around step 5 for
    an additional 3-4x speedup on the remap (see commented block below).
    """
    h, w = image_array.shape[:2]
    N_SLOTS = 32 * 32 * 32  # 32768 — exactly covers 5-bit quantized space

    # ── 1. Quantize + pack ────────────────────────────────────────────────
    # Shift right 3 bits: 256 values → 32 bins per channel
    # Pack three 5-bit values into a 15-bit uint16 key
    q = image_array >> 3
    flat = q.reshape(-1, 3)
    packed = ((flat[:, 0].astype(np.uint16) << 10) |
              (flat[:, 1].astype(np.uint16) << 5)  |
               flat[:, 2].astype(np.uint16))        # shape: (H*W,)

    # ── 2. LUT dedup: find unique colors + map every pixel to a compact ID ─
    lut_id = np.full(N_SLOTS, -1, dtype=np.int32)
    lut_id[packed] = 0                                    # mark present colors
    existing_keys = np.where(lut_id >= 0)[0].astype(np.uint16)
    lut_id[existing_keys] = np.arange(len(existing_keys), dtype=np.int32)
    inverse = lut_id[packed]                              # pixel → color index

    # ── 3. Reconstruct unique colors (center each 5-bit bin: × 8 + 4) ─────
    r = (((existing_keys >> 10) & 31) * 8 + 4).astype(np.uint8)
    g = (((existing_keys >>  5) & 31) * 8 + 4).astype(np.uint8)
    b = (( existing_keys        & 31) * 8 + 4).astype(np.uint8)
    unique_colors = np.stack([r, g, b], axis=1).astype(np.float32)

    # ── 4. Frequency-weighted KMeans on unique colors only ─────────────────
    counts = np.bincount(inverse, minlength=len(existing_keys))
    kmeans = MiniBatchKMeans(n_clusters=k, n_init=1, random_state=42,
                             max_iter=50, batch_size=2048)
    kmeans.fit(unique_colors, sample_weight=counts)
    unique_labels = kmeans.predict(unique_colors)

    # ── 5. Build output LUT + single-pass remap ────────────────────────────
    palette = kmeans.cluster_centers_.astype(np.uint8)
    full_lut = np.zeros((N_SLOTS, 3), dtype=np.uint8)
    full_lut[existing_keys] = palette[unique_labels]
    return full_lut[packed].reshape(h, w, 3)

    # ── Optional: parallel remap for multi-core machines ───────────────────
    # import os, concurrent.futures
    # NTHREADS = os.cpu_count()
    # chunks = np.array_split(packed, NTHREADS)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=NTHREADS) as ex:
    #     parts = list(ex.map(lambda c: full_lut[c], chunks))
    # return np.concatenate(parts).reshape(h, w, 3)
