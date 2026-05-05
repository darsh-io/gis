# GIS — Geometric Image Storage

> **Compress images using shapes instead of pixels.**

Instead of storing every pixel, GIS traces the color regions of an image as polygons — then stores just the shapes and their colors. The result is a compact binary format that reconstructs images with zero distortion after posterization.

---

## Results

| Image Type | vs. PNG | vs. JPEG |
|---|---|---|
| Structured / graphic | −62% | −45% |
| Real-world scene | −51% | −35% |
| Reconstruction fidelity | PSNR = ∞ | — |

> Pixel-perfect reconstruction relative to the posterized source. No lossy artifacts. No block noise.

---

## What It Does

A standard image stores millions of individual pixel values.  
GIS asks a different question: **what if you stored the shapes instead?**

1. Simplify the image to a small color palette
2. Find the boundary of each color region
3. Store those boundaries as polygons — with full nesting structure
4. Reconstruct by filling the polygons back in

The `.gis` binary holds polygon contours, hierarchy, and a palette. Nothing else.

---

## Pipeline

```
Input image
    │
    ▼
Bilateral filter          — smooth noise, preserve edges
    │
    ▼
Posterization             — reduce to compact color palette
  (MiniBatch KMeans)        via fast quantization
    │
    ▼
Color segmentation        — isolate each unique color region
    │
    ▼
Contour extraction        — polygon boundaries + hierarchy
  (CCOMP hierarchy)         outer regions, holes, nesting
    │
    ▼
GIS v3 encoding           — pack into custom binary (LZMA)
    │
    ▼
.gis file
    │
    ▼
Polygon rasterization     — reconstruct by filling contours
    │
    ▼
Reconstructed image
```

---

## Strengths

- **Exact reconstruction** — PSNR = ∞ vs. the posterized source. What goes in comes back out.
- **Strong compression on structured images** — illustrations, logos, maps, UI screenshots.
- **Hierarchy-aware** — holes and nested regions are encoded correctly, not approximated.
- **Simple, inspectable format** — the `.gis` binary is straightforward to decode.
- **No external compression codec dependency** — the geometry *is* the compression.

---

## Limitations

- **High-detail photographs compress less cleanly.** Complex scenes produce many small regions, which increases contour count and reduces savings.
- **Posterization is lossy.** The palette reduction step is irreversible. The format is lossless *after* that step.
- **Not a general-purpose codec.** GIS is an experiment in structure-aware compression, not a replacement for JPEG or WebP.
- **Rendering speed.** Reconstruction involves polygon rasterization per region, which is slower than decoding a pixel buffer.

---

## Usage

### Install dependencies

```bash
pip install numpy opencv-python matplotlib scikit-learn
```

### Encode an image → `.gis`

```bash
python main.py
# enter path to image when prompted
```

Output: a `.gis` file alongside your image.

### Decode `.gis` → rendered image

```bash
python graphIt.py
# enter path to .gis file when prompted
```

Output: a rendered image saved next to the `.gis` file, plus benchmark metrics printed to console.

---

## Project Structure

```
gis/
├── main.py               # End-to-end encoder + benchmark
├── graphIt.py            # Decoder and renderer
├── gis.py                # GIS v2/v3 binary encoder + decoder
├── Geometry/
│   ├── contours.py       # Contour extraction with CCOMP hierarchy
│   ├── segment.py        # Color segmentation → masks
│   └── main.py
└── Polify/
    ├── polify.py         # Posterization wrapper
    └── posterize.py      # MiniBatch KMeans palette reduction
```

---

## Console Output (Example)

```
Contours extracted: 3,847
Pixel match: 100.00%  |  PSNR: inf
PNG size:   412 KB
JPEG size:  187 KB
GIS size:   103 KB  ✓
```

---

## Future Work

- Adaptive palette sizing based on image complexity
- Progressive rendering from coarse to fine contours
- Viewer with native file association (no Python required)
- Benchmarks across standardized image corpora
- Explore variable-precision contour encoding for further size reduction

---

## Status

Experimental research prototype. The format is somewhat stable.  
Tested on Python 3.13+.

---

*Compression by geometry. Reconstruction by math.*
