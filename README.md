# Project Graphed Image

Project Graphed Image is an image-compression experiment built around a graph-based representation of image regions. Instead of storing the image as a flat grid of pixels, it reduces the image to a small set of posterized colors, traces the boundaries of those color regions as contours, preserves hole/nesting structure with contour hierarchy, and stores the result in a compact custom binary format (`.gis`).

The goal is not just smaller files. The goal is to preserve the visual structure of the image while replacing pixel-by-pixel storage with a region graph that can be encoded, decoded, and rendered back with very high fidelity.

## Purpose

The purpose of the project is to improve image compression using a new method based on graphs:

- first simplify the image into a limited color palette,
- then treat each connected color region as a graph of boundaries and nested holes,
- then serialize that structure into a compact binary format,
- and finally reconstruct the image from that graph structure when needed.

In short: it converts an image into a geometric/graph representation so that images with large flat regions, repeated shapes, or simple boundaries can be stored more efficiently than as raw pixels.

## What You Need

If you only have these files and folders, the project still works:

- `main.py`
- `graphIt.py`
- `gis.py`
- `Geometry/`
- `Polify/`

The code expects this structure:

```text
Project Graphed Image/
├─ main.py
├─ graphIt.py
├─ gis.py
├─ Geometry/
│  ├─ contours.py
│  ├─ segment.py
│  └─ main.py
└─ Polify/
   ├─ polify.py
   └─ posterize.py
```

## Install Dependencies

Install the Python packages the project uses:

```bash
pip install numpy opencv-python matplotlib scikit-learn
```

If you run into display issues on Windows, make sure `matplotlib` is installed normally and that your Python environment can open GUI windows.

## How To Use It

### 1) Encode an image into `.gis`

Run `main.py`:

```bash
python main.py
```

Then enter the path to an image file when prompted.

What happens next:

- the image is loaded,
- a bilateral filter smooths the image while trying to keep edges,
- `Polify/polify.py` posterizes the image into a smaller palette,
- `Geometry/segment.py` groups pixels by color,
- `Geometry/contours.py` extracts contours and hierarchy information,
- `gis.py` packs the result into a `.gis` file,
- and `graphIt.py` is used to render and verify the decoded result.

The output is a file with the same name as your image, but with a `.gis` extension.

### 2) Render a `.gis` file back to an image

Run `graphIt.py`:

```bash
python graphIt.py
```

Then enter the path to a `.gis` file.

If the file is GIS v3, the program will:

- decode the binary file,
- rebuild the color-region contours,
- render the image pixel-perfectly,
- and save a rendered preview image next to the source file.

### 3) Use the geometry/text contour path

`graphIt.py` can also read contour text files in the older polygon format:

```text
polygon((x1, y1), (x2, y2), ...)
```

That mode is mainly for compatibility and experimentation. The main compression path is GIS v3.

## How It Works

### 1. Image preprocessing

`main.py` begins by loading the input image and converting it to RGB. It applies a bilateral filter to reduce noise while preserving edges. That helps the later contour extraction stage by making regions cleaner and more stable.

### 2. Posterization

`Polify/posterize.py` reduces the image to a smaller set of colors.

This matters because raw photographs usually contain far too many distinct colors. By collapsing the image into a compact palette, the project creates larger uniform regions that are easier to represent as shapes instead of pixels.

The current posterizer uses a fast quantization-first strategy and MiniBatch KMeans so that palette selection is much faster than training a full clustering model on every pixel.

### 3. Color segmentation

`Geometry/segment.py` groups the posterized image by unique colors.

Each unique color becomes a mask of the pixels that belong to that color. This produces the region map that contour extraction works on.

### 4. Contour extraction with hierarchy

`Geometry/contours.py` turns each color mask into contours.

The important part is that it uses CCOMP hierarchy, which preserves nested structure:

- outer boundaries,
- holes inside regions,
- and region nesting.

That hierarchy is what makes the reconstruction much more accurate than a simple outline-only approach.

### 5. GIS v3 encoding

`gis.py` encodes the image as a compact binary file.

The encoder reconstructs the posterized region image, builds a palette, stores a palette index map, and compresses the result with LZMA. The file format is designed to be simple enough to decode, but compact enough to be useful as an experimental compression format.

### 6. Decoding and rendering

`graphIt.py` and `gis.py` work together to decode the binary format and render it back.

For GIS v3, the renderer uses the stored contour hierarchy to paint each region back onto a canvas. That is why the project can verify its own output and compare the reconstructed image against the posterized source.

## File Roles

- `main.py`: End-to-end encoder, verifier, and benchmark script.
- `Polify/polify.py`: Small wrapper that calls the posterizer.
- `Polify/posterize.py`: Fast palette reduction / posterization.
- `Geometry/segment.py`: Splits the posterized image into masks by color.
- `Geometry/contours.py`: Extracts contours and contour hierarchy from masks.
- `gis.py`: Custom GIS v2/v3 encoder and decoder.
- `graphIt.py`: Renderer for `.gis` files and contour text files.

## Output

When you run the full pipeline, the project prints:

- contour counts,
- pixel-match verification metrics,
- PSNR,
- and file-size comparisons for PNG, JPEG, and GIS v3.

This makes it easy to judge both compression efficiency and reconstruction accuracy.

## Notes

- This project is experimental and focused on structure-aware compression rather than general-purpose image compression.
- It works best on images with clear regions, repeated colors, or strong boundaries.
- Highly detailed photographs will compress less cleanly because they contain many small color changes and complex boundaries.

## Example Workflow

```bash
python main.py
# enter an input image path

python graphIt.py
# enter the generated .gis file path
```
---
