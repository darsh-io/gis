def polify(img, colors=16):
    from .posterize import posterize
    posterized = posterize(img, k=colors)
    return posterized
