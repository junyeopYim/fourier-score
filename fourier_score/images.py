import math
import numpy as np
from PIL import Image


def write_png(array, path):
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


def preview_grid(arrays, path, ncol=8):
    if not len(arrays):
        return
    n = len(arrays)
    h, w, c = arrays.shape[1:]
    ncol = min(ncol, n)
    canvas = np.zeros((math.ceil(n / ncol) * h, ncol * w, c), dtype=np.uint8)
    for i, a in enumerate(arrays):
        canvas[
            (i // ncol) * h : (i // ncol + 1) * h, (i % ncol) * w : (i % ncol + 1) * w
        ] = a
    write_png(canvas, path)
