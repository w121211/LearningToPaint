import numpy as np
import cv2
from PIL import Image, ImageDraw


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def draw(f, width=128):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype("float32")
    tmp = 1.0 / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
        y = (int)((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
        z = (int)((1 - t) * z0 + t * z2)
        w = (1 - t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))
    # return cv2.resize(canvas, dsize=(width, width))


def draw_rect(f, width=128):
    x0, y0, x1, y1 = f
    x0 = normal(x0, width)
    y0 = normal(y0, width)
    x1 = normal(x1, width)
    y1 = normal(y1, width)
    im = Image.new("F", (width, width))
    draw = ImageDraw.Draw(im)
    draw.rectangle([x0, y0, x1, y1], fill=1)
    im = np.array(im)  # shape: (width, width)
    return im


def rand_draw(draw_fn=draw_rect, n_strokes=3, width=128):
    canvas = np.zeros((width, width, 3), dtype=int)
    for i in range(n_strokes):
        x = np.random.rand(4)
        stroke = draw_fn(x, width)  # (w, w)
        stroke = np.expand_dims(stroke, axis=2)  # (w, w, 1)
        color = np.random.randint(255, size=(3))  # (3)
        canvas = canvas * (1 - stroke) + stroke * color  # (w, h, 3)
    return canvas.astype(int)
