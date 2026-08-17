"""Turn an arbitrary user-drawn image into something that looks like MNIST.

MNIST digits are not just "28x28 grayscale". They were produced by a specific
pipeline, and matching it matters far more than model architecture does:

  1. white ink on a black background
  2. the digit is cropped to its bounding box
  3. scaled (aspect ratio preserved) to fit inside a 20x20 box
  4. pasted into a 28x28 field so that the digit's CENTER OF MASS sits at the
     center of the image -- not the center of its bounding box

Skip step 4 and a model trained on MNIST will quietly lose several points of
accuracy on your own drawings.
"""

from io import BytesIO

import numpy as np
from PIL import Image

TARGET_SIZE = 28
DIGIT_BOX = 20
INK_THRESHOLD = 20  # 0-255; anything above this counts as ink


def _to_grayscale_ink_on_black(image: Image.Image) -> np.ndarray:
    """Return a float array where 0 = background and 255 = ink."""
    # Flatten transparency onto white, otherwise RGBA canvases become all-black.
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        backdrop = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(backdrop, image)

    arr = np.asarray(image.convert("L"), dtype=np.float32)

    # Auto-detect polarity from the border, which is nearly always background.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(border) > 127:  # light background -> dark ink, so invert
        arr = 255.0 - arr

    return arr


def _center_of_mass(arr: np.ndarray) -> tuple[float, float]:
    total = arr.sum()
    if total == 0:
        return (arr.shape[0] - 1) / 2, (arr.shape[1] - 1) / 2
    rows = np.arange(arr.shape[0])[:, None]
    cols = np.arange(arr.shape[1])[None, :]
    return float((arr * rows).sum() / total), float((arr * cols).sum() / total)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Bytes of any PIL-readable image -> (28, 28) float32 array in [0, 1].

    Raises ValueError if the image contains no ink at all.
    """
    image = Image.open(BytesIO(image_bytes))
    arr = _to_grayscale_ink_on_black(image)

    ys, xs = np.nonzero(arr > INK_THRESHOLD)
    if ys.size == 0:
        raise ValueError("The image is blank -- draw a digit first.")

    # 1. Crop to the ink's bounding box.
    cropped = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    # 2. Scale the long edge to 20px, preserving aspect ratio.
    h, w = cropped.shape
    scale = DIGIT_BOX / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(cropped.astype(np.uint8)).resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )

    # 3. Paste roughly centered, then shift so the center of mass is centered.
    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    top = (TARGET_SIZE - new_h) // 2
    left = (TARGET_SIZE - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = resized

    com_y, com_x = _center_of_mass(canvas)
    shift_y = int(round((TARGET_SIZE - 1) / 2 - com_y))
    shift_x = int(round((TARGET_SIZE - 1) / 2 - com_x))
    canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))

    # Rolling wraps around; blank out anything that wrapped past an edge.
    if shift_y > 0:
        canvas[:shift_y, :] = 0
    elif shift_y < 0:
        canvas[shift_y:, :] = 0
    if shift_x > 0:
        canvas[:, :shift_x] = 0
    elif shift_x < 0:
        canvas[:, shift_x:] = 0

    return np.clip(canvas / 255.0, 0.0, 1.0).astype(np.float32)


def to_png_bytes(arr: np.ndarray) -> bytes:
    """Render a (28, 28) [0,1] array as a PNG, for the 'what the model sees' panel."""
    buffer = BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()
