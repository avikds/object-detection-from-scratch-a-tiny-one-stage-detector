"""
Object Detection from Scratch: A Tiny One-Stage Detector

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - box_area
import numpy as np

def box_area(boxes):
    """
    Calculate the area of each axis-aligned bounding box.

    Args:
        boxes: Array-like of shape (N, 4) or a single box of shape (4,).
               Each box is represented as [x0, y0, x1, y1].

    Returns:
        A float64 NumPy array of shape (N,) containing the box areas.
    """
    # Convert the input into a float64 NumPy array.
    boxes = np.asarray(boxes, dtype=np.float64)

    # Handle a single box by adding a batch dimension.
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)

    # Ensure the input has the expected shape.
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4) or (4,)")

    # Calculate width and height, clamping negative dimensions to zero.
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)

    # Calculate and return the area of each bounding box.
    return widths * heights

# Step 2 - box_iou
def box_iou(a, b):
    """
    Calculate pairwise Intersection over Union (IoU) between two sets of boxes.

    Args:
        a: Array-like of shape (N, 4) or a single box of shape (4,).
        b: Array-like of shape (M, 4) or a single box of shape (4,).

    Returns:
        A float64 NumPy array of shape (N, M) containing pairwise IoU values.
    """
    # Convert inputs to float64 NumPy arrays.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Handle single boxes by adding a batch dimension.
    if a.ndim == 1:
        a = a.reshape(1, 4)

    if b.ndim == 1:
        b = b.reshape(1, 4)

    # Validate input shapes.
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("a must have shape (N, 4) or (4,)")

    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError("b must have shape (M, 4) or (4,)")

    # Calculate pairwise intersection coordinates.
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])

    # Calculate intersection dimensions, clamping negative values to zero.
    inter_width = np.maximum(x1 - x0, 0.0)
    inter_height = np.maximum(y1 - y0, 0.0)

    # Calculate intersection areas.
    intersection = inter_width * inter_height

    # Calculate individual box areas using the previously defined function.
    area_a = box_area(a)
    area_b = box_area(b)

    # Calculate union areas.
    union = area_a[:, None] + area_b[None, :] - intersection

    # Calculate IoU, returning zero wherever the union is zero.
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0
    )

    return iou

# Step 3 - shape_mask
CLASSES = ("circle", "square", "triangle")


def shape_mask(kind, cx, cy, r, size):
    """
    Rasterize a shape into a boolean mask.

    Args:
        kind: Shape type (0 = circle, 1 = square, 2 = triangle).
        cx: Center x-coordinate.
        cy: Center y-coordinate.
        r: Radius or half-size of the shape.
        size: Width and height of the output mask.

    Returns:
        Boolean NumPy array of shape (size, size).
    """
    # Validate the shape type.
    if kind not in (0, 1, 2):
        raise ValueError("kind must be 0 (circle), 1 (square), or 2 (triangle)")

    # Create pixel-center coordinates.
    # Column j has x-coordinate j + 0.5.
    # Row i has y-coordinate i + 0.5.
    x = np.arange(size, dtype=np.float64) + 0.5
    y = np.arange(size, dtype=np.float64) + 0.5

    # Generate a 2D coordinate grid.
    xx, yy = np.meshgrid(x, y)

    # Circle: (x - cx)^2 + (y - cy)^2 <= r^2
    if kind == 0:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    # Square: |x - cx| <= r and |y - cy| <= r
    elif kind == 1:
        mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)

    # Triangle: upward-pointing triangle.
    # Its apex is at (cx, cy - r), and its base is at y = cy + r.
    else:
        if r <= 0:
            return np.zeros((size, size), dtype=bool)

        # Normalized vertical position within the triangle.
        t = (yy - (cy - r)) / (2.0 * r)

        # A pixel belongs to the triangle when:
        # 0 <= t <= 1 and |x - cx| <= r * t.
        mask = (
            (t >= 0.0)
            & (t <= 1.0)
            & (np.abs(xx - cx) <= r * t)
        )

    return mask.astype(bool)

# Step 4 - mask_to_box
def mask_to_box(mask):
    """
    Convert a boolean mask into its tight bounding box.

    Args:
        mask: Boolean NumPy array of shape (H, W).

    Returns:
        A float64 NumPy array [x0, y0, x1, y1] representing
        the tight bounding box in pixel-edge coordinates.

    Raises:
        ValueError: If the mask contains no True pixels or is not 2D.
    """
    # Convert the input into a Boolean NumPy array.
    mask = np.asarray(mask, dtype=bool)

    # Ensure the mask is two-dimensional.
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    # Find the row and column indices of all True pixels.
    rows, cols = np.where(mask)

    # Raise an error if the mask contains no True pixels.
    if len(rows) == 0:
        raise ValueError("mask contains no True pixels")

    # Calculate the tight bounding box.
    # Add 1 to maximum coordinates because boxes use pixel-edge coordinates.
    x0 = cols.min()
    y0 = rows.min()
    x1 = cols.max() + 1
    y1 = rows.max() + 1

    # Return the bounding box as a float64 NumPy array.
    return np.array([x0, y0, x1, y1], dtype=np.float64)

# Step 5 - make_scene
def make_scene(rng, size=32, max_objects=4, r_range=(3.0, 8.0),
               contrast=(0.25, 0.7), noise=0.08):
    """
    Generate one synthetic grayscale scene with randomly placed objects.

    Args:
        rng: NumPy random Generator instance.
        size: Height and width of the square image.
        max_objects: Maximum number of objects to generate.
        r_range: Minimum and maximum object radius.
        contrast: Minimum and maximum object intensity contrast.
        noise: Standard deviation of background Gaussian noise.

    Returns:
        image: float32 array of shape (size, size).
        boxes: float64 array of shape (K, 4).
        labels: int64 array of shape (K,).
    """

    # --------------------------------------------------
    # 1. Generate the sloped background.
    # --------------------------------------------------

    # Draw background level first, followed by vertical and horizontal slopes.
    level = rng.uniform(0.1, 0.4)
    gy, gx = rng.uniform(-0.15, 0.15, 2)

    # Create normalized pixel-center coordinates.
    ys, xs = np.mgrid[0:size, 0:size] / size - 0.5

    # Initialize the image with a sloped background.
    image = level + gy * ys + gx * xs

    # --------------------------------------------------
    # 2. Determine the number of objects.
    # --------------------------------------------------

    n = int(rng.integers(1, max_objects + 1))

    boxes = []
    labels = []

    # Allow at most 40 placement attempts.
    for _ in range(40):

        # Stop once the requested number of objects is placed.
        if len(boxes) >= n:
            break

        # Draw the shape type: 0 = circle, 1 = square, 2 = triangle.
        kind = int(rng.integers(0, 3))

        # Draw the object's radius.
        r = rng.uniform(*r_range)

        # Draw the object's center coordinates.
        cx, cy = rng.uniform(r, size - r, 2)

        # Generate the object's Boolean mask.
        mask = shape_mask(kind, cx, cy, r, size)

        # Calculate the tight bounding box of the object.
        box = mask_to_box(mask)

        # Skip objects containing fewer than four pixels.
        if np.sum(mask) < 4:
            continue

        # --------------------------------------------------
        # Check overlap with previously placed objects.
        # --------------------------------------------------

        if len(boxes) > 0:

            # Expand each existing box by one pixel on every side.
            existing_boxes = np.asarray(boxes, dtype=np.float64)
            grown_boxes = existing_boxes + np.array(
                [-1.0, -1.0, 1.0, 1.0],
                dtype=np.float64
            )

            # Calculate IoU between the candidate and expanded boxes.
            overlaps = box_iou(
                box.reshape(1, 4),
                grown_boxes
            )

            # Reject the candidate if any IoU is greater than zero.
            if np.any(overlaps > 0):
                continue

        # --------------------------------------------------
        # Draw the object with random brightness or darkness.
        # --------------------------------------------------

        # Objects are bright with probability 0.7 and dark otherwise.
        sign = 1.0 if rng.random() < 0.7 else -1.0

        # Draw the object's contrast.
        intensity = rng.uniform(*contrast)

        # Apply the intensity change only inside the object's mask.
        image[mask] += sign * intensity

        # Store the object's bounding box and class label.
        boxes.append(box)
        labels.append(kind)

    # --------------------------------------------------
    # 3. Add Gaussian noise to the entire image.
    # --------------------------------------------------

    image += rng.normal(0.0, noise, (size, size))

    # Clip all pixel intensities to the valid [0, 1] range.
    image = np.clip(image, 0.0, 1.0)

    # --------------------------------------------------
    # 4. Convert outputs to the required data types.
    # --------------------------------------------------

    image = image.astype(np.float32)

    # Ensure boxes always have shape (K, 4), including when K = 0.
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    labels = np.asarray(labels, dtype=np.int64)

    return image, boxes, labels

# Step 6 - make_dataset
DATA_STREAM = 101


def make_dataset(n, seed, size=32, **scene_kw):
    """
    Generate a dataset of synthetic grayscale scenes.

    Args:
        n: Number of scenes to generate.
        seed: Random seed for reproducibility.
        size: Height and width of each image.
        **scene_kw: Additional keyword arguments passed to make_scene().

    Returns:
        images: Float32 NumPy array of shape (n, 1, size, size).
        boxes: List containing each scene's bounding-box array.
        labels: List containing each scene's class-label array.
    """

    # Create an independent random generator for dataset generation.
    rng = np.random.default_rng([DATA_STREAM, seed])

    # Allocate the image array.
    # The channel dimension is included for compatibility with convolutional networks.
    images = np.empty((n, 1, size, size), dtype=np.float32)

    # Store bounding boxes and labels separately for each scene.
    boxes = []
    labels = []

    # Generate n scenes using the same random generator.
    for i in range(n):

        # Generate one scene using the previously implemented function.
        image, scene_boxes, scene_labels = make_scene(
            rng,
            size=size,
            **scene_kw
        )

        # Store the image in the corresponding dataset position.
        images[i, 0] = image

        # Append the scene's bounding boxes and labels.
        boxes.append(scene_boxes)
        labels.append(scene_labels)

    return images, boxes, labels

