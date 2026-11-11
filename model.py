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

