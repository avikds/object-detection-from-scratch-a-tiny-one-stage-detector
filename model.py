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

