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

# Step 7 - make_anchors
def make_anchors(grid, stride, sizes):
    """
    Generate square anchor boxes for a square feature map.

    Args:
        grid: Number of rows and columns in the feature map.
        stride: Distance between neighboring feature-map cell centers.
        sizes: Sequence of anchor side lengths.

    Returns:
        A float64 NumPy array of shape (grid * grid * len(sizes), 4).
        Each anchor is represented as [x0, y0, x1, y1].
    """

    # Convert anchor sizes to a float64 NumPy array.
    sizes = np.asarray(sizes, dtype=np.float64)

    # Generate feature-map row and column indices.
    rows, cols = np.meshgrid(
        np.arange(grid),
        np.arange(grid),
        indexing="ij"
    )

    # Calculate anchor center coordinates in image pixels.
    cx = (cols.ravel() + 0.5) * stride
    cy = (rows.ravel() + 0.5) * stride

    # Combine the center coordinates into (grid * grid, 2).
    centers = np.stack((cx, cy), axis=1)

    # Calculate half the side length of each anchor.
    half_sizes = sizes / 2.0

    # Calculate top-left corners for every center and anchor size.
    top_left = (
        centers[:, None, :]
        - half_sizes[None, :, None]
    )

    # Calculate bottom-right corners for every center and anchor size.
    bottom_right = (
        centers[:, None, :]
        + half_sizes[None, :, None]
    )

    # Combine coordinates into [x0, y0, x1, y1].
    anchors = np.concatenate(
        (top_left, bottom_right),
        axis=2
    )

    # Flatten while preserving row-major cell order and sizes innermost.
    return anchors.reshape(-1, 4).astype(np.float64)

# Step 8 - encode_boxes
def encode_boxes(gt, anchors):
    """
    Encode ground-truth boxes relative to their corresponding anchors.

    Args:
        gt: Ground-truth boxes of shape (N, 4).
        anchors: Anchor boxes of shape (N, 4).

    Returns:
        Float64 NumPy array of shape (N, 4), containing
        (tx, ty, tw, th) regression targets.
    """

    # Convert inputs to float64 NumPy arrays.
    gt = np.asarray(gt, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)

    # Validate input shapes.
    if gt.ndim != 2 or gt.shape[1] != 4:
        raise ValueError("gt must have shape (N, 4)")

    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError("anchors must have shape (N, 4)")

    if gt.shape != anchors.shape:
        raise ValueError("gt and anchors must have the same shape")

    # Calculate ground-truth box widths and heights.
    gt_w = gt[:, 2] - gt[:, 0]
    gt_h = gt[:, 3] - gt[:, 1]

    # Calculate anchor widths and heights.
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]

    # Calculate ground-truth box centers.
    gt_cx = (gt[:, 0] + gt[:, 2]) / 2.0
    gt_cy = (gt[:, 1] + gt[:, 3]) / 2.0

    # Calculate anchor box centers.
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0

    # Encode center offsets relative to anchor dimensions.
    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h

    # Encode width and height ratios logarithmically.
    tw = np.log(gt_w / anchor_w)
    th = np.log(gt_h / anchor_h)

    # Combine the four regression targets.
    return np.column_stack((tx, ty, tw, th)).astype(np.float64)

# Step 9 - decode_boxes
MAX_LOG_SCALE = float(np.log(1000.0 / 16))


def decode_boxes(deltas, anchors):
    """
    Decode bounding-box regression deltas relative to anchor boxes.

    Args:
        deltas: Float array of shape (N, 4), containing
                (tx, ty, tw, th).
        anchors: Float array of shape (N, 4), containing
                 [x0, y0, x1, y1].

    Returns:
        Float64 NumPy array of shape (N, 4), containing
        decoded bounding boxes [x0, y0, x1, y1].
    """

    # Convert inputs to float64 NumPy arrays.
    deltas = np.asarray(deltas, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)

    # Validate input shapes.
    if deltas.ndim != 2 or deltas.shape[1] != 4:
        raise ValueError("deltas must have shape (N, 4)")

    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError("anchors must have shape (N, 4)")

    if deltas.shape != anchors.shape:
        raise ValueError("deltas and anchors must have the same shape")

    # Extract anchor widths and heights.
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]

    # Calculate anchor centers.
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0

    # Extract predicted regression deltas.
    tx = deltas[:, 0]
    ty = deltas[:, 1]

    # Clamp logarithmic scale deltas to prevent exponential overflow.
    tw = np.minimum(deltas[:, 2], MAX_LOG_SCALE)
    th = np.minimum(deltas[:, 3], MAX_LOG_SCALE)

    # Decode predicted box centers.
    cx = anchor_cx + tx * anchor_w
    cy = anchor_cy + ty * anchor_h

    # Decode predicted box widths and heights.
    w = anchor_w * np.exp(tw)
    h = anchor_h * np.exp(th)

    # Convert decoded centers and dimensions back to corner coordinates.
    x0 = cx - w / 2.0
    y0 = cy - h / 2.0
    x1 = cx + w / 2.0
    y1 = cy + h / 2.0

    # Return decoded boxes as a float64 NumPy array.
    return np.column_stack((x0, y0, x1, y1)).astype(np.float64)

# Step 10 - assign_targets
def assign_targets(anchors, gt_boxes, gt_labels, num_classes,
                   pos_iou=0.5, neg_iou=0.4, force_best=True):
    """
    Assign classification and box regression targets to anchors.

    Args:
        anchors: Array of shape (A, 4) containing anchor boxes.
        gt_boxes: Array of shape (M, 4) containing ground-truth boxes.
        gt_labels: Array of shape (M,) containing ground-truth class labels.
        num_classes: Number of object classes.
        pos_iou: IoU threshold for positive anchors.
        neg_iou: IoU threshold for negative anchors.
        force_best: Whether to force the best anchor for each ground-truth box.

    Returns:
        cls_t: Float32 array of shape (A, num_classes), one-hot class targets.
        box_t: Float32 array of shape (A, 4), encoded box regression targets.
        pos: Boolean array of shape (A,), indicating positive anchors.
        valid: Boolean array of shape (A,), indicating non-ignored anchors.
    """

    # Convert inputs to NumPy arrays.
    anchors = np.asarray(anchors, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    # Determine the number of anchors and ground-truth boxes.
    A = len(anchors)
    M = len(gt_boxes)

    # Initialize classification and box regression targets.
    cls_t = np.zeros((A, num_classes), dtype=np.float32)
    box_t = np.zeros((A, 4), dtype=np.float32)

    # Initialize positive and valid anchor masks.
    pos = np.zeros(A, dtype=bool)
    valid = np.ones(A, dtype=bool)

    # If there are no ground-truth objects, all anchors are negatives.
    if M == 0:
        return cls_t, box_t, pos, valid

    # Calculate the pairwise IoU matrix of shape (A, M).
    ious = box_iou(anchors, gt_boxes)

    # Match every anchor to its ground-truth box with the highest IoU.
    # np.argmax returns the first index when multiple values are tied.
    matched_gt = np.argmax(ious, axis=1)

    # Find the best IoU for every anchor.
    best_iou = np.max(ious, axis=1)

    # Assign positive anchors.
    pos = best_iou >= pos_iou

    # Assign negative anchors.
    neg = best_iou < neg_iou

    # Anchors between the thresholds are ignored.
    valid = pos | neg

    # --------------------------------------------------
    # Force the best-matching anchor for each ground-truth box.
    # --------------------------------------------------
    if force_best:
        for m in range(M):

            # Find the anchor with the highest IoU for ground-truth box m.
            # np.argmax selects the first anchor in case of a tie.
            a = np.argmax(ious[:, m])

            # Only force a match when the best IoU is greater than zero.
            if ious[a, m] > 0:

                # Mark the selected anchor as positive and valid.
                pos[a] = True
                valid[a] = True

                # Assign this anchor to the current ground-truth box.
                matched_gt[a] = m

    # --------------------------------------------------
    # Generate classification and box regression targets.
    # --------------------------------------------------

    # Get the indices of all positive anchors.
    pos_idx = np.flatnonzero(pos)

    if len(pos_idx) > 0:

        # Retrieve the ground-truth box matched to each positive anchor.
        matched_boxes = gt_boxes[matched_gt[pos_idx]]

        # Retrieve the corresponding class labels.
        matched_labels = gt_labels[matched_gt[pos_idx]]

        # Assign one-hot classification targets.
        cls_t[pos_idx, matched_labels] = 1.0

        # Encode ground-truth boxes relative to their matched anchors.
        box_t[pos_idx] = encode_boxes(
            matched_boxes,
            anchors[pos_idx]
        ).astype(np.float32)

    return cls_t, box_t, pos, valid

# Step 11 - build_targets
def build_targets(anchors, boxes_list, labels_list, num_classes, **assign_kw):
    """
    Build classification and bounding-box targets for an entire dataset.

    Args:
        anchors: Array of shape (A, 4) containing anchor boxes.
        boxes_list: List of ground-truth box arrays, one per image.
        labels_list: List of ground-truth label arrays, one per image.
        num_classes: Number of object classes.
        **assign_kw: Additional arguments passed to assign_targets().

    Returns:
        Dictionary containing:
            cls: Float32 array of shape (N, A, C).
            box: Float32 array of shape (N, A, 4).
            pos: Float32 array of shape (N, A).
            valid: Float32 array of shape (N, A).
    """

    # Convert anchors to a NumPy array.
    anchors = np.asarray(anchors, dtype=np.float64)

    # Validate the anchor array.
    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError("anchors must have shape (A, 4)")

    # Ensure boxes and labels contain the same number of images.
    if len(boxes_list) != len(labels_list):
        raise ValueError("boxes_list and labels_list must have equal lengths")

    # Determine the dataset size and number of anchors.
    N = len(boxes_list)
    A = len(anchors)

    # Initialize lists to collect targets for each image.
    cls_targets = []
    box_targets = []
    pos_targets = []
    valid_targets = []

    # Process each image in the original dataset order.
    for gt_boxes, gt_labels in zip(boxes_list, labels_list):

        # Assign targets to all anchors for the current image.
        cls_t, box_t, pos, valid = assign_targets(
            anchors,
            gt_boxes,
            gt_labels,
            num_classes,
            **assign_kw
        )

        # Store classification targets.
        cls_targets.append(cls_t)

        # Store bounding-box regression targets.
        box_targets.append(box_t)

        # Convert positive-anchor indicators to float32.
        pos_targets.append(pos.astype(np.float32))

        # Convert valid-anchor indicators to float32.
        valid_targets.append(valid.astype(np.float32))

    # Stack the per-image targets into batch arrays.
    # Handle an empty dataset while preserving the required dimensions.
    if N == 0:
        return {
            "cls": np.empty((0, A, num_classes), dtype=np.float32),
            "box": np.empty((0, A, 4), dtype=np.float32),
            "pos": np.empty((0, A), dtype=np.float32),
            "valid": np.empty((0, A), dtype=np.float32),
        }

    return {
        "cls": np.stack(cls_targets, axis=0).astype(np.float32),
        "box": np.stack(box_targets, axis=0).astype(np.float32),
        "pos": np.stack(pos_targets, axis=0).astype(np.float32),
        "valid": np.stack(valid_targets, axis=0).astype(np.float32),
    }

# Step 12 - Conv
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c_in, c_out, rng, stride=1, std=None, bias=0.0):
        super().__init__()

        # Use He initialization by default.
        if std is None:
            std = np.sqrt(2.0 / (9.0 * c_in))

        # Generate weights using the supplied NumPy random generator.
        # Shape: (c_out, c_in, 3, 3)
        weight = rng.normal(
            0.0,
            std,
            (c_out, c_in, 3, 3)
        ).astype(np.float32)

        # Register weights as a trainable parameter BEFORE the bias.
        self.w = nn.Parameter(torch.from_numpy(weight))

        # Initialize the bias as a trainable float32 parameter.
        self.b = nn.Parameter(
            torch.full(
                (c_out,),
                float(bias),
                dtype=torch.float32
            )
        )

        # Store the convolution stride.
        self.stride = stride

    def forward(self, x):
        # Apply 3×3 convolution with padding of 1.
        return F.conv2d(
            x,
            self.w,
            self.b,
            stride=self.stride,
            padding=1
        )

# Step 13 - prior_bias
import math


def prior_bias(pi):
    """
    Calculate the logit bias corresponding to a prior probability.

    Args:
        pi: Prior probability, which must satisfy 0 < pi < 1.

    Returns:
        The corresponding logit bias as a float.

    Raises:
        ValueError: If pi is not strictly between 0 and 1.
    """

    # Validate the prior probability.
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must satisfy 0 < pi < 1")

    # Calculate and return the logit bias.
    return float(-math.log((1.0 - pi) / pi))

# Step 14 - flatten_head
def flatten_head(t, k):
    """
    Reshape a detection head output into one row per anchor.

    Args:
        t: PyTorch tensor of shape (N, A*k, G, G).
        k: Number of output values per anchor.

    Returns:
        PyTorch tensor of shape (N, G*G*A, k).
    """

    # Get the input tensor dimensions.
    N, C, G, _ = t.shape

    # Calculate the number of anchors per feature-map cell.
    A = C // k

    # Separate the anchor and output dimensions.
    # Shape: (N, A, k, G, G)
    t = t.reshape(N, A, k, G, G)

    # Rearrange dimensions to:
    # (N, G, G, A, k)
    # This preserves row-major feature-map and anchor ordering.
    t = t.permute(0, 3, 4, 1, 2)

    # Flatten the spatial and anchor dimensions.
    # Final shape: (N, G*G*A, k)
    return t.reshape(N, G * G * A, k)

# Step 15 - TinyDetector
INIT_STREAM = 202


class TinyDetector(nn.Module):
    def __init__(self, num_classes, num_anchors, seed=0, width=32, pi=0.01):
        super().__init__()

        # Create an independent random generator for model initialization.
        rng = np.random.default_rng([INIT_STREAM, seed])

        # Store the number of object classes and anchors per feature-map cell.
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # --------------------------------------------------
        # 1. Build the five-layer convolutional backbone.
        # --------------------------------------------------

        self.backbone = nn.ModuleList([
            Conv(1, width // 2, rng),
            Conv(width // 2, width, rng, stride=2),
            Conv(width, width, rng),
            Conv(width, width, rng, stride=2),
            Conv(width, width, rng)
        ])

        # --------------------------------------------------
        # 2. Classification head.
        # --------------------------------------------------

        # Predict num_classes classification logits for each anchor.
        # Initialize the classification bias using the prior probability.
        self.cls_head = Conv(
            width,
            num_anchors * num_classes,
            rng,
            std=0.01,
            bias=prior_bias(pi)
        )

        # --------------------------------------------------
        # 3. Bounding-box regression head.
        # --------------------------------------------------

        # Predict four bounding-box regression values per anchor.
        self.box_head = Conv(
            width,
            num_anchors * 4,
            rng,
            std=0.01
        )

    def forward(self, x):
        """
        Forward pass through the detector.

        Args:
            x: Input images of shape (N, 1, H, W).

        Returns:
            cls_logits: Classification logits of shape (N, G*G*A, C).
            box_deltas: Bounding-box deltas of shape (N, G*G*A, 4).
        """

        # Apply each backbone convolution followed by ReLU activation.
        for layer in self.backbone:
            x = torch.relu(layer(x))

        # Generate classification logits and flatten them into anchor order.
        cls_logits = flatten_head(
            self.cls_head(x),
            self.num_classes
        )

        # Generate bounding-box regression predictions.
        box_deltas = flatten_head(
            self.box_head(x),
            4
        )

        return cls_logits, box_deltas

# Step 16 - focal_loss
def focal_loss(logits, targets, valid, gamma=2.0, alpha=0.25):
    """
    Compute the summed sigmoid focal loss.

    Args:
        logits: Raw classification logits of shape (N, A, C).
        targets: Binary classification targets of shape (N, A, C).
        valid: Valid-anchor mask of shape (N, A).
        gamma: Focal loss modulation parameter.
        alpha: Positive/negative class balancing parameter.
               A negative value disables alpha weighting.

    Returns:
        Scalar PyTorch tensor containing the summed focal loss.
    """

    # Ensure targets have the same data type as the logits.
    targets = targets.to(dtype=logits.dtype)

    # Calculate sigmoid probabilities.
    p = torch.sigmoid(logits)

    # Calculate sigmoid cross-entropy in a numerically stable form:
    # CE(x, y) = max(x, 0) - x*y + log(1 + exp(-|x|))
    ce = (
        torch.clamp_min(logits, 0.0)
        - logits * targets
        + torch.log1p(torch.exp(-torch.abs(logits)))
    )

    # Calculate the probability assigned to the correct target.
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    # Apply focal modulation when gamma is nonzero.
    if gamma != 0:
        ce = ce * (1.0 - p_t) ** gamma

    # Apply alpha balancing when alpha is nonnegative.
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        ce = ce * alpha_t

    # Mask out ignored anchors.
    # Expand valid from (N, A) to (N, A, 1) for broadcasting.
    valid_mask = valid.to(dtype=logits.dtype).unsqueeze(-1)

    loss = ce * valid_mask

    # Return the summed loss without normalization.
    return loss.sum()

# Step 17 - smooth_l1
def smooth_l1(pred, target, pos, beta=0.1):
    """
    Compute the summed Smooth L1 loss over positive anchors.

    Args:
        pred: Predicted box deltas of shape (N, A, 4).
        target: Target box deltas of shape (N, A, 4).
        pos: Positive-anchor mask of shape (N, A).
        beta: Threshold separating quadratic and linear regions.

    Returns:
        Scalar PyTorch tensor containing the summed Smooth L1 loss.
    """

    # Calculate the absolute difference between predictions and targets.
    d = torch.abs(pred - target)

    # Apply the piecewise Smooth L1 loss.
    # Quadratic region: 0.5 * d^2 / beta
    # Linear region:    d - 0.5 * beta
    loss = torch.where(
        d < beta,
        0.5 * d.pow(2) / beta,
        d - 0.5 * beta
    )

    # Sum the loss across the four bounding-box coordinates.
    loss = loss.sum(dim=-1)

    # Convert the positive-anchor mask to match the prediction tensor.
    pos_mask = pos.to(device=pred.device, dtype=pred.dtype)

    # Keep losses only for positive anchors.
    loss = loss * pos_mask

    # Return the summed loss without normalization.
    return loss.sum()

# Step 18 - detection_loss
def detection_loss(cls_logits, box_deltas, cls_t, box_t, pos, valid,
                   gamma=2.0, alpha=0.25, box_weight=1.0):
    """
    Combine classification and bounding-box regression losses.

    Args:
        cls_logits: Classification logits of shape (N, A, C).
        box_deltas: Predicted box deltas of shape (N, A, 4).
        cls_t: Classification targets of shape (N, A, C).
        box_t: Bounding-box regression targets of shape (N, A, 4).
        pos: Positive-anchor mask of shape (N, A).
        valid: Valid-anchor mask of shape (N, A).
        gamma: Focal loss gamma parameter.
        alpha: Focal loss alpha parameter.
        box_weight: Weight applied to the box regression loss.

    Returns:
        A tuple containing:
            total: Combined detection loss.
            cls: Normalized classification loss.
            box: Normalized bounding-box regression loss.
    """

    # Count positive anchors across the entire minibatch.
    # Clamp the count to at least 1 to avoid division by zero.
    n_pos = torch.clamp(pos.sum(), min=1.0)

    # Calculate the summed classification focal loss.
    cls = focal_loss(
        cls_logits,
        cls_t,
        valid,
        gamma=gamma,
        alpha=alpha
    )

    # Normalize classification loss by the number of positive anchors.
    cls = cls / n_pos

    # Calculate the summed Smooth L1 bounding-box regression loss.
    box = smooth_l1(
        box_deltas,
        box_t,
        pos
    )

    # Normalize bounding-box regression loss by the number of positive anchors.
    box = box / n_pos

    # Combine both normalized losses.
    total = cls + box_weight * box

    return total, cls, box

# Step 19 - Trainer
TRAIN_STREAM = 303


class Trainer:
    def __init__(self, model, lr=3e-3, gamma=2.0, alpha=0.25,
                 box_weight=1.0, b1=0.9, b2=0.999, eps=1e-8):

        # Store model and training hyperparameters.
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.box_weight = box_weight

        # Initialize the Adam optimizer.
        self.opt = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(b1, b2),
            eps=eps
        )

    def fit(self, images, targets, steps, batch=32, seed=0, cosine=False):
        """
        Train the detector using Adam and randomly sampled minibatches.

        Returns:
            Float64 NumPy array of shape (steps, 3), containing
            total, classification, and box losses.
        """

        # Create an independent random generator for minibatch ordering.
        rng = np.random.default_rng([TRAIN_STREAM, seed])

        # Store the training loss history.
        history = []

        # Set the model to training mode.
        self.model.train()

        for i in range(steps):

            # Sample minibatch indices without replacement.
            idx = rng.choice(
                len(images),
                batch,
                replace=False
            )

            # Set the learning rate.
            if cosine:
                current_lr = self.lr * (
                    0.5 + 0.5 * math.cos(math.pi * i / steps)
                )
            else:
                current_lr = self.lr

            # Update the learning rate for every parameter group.
            for param_group in self.opt.param_groups:
                param_group["lr"] = current_lr

            # Convert minibatch images to PyTorch tensors.
            x = torch.as_tensor(
                images[idx],
                dtype=torch.float32
            )

            # Convert minibatch targets to PyTorch tensors.
            cls_t = torch.as_tensor(
                targets["cls"][idx],
                dtype=torch.float32
            )

            box_t = torch.as_tensor(
                targets["box"][idx],
                dtype=torch.float32
            )

            pos = torch.as_tensor(
                targets["pos"][idx],
                dtype=torch.float32
            )

            valid = torch.as_tensor(
                targets["valid"][idx],
                dtype=torch.float32
            )

            # Clear gradients from the previous iteration.
            self.opt.zero_grad()

            # Forward pass.
            cls_logits, box_deltas = self.model(x)

            # Calculate the combined detection loss.
            total, cls, box = detection_loss(
                cls_logits,
                box_deltas,
                cls_t,
                box_t,
                pos,
                valid,
                gamma=self.gamma,
                alpha=self.alpha,
                box_weight=self.box_weight
            )

            # Record losses before updating model parameters.
            history.append([
                total.item(),
                cls.item(),
                box.item()
            ])

            # Backpropagation.
            total.backward()

            # Update model parameters using Adam.
            self.opt.step()

        # Return loss history as float64 to preserve clean rounded values.
        return np.asarray(history, dtype=np.float64).reshape(steps, 3)

# Step 20 - predict_raw
def predict_raw(model, images, batch=32):
    """
    Run the detector over a NumPy image array in batches.

    Args:
        model: Trained TinyDetector model.
        images: NumPy array of shape (N, 1, H, W).
        batch: Number of images processed per batch.

    Returns:
        scores: Float32 NumPy array of shape (N, A, C),
                containing sigmoid classification probabilities.
        deltas: Float32 NumPy array of shape (N, A, 4),
                containing predicted bounding-box deltas.
    """

    # Validate the batch size.
    if batch <= 0:
        raise ValueError("batch must be a positive integer")

    # Convert images to a NumPy array if necessary.
    images = np.asarray(images)

    # Determine the model's device.
    device = next(model.parameters()).device

    # Store the original training/evaluation state.
    was_training = model.training

    # Switch the model to evaluation mode.
    model.eval()

    scores_list = []
    deltas_list = []

    try:
        # Disable gradient calculations during inference.
        with torch.no_grad():

            # Process images in consecutive batches.
            for start in range(0, len(images), batch):

                # Extract the current batch.
                batch_images = torch.as_tensor(
                    images[start:start + batch],
                    dtype=torch.float32,
                    device=device
                )

                # Forward pass through the detector.
                cls_logits, box_deltas = model(batch_images)

                # Convert classification logits into probabilities.
                probabilities = torch.sigmoid(cls_logits)

                # Convert outputs into NumPy arrays.
                scores_list.append(
                    probabilities.detach().cpu().numpy()
                )

                deltas_list.append(
                    box_deltas.detach().cpu().numpy()
                )

    finally:
        # Restore the model's original training/evaluation state.
        model.train(was_training)

    # Concatenate batch outputs in their original image order.
    scores = np.concatenate(scores_list, axis=0)
    deltas = np.concatenate(deltas_list, axis=0)

    return scores, deltas

# Step 21 - nms
def nms(boxes, scores, iou_thresh=0.5):
    """
    Perform greedy Non-Maximum Suppression.

    Args:
        boxes: NumPy array of shape (K, 4), containing bounding boxes.
        scores: NumPy array of shape (K,), containing confidence scores.
        iou_thresh: IoU threshold for suppressing overlapping boxes.

    Returns:
        An int64 NumPy array containing the indices of kept boxes,
        ordered by descending confidence score.
    """

    # Convert inputs to NumPy arrays.
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores)

    # Validate input shapes.
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (K, 4)")

    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape (K,) matching boxes")

    # Return an empty int64 array when there are no boxes.
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    # Sort boxes by descending confidence score.
    # Stable sorting ensures lower original indices come first when scores tie.
    order = np.argsort(-scores, kind="stable")

    # Store the indices of boxes that survive suppression.
    keep = []

    # Greedy Non-Maximum Suppression.
    while len(order) > 0:

        # Select the highest-scoring remaining box.
        current = order[0]

        # Keep the selected box.
        keep.append(current)

        # Get the indices of all remaining boxes.
        remaining = order[1:]

        # Stop if no boxes remain.
        if len(remaining) == 0:
            break

        # Calculate IoU between the selected box and remaining boxes.
        ious = box_iou(
            boxes[current:current + 1],
            boxes[remaining]
        )[0]

        # Suppress boxes whose IoU is strictly greater than the threshold.
        # Boxes with IoU equal to the threshold are retained.
        order = remaining[ious <= iou_thresh]

    # Return the kept indices as an int64 NumPy array.
    return np.asarray(keep, dtype=np.int64)

# Step 22 - postprocess
def postprocess(scores, deltas, anchors, size, score_thresh=0.05,
                iou_thresh=0.5, top_k=100, max_det=20, use_nms=True):
    """
    Convert raw predictions into final object detections.

    Args:
        scores: Classification probabilities of shape (A, C).
        deltas: Bounding-box regression deltas of shape (A, 4).
        anchors: Anchor boxes of shape (A, 4).
        size: Image height and width (assumed square).
        score_thresh: Minimum confidence score (strictly greater).
        iou_thresh: IoU threshold used by NMS.
        top_k: Maximum number of candidates considered.
        max_det: Maximum number of final detections.
        use_nms: Whether to apply per-class NMS.

    Returns:
        boxes: Float64 array of shape (K, 4).
        det_scores: Array of shape (K,) containing detection scores.
        labels: Int64 array of shape (K,) containing class indices.
    """

    # Convert inputs to NumPy arrays.
    scores = np.asarray(scores)
    deltas = np.asarray(deltas, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)

    # Validate input shapes.
    if scores.ndim != 2:
        raise ValueError("scores must have shape (A, C)")

    if deltas.ndim != 2 or deltas.shape[1] != 4:
        raise ValueError("deltas must have shape (A, 4)")

    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError("anchors must have shape (A, 4)")

    if scores.shape[0] != deltas.shape[0] or scores.shape[0] != anchors.shape[0]:
        raise ValueError("scores, deltas, and anchors must have the same number of anchors")

    if top_k < 0 or max_det < 0:
        raise ValueError("top_k and max_det must be non-negative")

    # Helper for returning correctly shaped empty outputs.
    def empty_result():
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty((0,), dtype=scores.dtype),
            np.empty((0,), dtype=np.int64)
        )

    # Return immediately when no detections can be produced.
    if top_k == 0 or max_det == 0:
        return empty_result()

    # --------------------------------------------------
    # 1. Find anchor-class pairs above the score threshold.
    # --------------------------------------------------

    # np.argwhere produces anchor-then-class ordering.
    candidates = np.argwhere(scores > score_thresh)

    if len(candidates) == 0:
        return empty_result()

    anchor_idx = candidates[:, 0]
    candidate_labels = candidates[:, 1]

    # Extract confidence scores for all candidates.
    candidate_scores = scores[anchor_idx, candidate_labels]

    # --------------------------------------------------
    # 2. Keep the top-k highest-scoring candidates.
    # --------------------------------------------------

    # Stable sorting preserves anchor-then-class ordering for tied scores.
    order = np.argsort(-candidate_scores, kind="stable")[:top_k]

    anchor_idx = anchor_idx[order]
    candidate_labels = candidate_labels[order]
    candidate_scores = candidate_scores[order]

    # --------------------------------------------------
    # 3. Decode predicted boxes relative to their anchors.
    # --------------------------------------------------

    decoded_boxes = decode_boxes(
        deltas[anchor_idx],
        anchors[anchor_idx]
    )

    # --------------------------------------------------
    # 4. Clip bounding-box coordinates to image boundaries.
    # --------------------------------------------------

    decoded_boxes = np.clip(
        decoded_boxes,
        0.0,
        float(size)
    ).astype(np.float64)

    # --------------------------------------------------
    # 5. Apply per-class Non-Maximum Suppression.
    # --------------------------------------------------

    # Maintain candidate positions rather than appending class-wise results.
    # This preserves the original global candidate order across classes.
    keep_mask = np.zeros(len(candidate_scores), dtype=bool)

    if use_nms:

        # Process classes in increasing class-index order.
        for cls in range(scores.shape[1]):

            # Find candidates belonging to the current class.
            class_positions = np.flatnonzero(candidate_labels == cls)

            if len(class_positions) == 0:
                continue

            # Run NMS independently for this class.
            class_keep = nms(
                decoded_boxes[class_positions],
                candidate_scores[class_positions],
                iou_thresh
            )

            # Mark surviving candidates.
            keep_mask[class_positions[class_keep]] = True

    else:
        # Keep every candidate when NMS is disabled.
        keep_mask[:] = True

    # --------------------------------------------------
    # 6. Sort surviving detections by descending score.
    # --------------------------------------------------

    # Survivors retain their top-k ranking order.
    # Stable sorting ensures earlier candidates win score ties.
    survivors = np.flatnonzero(keep_mask)

    if len(survivors) == 0:
        return empty_result()

    final_order = np.argsort(
        -candidate_scores[survivors],
        kind="stable"
    )

    # --------------------------------------------------
    # 7. Limit the number of final detections.
    # --------------------------------------------------

    final_idx = survivors[final_order[:max_det]]

    # Return the final detections.
    boxes = decoded_boxes[final_idx].astype(np.float64)
    det_scores = candidate_scores[final_idx].copy()
    labels = candidate_labels[final_idx].astype(np.int64)

    return boxes, det_scores, labels

# Step 23 - detect
def detect(model, images, anchors, size=32, **post_kw):
    """
    Run the complete object detection inference pipeline.

    Args:
        model: Trained TinyDetector model.
        images: NumPy image array of shape (N, 1, H, W).
        anchors: Anchor boxes of shape (A, 4).
        size: Image size used for clipping predicted boxes.
        **post_kw: Additional arguments passed to postprocess().

    Returns:
        A list of (boxes, scores, labels) tuples, one per image.
    """

    # Run the model once over all images using batched inference.
    scores, deltas = predict_raw(model, images)

    # Store the final detections for each image.
    detections = []

    # Apply postprocessing independently to every image.
    for i in range(len(images)):

        result = postprocess(
            scores[i],
            deltas[i],
            anchors,
            size,
            **post_kw
        )

        detections.append(result)

    # Return detections in the original image order.
    return detections

# Step 24 - match_detections
def match_detections(det_boxes, gt_boxes, iou_thresh):
    """
    Greedily match detections to ground-truth boxes.

    Args:
        det_boxes: Detection boxes of shape (D, 4), sorted by
                   decreasing confidence score.
        gt_boxes: Ground-truth boxes of shape (G, 4).
        iou_thresh: Minimum IoU required for a true positive.

    Returns:
        Boolean NumPy array of shape (D,).
        True indicates a true positive; False indicates a false positive.
    """

    # Convert inputs to float64 NumPy arrays.
    det_boxes = np.asarray(det_boxes, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)

    # Validate input shapes.
    if det_boxes.ndim != 2 or det_boxes.shape[1] != 4:
        raise ValueError("det_boxes must have shape (D, 4)")

    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError("gt_boxes must have shape (G, 4)")

    # Initialize all detections as false positives.
    D = len(det_boxes)
    G = len(gt_boxes)

    tp = np.zeros(D, dtype=bool)

    # Return all False when there are no detections or ground-truth boxes.
    if D == 0 or G == 0:
        return tp

    # Track which ground-truth boxes have already been matched.
    matched = np.zeros(G, dtype=bool)

    # Process detections in their given confidence-score order.
    for i in range(D):

        # Find the indices of unmatched ground-truth boxes.
        unmatched = np.flatnonzero(~matched)

        # Stop if every ground-truth box has already been matched.
        if len(unmatched) == 0:
            break

        # Calculate IoU between the current detection and unmatched boxes.
        ious = box_iou(
            det_boxes[i:i + 1],
            gt_boxes[unmatched]
        )[0]

        # Find the best unmatched ground-truth match.
        # np.argmax selects the first index when IoU values are tied.
        best_idx = np.argmax(ious)
        best_iou = ious[best_idx]

        # Mark as a true positive if the IoU meets the threshold.
        if best_iou >= iou_thresh:

            # Mark the detection as a true positive.
            tp[i] = True

            # Mark the matched ground-truth box as unavailable.
            matched[unmatched[best_idx]] = True

    return tp

