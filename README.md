# Object Detection from Scratch: A Tiny One-Stage Detector

Build a complete one-stage object detector from nothing but numpy and PyTorch tensors, train it on a CPU in under a minute, and score it with the same mean average precision that the COCO benchmark uses. The data comes from a seeded scene generator you write first: 32 by 32 grayscale images holding one to four circles, squares and triangles of 5 to 16 pixels, on a sloped, noisy background, with bright and dark objects, so every box label is exact and a fresh dataset costs a second. From there you build the pieces of a RetinaNet-style detector one function at a time: box area and intersection over union, a grid of 128 anchors, the delta encoding of boxes relative to anchors and its inverse, IoU-based target assignment with positive, negative and ignored anchors and a forced best match for small objects, a five-layer convolutional backbone with a classification head and a box head, the prior-probability bias initialization, the focal loss and a smooth L1 box loss normalized by the number of positives, an Adam training loop, decoding with score threshold, top-k and per-class non-maximum suppression, greedy matching, 101-point interpolated average precision, mAP over ten IoU thresholds from 0.5 to 0.95, and a paired ablation harness. Data, weight initialization and minibatch order each come from their own random stream, so an ablation changes exactly one thing. What the full sweep found (eight seeds per setting, 1,000 training scenes, 400 steps of 32 images, 256 validation scenes, on the prod image). With Adam at a constant 0.003, the RetinaNet recipe (focal loss with gamma 2 and alpha 0.25, prior 0.01) reaches mAP@[.5:.95] 0.586 on average (0.511 to 0.646 across seeds), mAP@.5 0.87 and mAP@.75 0.71, and beats plain binary cross-entropy (0.496) on every seed, by 0.002 to 0.215; cross-entropy finds the objects as often (mAP@.5 0.94) but localizes them worse (mAP@.75 0.46). That ranking is not robust: under a cosine schedule from 0.006 cross-entropy reaches 0.639, the best setting measured, and the focal loss falls to 0.459, again on every seed. What did hold everywhere is the RetinaNet paper's other claim: without the prior initialization (prior 0.5), cross-entropy loses 0.15 to 0.24 mAP on average and on every seed, while the focal loss loses 0.03 or less, and not on every seed. Non-maximum suppression is worth 0.10 to 0.21 mAP; in a smaller three-seed sweep the forced best-anchor match and a third anchor size changed little on these scenes. At every setting AP is near 0.9 at IoU 0.5 and near zero at 0.95: the remaining error is a pixel or two of localization on small objects. The full run trains the RetinaNet recipe and the cross-entropy ablation once each, with one seed, in about 30 seconds on the prod image, and prints the data statistics, the loss curve, mAP by threshold and by class, the effect of switching NMS off, two example scenes with their detections, and the ablation. Its numbers come from a single seed and are noisy; the sweep above says which comparisons held on every seed.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** box_area
- [x] **2.** box_iou

---

Built on Deep-ML.
