# Object Detection from Scratch: A Tiny One-Stage Detector

Build a complete one-stage object detector from nothing but numpy and PyTorch tensors, train it on a CPU in under a minute, and score it with the same mean average precision that the COCO benchmark uses. The data comes from a seeded scene generator you write first: 32 by 32 grayscale images holding one to four circles, squares and triangles of 5 to 16 pixels, on a sloped, noisy background, with bright and dark objects, so every box label is exact and a fresh dataset costs a second. From there you build the pieces of a RetinaNet-style detector one function at a time: box area and intersection over union, a grid of 128 anchors, the delta encoding of boxes relative to anchors and its inverse, IoU-based target assignment with positive, negative and ignored anchors and a forced best match for small objects, a five-layer convolutional backbone with a classification head and a box head, the prior-probability bias initialization, the focal loss and a smooth L1 box loss normalized by the number of positives, an Adam training loop, decoding with score threshold, top-k and per-class non-maximum suppression, greedy matching, 101-point interpolated average precision, mAP over ten IoU thresholds from 0.5 to 0.95, and a paired ablation harness. Data, weight initialization and minibatch order each come from their own random stream, so an ablation changes exactly one thing. What the full sweep found (eight seeds per setting, 1,000 training scenes, 400 steps of 32 images, 256 validation scenes, on the prod image). With Adam at a constant 0.003, the RetinaNet recipe (focal loss with gamma 2 and alpha 0.25, prior 0.01) reaches mAP@[.5:.95] 0.586 on average (0.511 to 0.646 across seeds), mAP@.5 0.87 and mAP@.75 0.71, and beats plain binary cross-entropy (0.496) on every seed, by 0.002 to 0.215; cross-entropy finds the objects as often (mAP@.5 0.94) but localizes them worse (mAP@.75 0.46). That ranking is not robust: under a cosine schedule from 0.006 cross-entropy reaches 0.639, the best setting measured, and the focal loss falls to 0.459, again on every seed. What did hold everywhere is the RetinaNet paper's other claim: without the prior initialization (prior 0.5), cross-entropy loses 0.15 to 0.24 mAP on average and on every seed, while the focal loss loses 0.03 or less, and not on every seed. Non-maximum suppression is worth 0.10 to 0.21 mAP; in a smaller three-seed sweep the forced best-anchor match and a third anchor size changed little on these scenes. At every setting AP is near 0.9 at IoU 0.5 and near zero at 0.95: the remaining error is a pixel or two of localization on small objects. The full run trains the RetinaNet recipe and the cross-entropy ablation once each, with one seed, in about 30 seconds on the prod image, and prints the data statistics, the loss curve, mAP by threshold and by class, the effect of switching NMS off, two example scenes with their detections, and the ablation. Its numbers come from a single seed and are noisy; the sweep above says which comparisons held on every seed.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** box_area
- [x] **2.** box_iou
- [x] **3.** shape_mask
- [x] **4.** mask_to_box
- [x] **5.** make_scene
- [x] **6.** make_dataset
- [x] **7.** make_anchors
- [x] **8.** encode_boxes
- [x] **9.** decode_boxes
- [x] **10.** assign_targets
- [x] **11.** build_targets
- [x] **12.** Conv
- [x] **13.** prior_bias
- [x] **14.** flatten_head
- [x] **15.** TinyDetector
- [x] **16.** focal_loss
- [x] **17.** smooth_l1
- [x] **18.** detection_loss
- [x] **19.** Trainer
- [x] **20.** predict_raw
- [x] **21.** nms
- [x] **22.** postprocess
- [x] **23.** detect
- [x] **24.** match_detections
- [x] **25.** average_precision
- [x] **26.** evaluate_map
- [x] **27.** run_ablation

## Results

```
1. Scenes, anchors and targets
   1000 training scenes with 2440 objects (784 circles, 842 squares, 814 triangles), box widths 5 to 16 px; 256 validation scenes with 639 objects
   128 anchors per image; per image 4.5 positive, 4.2 ignored, 119.4 negative; 14% of objects have no anchor at IoU 0.5 and get one only through the forced best match

2. Training the detector (focal loss, gamma 2, alpha 0.25, prior 0.01, Adam at 0.003, 400 steps of 32)
   step    1: loss 1.531 (classification 1.142, box 0.389)
   step   50: loss 0.801 (classification 0.529, box 0.272)
   step  100: loss 0.515 (classification 0.337, box 0.178)
   step  200: loss 0.322 (classification 0.235, box 0.087)
   step  300: loss 0.220 (classification 0.158, box 0.062)
   step  400: loss 0.165 (classification 0.108, box 0.057)
   trained in 10 s

3. Evaluation on the validation scenes
   mAP@[.5:.95] 0.568   mAP@.5 0.886   mAP@.75 0.676
   AP by IoU threshold: 0.50 0.89  0.55 0.88  0.60 0.87  0.65 0.85  0.70 0.80  0.75 0.68  0.80 0.47  0.85 0.21  0.90 0.03  0.95 0.00
   AP@[.5:.95] by class: circle 0.541, square 0.588, triangle 0.577
   the same detector without NMS: mAP@[.5:.95] 0.416, mAP@.5 0.602
   scene 0: truth circle [15, 6, 24, 15], circle [6, 20, 14, 28], square [3, 0, 13, 10], triangle [20, 22, 29, 31]
            found triangle [17.4, 20.5, 29.0, 31.7] score 0.70 IoU 0.62, circle [14.9, 6.9, 23.7, 15.4] score 0.56 IoU 0.82, triangle [5.8, 19.9, 14.4, 28.4] score 0.42 IoU 0.88, circle [5.8, 19.9, 14.4, 28.4] score 0.37 IoU 0.88, square [14.9, 6.9, 23.7, 15.4] score 0.36 IoU 0.82, square [2.3, 0.0, 13.3, 10.4] score 0.32 IoU 0.88
   scene 1: truth triangle [10, 7, 25, 22]
            found triangle [9.7, 7.7, 25.6, 23.1] score 0.95 IoU 0.84, triangle [16.6, 13.5, 25.2, 22.1] score 0.33 IoU 0.31, triangle [14.5, 7.7, 22.4, 16.2] score 0.31 IoU 0.29, circle [10.4, 8.3, 25.4, 22.4] score 0.31 IoU 0.85

4. Ablation: the same scenes, initial weights and minibatches, trained with plain cross-entropy
   focal loss:    mAP@[.5:.95] 0.568   mAP@.5 0.886   mAP@.75 0.676
   cross-entropy: mAP@[.5:.95] 0.531   mAP@.5 0.951   mAP@.75 0.512
   Over eight seeds at this budget the focal loss won every time (0.586 against 0.496 on average, by 0.002 to 0.215), mostly by localizing better; with a cosine learning-rate schedule the order reverses (cross-entropy 0.639, focal 0.459).

Total 21 s
```
