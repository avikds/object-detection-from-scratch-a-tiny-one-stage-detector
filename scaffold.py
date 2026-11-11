"""
Object Detection from Scratch: A Tiny One-Stage Detector scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""A tiny one-stage detector, end to end, in about 30 seconds on one CPU.

This run is a miniature next to the sweep the project description reports (eight seeds per setting, two learning-rate
schedules, four loss and initialization variants): here there is one seed, 1,000 training scenes, 400 training steps per
model and 256 validation scenes. Its numbers are noisy; on other seeds the gaps below are smaller or larger, and the
description says which comparisons held on every seed."""
from solution import *
import time
import numpy as np

T0 = time.time()
SIZE, STRIDE = 32, 4
GRID = SIZE // STRIDE
ANCHOR_SIZES = (8.0, 14.0)
NUM_CLASSES = 3
STEPS = 400
SEED = 0

print("1. Scenes, anchors and targets")
X, B, L = make_dataset(1000, seed=SEED)             # data stream [101, 0]
Xv, Bv, Lv = make_dataset(256, seed=SEED + 1)       # data stream [101, 1]: different scenes
labels = np.concatenate(L)
widths = np.concatenate([b[:, 2] - b[:, 0] for b in B])
print(f"   {len(X)} training scenes with {len(labels)} objects ({', '.join(f'{n} {c}s' for c, n in zip(CLASSES, np.bincount(labels)))}), "
      f"box widths {widths.min():.0f} to {widths.max():.0f} px; {len(Xv)} validation scenes with {sum(len(b) for b in Bv)} objects")
anchors = make_anchors(GRID, STRIDE, ANCHOR_SIZES)
T = build_targets(anchors, B, L, NUM_CLASSES)
best = np.concatenate([box_iou(anchors, b).max(0) for b in B if len(b)])
print(f"   {len(anchors)} anchors per image; per image {T['pos'].sum(1).mean():.1f} positive, {(1 - T['valid']).sum(1).mean():.1f} ignored, "
      f"{(T['valid'] - T['pos']).sum(1).mean():.1f} negative; {np.mean(best < 0.5):.0%} of objects have no anchor at IoU 0.5 "
      f"and get one only through the forced best match")

print(f"\n2. Training the detector (focal loss, gamma 2, alpha 0.25, prior 0.01, Adam at 0.003, {STEPS} steps of 32)")
model = TinyDetector(NUM_CLASSES, len(ANCHOR_SIZES), seed=SEED)   # init stream [202, 0]
t = time.time()
hist = Trainer(model).fit(X, T, STEPS, seed=SEED)                # minibatch stream [303, 0]
for i in (0, 49, 99, 199, 299, STEPS - 1):
    w = hist[max(0, i - 9):i + 1].mean(0)
    print(f"   step {i + 1:4d}: loss {w[0]:.3f} (classification {w[1]:.3f}, box {w[2]:.3f})")
print(f"   trained in {time.time() - t:.0f} s")

print("\n3. Evaluation on the validation scenes")
dets = detect(model, Xv, anchors, SIZE)
r = evaluate_map(dets, Bv, Lv, NUM_CLASSES)
print(f"   mAP@[.5:.95] {r['map']:.3f}   mAP@.5 {r['map50']:.3f}   mAP@.75 {r['map75']:.3f}")
print("   AP by IoU threshold: " + "  ".join(f"{thr:.2f} {v:.2f}" for thr, v in zip(IOU_THRESHOLDS, np.nanmean(r["ap"], 0))))
print("   AP@[.5:.95] by class: " + ", ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, np.nanmean(r["ap"], 1))))
r_off = evaluate_map(detect(model, Xv, anchors, SIZE, use_nms=False), Bv, Lv, NUM_CLASSES)
print(f"   the same detector without NMS: mAP@[.5:.95] {r_off['map']:.3f}, mAP@.5 {r_off['map50']:.3f}")
for i in range(2):
    print(f"   scene {i}: truth " + ", ".join(f"{CLASSES[l]} {b.astype(int).tolist()}" for b, l in zip(Bv[i], Lv[i])))
    db, ds, dl = dets[i]
    shown = [(b, s, l) for b, s, l in zip(db, ds, dl) if s >= 0.3]
    iou = box_iou(np.array([x[0] for x in shown]).reshape(-1, 4), Bv[i]).max(1, initial=0) if len(Bv[i]) else np.zeros(len(shown))
    print("            found " + (", ".join(f"{CLASSES[l]} {np.round(b, 1).tolist()} score {s:.2f} IoU {q:.2f}" for (b, s, l), q in zip(shown, iou)) or "nothing above 0.3"))

print(f"\n4. Ablation: the same scenes, initial weights and minibatches, trained with plain cross-entropy")
ab = run_ablation({"cross-entropy": {"gamma": 0.0, "alpha": -1.0}}, (X, T), (Xv, Bv, Lv), anchors, STEPS, seed=SEED)["cross-entropy"]
print(f"   focal loss:    mAP@[.5:.95] {r['map']:.3f}   mAP@.5 {r['map50']:.3f}   mAP@.75 {r['map75']:.3f}")
print(f"   cross-entropy: mAP@[.5:.95] {ab['map']:.3f}   mAP@.5 {ab['map50']:.3f}   mAP@.75 {ab['map75']:.3f}")
print("   Over eight seeds at this budget the focal loss won every time (0.586 against 0.496 on average, by 0.002 to 0.215), mostly by "
      "localizing better; with a cosine learning-rate schedule the order reverses (cross-entropy 0.639, focal 0.459).")
print(f"\nTotal {time.time() - T0:.0f} s")

