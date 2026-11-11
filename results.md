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
