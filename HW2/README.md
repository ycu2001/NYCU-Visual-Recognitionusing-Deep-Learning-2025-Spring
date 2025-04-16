# HW2: Digit Recognition with Faster R-CNN

This project implements a digit detection and recognition pipeline using the Faster R-CNN framework. It is designed for a two-part task:
1. **Task 1** – Detect all digits in an image (object detection).
2. **Task 2** – Predict the digit sequence as a string (digit recognition).

## Dataset

- Dataset format: COCO-style annotations (for train/valid).
- Each image contains 1–4 digits.
- Test images are provided as PNG files without annotations.

```bash
./nycu-hw2-data/
├── train/       # Training images
├── valid/       # Validation images
├── test/        # Test images
├── train.json   # COCO annotations for training
├── valid.json   # COCO annotations for validation
```

## Results
1. **Task 1** – mAP: 0.35
2. **Task 2** – Accuracy: 0.76

## Environment
Colab
