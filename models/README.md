# PlantCare Raspberry Pi Model Storage

This directory holds the exported lightweight inference models for the Raspberry Pi 5.

## Required Model Files:

1. **Leaf Object Detector (Non-YOLO MobileNet-SSD)**:
   - Primary: `leaf_detector.tflite`
   - Alternative: `leaf_detector.onnx`
   - Target task: Detects multiple leaves in the camera frame and outputs bounding boxes `[ymin, xmin, ymax, xmax]` with confidence scores.
   - Recommended resolution: 300x300 or 320x320.
   - Runtime size: ~5 MB to 12 MB.

2. **Disease Classifier (MobileNetV3 38-Class)**:
   - Primary: `disease_classifier.tflite`
   - Alternative: `disease_classifier.onnx`
   - Target task: Classifies individual cropped leaf patches into one of the 38 plant disease / healthy categories.
   - Input resolution: 224x224 RGB.
   - Runtime size: ~8 MB to 15 MB.

## How to Obtain or Generate Models:
Follow the training and export instructions in `training/colab_train_detector.py` or run the Google Colab notebook in `notebooks/train_leaf_detector_and_classifier.ipynb`. Once exported, transfer the `.tflite` or `.onnx` files into this directory:
```bash
# Example scp transfer from your laptop to Raspberry Pi:
scp leaf_detector.tflite pi@<raspberry-pi-ip>:~/plantCare/models/
scp disease_classifier.tflite pi@<raspberry-pi-ip>:~/plantCare/models/
```
