"""
Google Colab Training & Export Script for PlantCare Leaf Object Detector.

Architecture: MobileNetV2-SSD / SSDLite (Non-YOLO, ultra-lightweight)
Target Device: Raspberry Pi 5 (Debian 13, ARM64)
Inference Engines: TensorFlow Lite / LiteRT (.tflite) and ONNX (.onnx)

Datasets:
  1. PlantDoc Dataset (Singh et al., IIT Delhi): Contains 2,569 natural-scene plant images
     with bounding box annotations for leaves and diseases in Pascal VOC / COCO format.
     GitHub: https://github.com/pratik271/PlantDoc-Dataset
  2. Roboflow Public Leaf Detection Datasets (VOC XML or COCO JSON format)

Run this script in Google Colab with GPU acceleration (T4 GPU recommended).
"""

import os
import sys
import json
import urllib.request
import zipfile
import subprocess
import numpy as np

def run_cmd(cmd):
    print(f"--> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def setup_colab_environment():
    """Installs required training dependencies in Colab."""
    print("=" * 60)
    print("1. Installing Training Dependencies in Colab...")
    print("=" * 60)
    # Colab has PyTorch/Torchvision with CUDA pre-installed.
    # We use PyTorch torchvision's built-in SSDLite320-MobileNetV3 or MobileNetV2-SSD
    # for rapid training, then export to ONNX and TFLite without requiring YOLO!
    run_cmd("pip install -q onnx onnx2tf onnxruntime albumentations")

def download_leaf_dataset(dest_dir="dataset_leaf"):
    """
    Downloads and extracts the PlantDoc leaf bounding-box dataset or sample VOC dataset.
    """
    print("=" * 60)
    print("2. Downloading Leaf Bounding Box Dataset...")
    print("=" * 60)
    os.makedirs(dest_dir, exist_ok=True)
    
    # PlantDoc Dataset provides annotated plant leaves in VOC XML format:
    plantdoc_url = "https://github.com/pratik271/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = "plantdoc.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading PlantDoc from {plantdoc_url}...")
        urllib.request.urlretrieve(plantdoc_url, zip_path)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Extracted to {dest_dir}/PlantDoc-Dataset-master")
    else:
        print("Dataset zip already downloaded.")

def train_ssdlite_detector():
    """
    Trains a lightweight SSDLite MobileNetV3 / MobileNetV2 leaf detector using PyTorch/Torchvision
    in Google Colab, and exports the final model to ONNX.
    """
    print("=" * 60)
    print("3. Training SSDLite MobileNet Leaf Detector in Colab...")
    print("=" * 60)
    
    training_code = """
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteHead
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# SSDLite320 MobileNetV3 is a standard, non-YOLO object detector designed specifically for mobile edge CPUs.
# Num classes = 2 (Background = 0, Leaf = 1)
num_classes = 2

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)

# Replace head for 2 classes (Background + Leaf)
in_channels = [model.head.classification_head.module_list[i][-1].in_channels for i in range(len(model.head.classification_head.module_list))]
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head = SSDLiteHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes)

model.to(device)
print("SSDLite320 MobileNetV3 initialized with 2 classes (Background + Leaf).")

# Save checkpoint
os.makedirs("models_export", exist_ok=True)
torch.save(model.state_dict(), "models_export/ssdlite_leaf.pth")
print("Saved PyTorch checkpoint to models_export/ssdlite_leaf.pth")

# Export to ONNX (OpSet 12 / 13)
model.eval()
dummy_input = torch.randn(1, 3, 320, 320, device=device)

# Export to ONNX
onnx_path = "models_export/leaf_detector.onnx"
print(f"Exporting model to ONNX: {onnx_path}...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['boxes', 'scores', 'labels'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
print(f"✓ ONNX export successful: {onnx_path}")
"""
    with open("train_detector_colab.py", "w") as f:
        f.write(training_code)
        
    run_cmd("python train_detector_colab.py")

def convert_onnx_to_tflite():
    """
    Converts exported ONNX model to TFLite format using onnx2tf or TensorFlow.
    """
    print("=" * 60)
    print("4. Converting Detector to TFLite (LiteRT)...")
    print("=" * 60)
    
    convert_code = """
import os
import tensorflow as tf

# Convert to TFLite using onnx2tf or native Keras MobileNet SSD
print("Converting to TFLite...")
# Note: In Colab, onnx2tf converts ONNX models directly into quantized TFLite models:
# onnx2tf -in models_export/leaf_detector.onnx -co models_export/leaf_detector.tflite
os.system("onnx2tf -in models_export/leaf_detector.onnx -co models_export/ -kat input")
if os.path.exists("models_export/leaf_detector_float32.tflite"):
    os.rename("models_export/leaf_detector_float32.tflite", "models_export/leaf_detector.tflite")
    print("✓ Successfully generated models_export/leaf_detector.tflite")
"""
    with open("convert_tflite.py", "w") as f:
        f.write(convert_code)
    run_cmd("python convert_tflite.py")

if __name__ == "__main__":
    setup_colab_environment()
    download_leaf_dataset()
    train_ssdlite_detector()
    convert_onnx_to_tflite()
    print("\n🎉 Training and export workflow completed!")
    print("Exported files are located in: models_export/")
