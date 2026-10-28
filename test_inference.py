"""
Independent AI Inference Test Script for Raspberry Pi 5.
Tests Leaf Detection and Disease Classification independently on a static image
without requiring camera hardware or live streaming.

Usage:
    python test_inference.py
    python test_inference.py --image my_leaf.jpg
    python test_inference.py --allow-mock
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestInference")

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.detector import LeafDetector
from src.classifier import DiseaseClassifierLite
from src.visualizer import PlantCareVisualizer


def create_synthetic_leaf_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates a synthetic plant leaf image for zero-hardware diagnostic testing."""
    # Background garden soil tone
    img = np.full((height, width, 3), (40, 50, 45), dtype=np.uint8)

    # Draw simulated leaf 1 (large green leaf on left)
    cv2.ellipse(img, (220, 240), (140, 90), 30, 0, 360, (50, 160, 60), -1)
    # Leaf stem
    cv2.line(img, (100, 300), (330, 180), (30, 120, 40), 3)
    # Simulated disease spots (Early Blight concentric rings)
    cv2.circle(img, (200, 220), 18, (20, 70, 110), -1)
    cv2.circle(img, (250, 260), 12, (15, 60, 90), -1)

    # Draw simulated leaf 2 (healthy smaller leaf on right)
    cv2.ellipse(img, (460, 260), (100, 60), -20, 0, 360, (60, 190, 70), -1)
    cv2.line(img, (380, 290), (530, 230), (35, 140, 45), 2)

    return img


def main():
    parser = argparse.ArgumentParser(description="Test PlantCare Detector & Classifier on a sample image")
    parser.add_argument("--image", type=str, default="", help="Path to input test image (default: synthetic leaf)")
    parser.add_argument("--detector-model", type=str, default="models/leaf_detector.tflite", help="Path to detector model")
    parser.add_argument("--classifier-model", type=str, default="models/disease_classifier.tflite", help="Path to classifier model")
    parser.add_argument("--leaf-labels", type=str, default="labels/leaf_labels.txt", help="Path to leaf labels")
    parser.add_argument("--disease-labels", type=str, default="labels/disease_labels.txt", help="Path to disease labels")
    parser.add_argument("--output", type=str, default="test_inference_output.jpg", help="Path to save annotated output image")
    parser.add_argument("--allow-mock", action="store_true", default=False, help="Allow running in mock mode if weights are missing")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🌿 PlantCare AI Inference Diagnostic Test")
    logger.info("=" * 60)

    # 1. Load or Generate Test Image
    if args.image and os.path.exists(args.image):
        logger.info(f"Loading user image: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            logger.error(f"Failed to read image at {args.image}")
            sys.exit(1)
    else:
        logger.info("No input image specified. Generating synthetic multi-leaf test image...")
        frame = create_synthetic_leaf_image()

    logger.info(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]} (WxH)")

    # 2. Test Leaf Detector
    logger.info("Initializing Leaf Object Detector...")
    try:
        detector = LeafDetector(
            model_path=args.detector_model,
            labels_path=args.leaf_labels,
            conf_threshold=0.35,
            allow_mock=args.allow_mock
        )
    except Exception as e:
        logger.error(f"Detector initialization failed: {e}")
        logger.info("Tip: specify --allow-mock to test pipeline logic if models are not yet trained.")
        sys.exit(1)

    t0 = time.time()
    detections = detector.detect(frame)
    det_time = (time.time() - t0) * 1000.0
    logger.info(f"✓ Leaf Detection completed in {det_time:.1f} ms. Found {len(detections)} leaves.")

    for i, d in enumerate(detections):
        logger.info(f"   Leaf #{i+1}: Box={d['box']}, Confidence={d['score']*100:.1f}%")

    # 3. Test Disease Classifier
    logger.info("Initializing Disease Classifier...")
    classifier = None
    try:
        classifier = DiseaseClassifierLite(
            model_path=args.classifier_model,
            labels_path=args.disease_labels,
            conf_threshold=0.55,
            allow_mock=args.allow_mock
        )
    except Exception as e:
        logger.warning(f"Disease Classifier initialization failed: {e}. Skipping classification test.")

    cls_time = 0.0
    if classifier and len(detections) > 0:
        t0 = time.time()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            crop = frame[y1:y2, x1:x2]
            diag = classifier.classify_crop(crop)
            det["disease"] = diag
            logger.info(
                f"   Diagnosis for Leaf #{i+1}: {diag['crop']} - {diag['condition']} "
                f"({diag['confidence']*100:.1f}%, Severity: {diag['severity']})"
            )
        cls_time = (time.time() - t0) * 1000.0
        logger.info(f"✓ Disease Classification completed in {cls_time:.1f} ms.")

    # 4. Render Annotations
    visualizer = PlantCareVisualizer()
    annotated = visualizer.draw_detections(
        frame=frame,
        detections=detections,
        fps=1000.0 / max(1.0, (det_time + cls_time)),
        det_time_ms=det_time,
        cls_time_ms=cls_time,
        classification_enabled=(classifier is not None)
    )

    cv2.imwrite(args.output, annotated)
    logger.info(f"✓ Annotated result saved to: {args.output}")
    logger.info("=" * 60)
    logger.info("🎉 INFERENCE TEST COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
