"""
PlantCare: Real-Time Plant Leaf Detection and Disease Classification on Raspberry Pi 5.

Pipeline:
Camera Frame (640x480)
   │
   ▼
[Leaf Detector: MobileNet-SSD / SSDLite] (TFLite or ONNX)
   │
   ├──▶ Detected Bounding Boxes [x1, y1, x2, y2]
   │
   ▼
Crop & Preprocess Detected Leaf Patches
   │
   ▼
[Disease Classifier: MobileNetV3 38-Class] (TFLite or ONNX)
   │
   ├──▶ Crop Classification + Confidence Gating (Healthy / Disease / Uncertain)
   │
   ▼
Render Visual Badges, Severity Colors & HUD Overlay
   │
   ▼
Display (Local OpenCV Window / Remote Web Streamer)
"""

import os
import sys
import time
import argparse
import logging
import signal
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.camera import PiCameraStream
from src.detector import LeafDetector
from src.classifier import DiseaseClassifierLite
from src.visualizer import PlantCareVisualizer
from src.web_streamer import WebStreamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LiveLeafDetection")


def parse_args():
    parser = argparse.ArgumentParser(description="PlantCare: Embedded Leaf Detection & Disease Classification")
    parser.add_argument("--detector-model", type=str, default="models/leaf_detector.tflite",
                        help="Path to leaf detector model (.tflite or .onnx)")
    parser.add_argument("--classifier-model", type=str, default="models/disease_classifier.tflite",
                        help="Path to disease classifier model (.tflite or .onnx)")
    parser.add_argument("--leaf-labels", type=str, default="labels/leaf_labels.txt",
                        help="Path to leaf labels file")
    parser.add_argument("--disease-labels", type=str, default="labels/disease_labels.txt",
                        help="Path to 38-class disease labels file")
    parser.add_argument("--conf-detector", type=float, default=0.45,
                        help="Confidence threshold for leaf detector (default: 0.45)")
    parser.add_argument("--conf-classifier", type=float, default=0.60,
                        help="Confidence threshold for disease classifier (default: 0.60)")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera stream width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera stream height (default: 480)")
    parser.add_argument("--fps-target", type=int, default=30,
                        help="Target camera framerate (default: 30)")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Process inference every N frames (1 = every frame, 2 = every 2nd frame for low RAM/CPU)")
    parser.add_argument("--classify", action="store_true", default=True,
                        help="Enable disease classification for detected leaves")
    parser.add_argument("--no-classify", dest="classify", action="store_false",
                        help="Disable disease classification (leaf detection only)")
    parser.add_argument("--display", action="store_true", default=False,
                        help="Show local OpenCV GUI window (requires monitor or X11 forwarding)")
    parser.add_argument("--web", action="store_true", default=True,
                        help="Enable web streaming server for headless viewing in browser")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for web streaming server (default: 5000)")
    parser.add_argument("--allow-mock", action="store_true", default=False,
                        help="Allow running in mock mode if trained model files are not yet present")
    return parser.parse_args()


def get_memory_usage_mb() -> float:
    """Returns current process RAM usage in Megabytes."""
    try:
        import resource
        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return usage / 1024.0
        return usage / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def main():
    args = parse_args()

    # Determine display availability
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    show_window = args.display
    if show_window and not has_display:
        logger.warning(
            "Local display requested (--display) but no DISPLAY or WAYLAND_DISPLAY environment variable found!\n"
            "If connecting via SSH, consider viewing the live feed via the Web Streamer at http://<pi-ip>:%d\n"
            "or use SSH with X11 forwarding: ssh -X pi@<pi-ip>", args.port
        )
        show_window = False

    logger.info("=" * 60)
    logger.info("🌿 PlantCare Embedded AI: Raspberry Pi 5")
    logger.info(f"Resolution: {args.width}x{args.height} | Skip Frames: {args.skip_frames}")
    logger.info(f"Classification Enabled: {args.classify}")
    logger.info(f"Local GUI Window: {show_window} | Web Streamer: {args.web} (Port: {args.port})")
    logger.info("=" * 60)

    # 1. Initialize Leaf Detector
    logger.info("Loading Leaf Object Detector...")
    try:
        detector = LeafDetector(
            model_path=args.detector_model,
            labels_path=args.leaf_labels,
            conf_threshold=args.conf_detector,
            allow_mock=args.allow_mock
        )
    except Exception as e:
        logger.error(f"Failed to load Leaf Detector: {e}")
        logger.info(
            "\nTIP: If you haven't trained or downloaded models yet, run with --allow-mock to test the camera & UI:\n"
            "  python src/live_leaf_detection.py --allow-mock\n"
        )
        return

    # 2. Initialize Disease Classifier (if enabled)
    classifier = None
    if args.classify:
        logger.info("Loading Disease Classifier...")
        try:
            classifier = DiseaseClassifierLite(
                model_path=args.classifier_model,
                labels_path=args.disease_labels,
                conf_threshold=args.conf_classifier,
                allow_mock=args.allow_mock
            )
        except Exception as e:
            logger.warning(f"Could not load disease classifier: {e}. Continuing with Leaf Detection only.")
            args.classify = False

    # 3. Initialize Visualizer
    visualizer = PlantCareVisualizer()

    # 4. Initialize Web Streamer (if enabled)
    streamer = None
    if args.web:
        streamer = WebStreamer(host="0.0.0.0", port=args.port)
        streamer.start()

    # 5. Initialize Camera
    logger.info("Starting camera stream...")
    try:
        camera = PiCameraStream(
            width=args.width,
            height=args.height,
            framerate=args.fps_target,
            autofocus=True
        ).start()
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        if streamer:
            streamer.stop()
        return

    # Setup Graceful Exit handling for Ctrl+C
    is_running = True

    def signal_handler(sig, frame):
        nonlocal is_running
        logger.info("\nTermination signal received. Shutting down gracefully...")
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Performance Tracking Variables
    frame_count = 0
    fps = 0.0
    fps_start_time = time.time()
    fps_frame_counter = 0

    last_detections = []
    det_time_ms = 0.0
    cls_time_ms = 0.0

    logger.info("PlantCare AI pipeline running. Press 'q' in window or Ctrl+C in terminal to stop.\n")

    try:
        while is_running:
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            fps_frame_counter += 1

            # Compute smoothed FPS every 10 frames
            if fps_frame_counter >= 10:
                elapsed = time.time() - fps_start_time
                fps = fps_frame_counter / elapsed if elapsed > 0 else 0.0
                fps_frame_counter = 0
                fps_start_time = time.time()

            # Run detection every N frames to optimize CPU and RAM
            if frame_count % args.skip_frames == 0:
                # A. Run Leaf Object Detection
                t_det_start = time.time()
                detections = detector.detect(frame)
                det_time_ms = (time.time() - t_det_start) * 1000.0

                # B. Run Disease Classification for each detected leaf
                t_cls_start = time.time()
                if args.classify and classifier is not None and len(detections) > 0:
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        # Extract leaf crop with safety bounds
                        crop = frame[y1:y2, x1:x2]
                        if crop.shape[0] >= 15 and crop.shape[1] >= 15:
                            disease_result = classifier.classify_crop(crop)
                            det["disease"] = disease_result
                        else:
                            det["disease"] = {
                                "crop": "Leaf",
                                "condition": "Uncertain / Small",
                                "confidence": 0.0,
                                "severity": "uncertain",
                                "is_uncertain": True
                            }
                cls_time_ms = (time.time() - t_cls_start) * 1000.0
                last_detections = detections

            # C. Render annotations and HUD
            annotated_frame = visualizer.draw_detections(
                frame=frame,
                detections=last_detections,
                fps=fps,
                det_time_ms=det_time_ms,
                cls_time_ms=cls_time_ms,
                classification_enabled=args.classify
            )

            # D. Push to Web Streamer (if enabled)
            if streamer:
                streamer.update_frame(annotated_frame)

            # E. Display on local GUI window (if enabled)
            if show_window:
                cv2.imshow("PlantCare Live Detection (Press Q to quit)", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    logger.info("Quit key pressed. Exiting...")
                    break

            # Periodically log memory usage
            if frame_count % 150 == 0:
                mem_mb = get_memory_usage_mb()
                logger.info(
                    f"Status: FPS: {fps:.1f} | Leaves: {len(last_detections)} | "
                    f"Det: {det_time_ms:.1f}ms | Cls: {cls_time_ms:.1f}ms | RAM: {mem_mb:.1f} MB"
                )

    except Exception as e:
        logger.error(f"Unexpected error in live detection loop: {e}", exc_info=True)
    finally:
        logger.info("Releasing camera and system resources...")
        camera.stop()
        if streamer:
            streamer.stop()
        if show_window:
            cv2.destroyAllWindows()
        logger.info("PlantCare shutdown complete.")


if __name__ == "__main__":
    main()
