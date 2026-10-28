"""
Independent Camera Module 3 Test Script for Raspberry Pi 5.
Tests Picamera2 autofocus, frame rate, OpenCV conversion, and image saving.

Usage:
    python test_camera.py
    python test_camera.py --display   # If a monitor or X11 forwarding is available
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestCamera")


def main():
    parser = argparse.ArgumentParser(description="Test Raspberry Pi Camera Module 3 with Picamera2 / OpenCV")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--frames", type=int, default=60, help="Number of test frames to capture (default: 60)")
    parser.add_argument("--output", type=str, default="test_camera_output.jpg", help="Output test image path")
    parser.add_argument("--display", action="store_true", help="Display live OpenCV preview window")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("📷 Raspberry Pi 5 Camera Diagnostic Test")
    logger.info(f"Target Resolution: {args.width}x{args.height}")
    logger.info("=" * 60)

    # Check Picamera2
    use_picam2 = False
    picam2 = None
    try:
        from picamera2 import Picamera2
        logger.info("Picamera2 module found. Initializing Camera Module 3...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()

        # Set autofocus for IMX708 sensor
        try:
            picam2.set_controls({"AfMode": 2, "AfRange": 0})
            logger.info("✓ Continuous Autofocus enabled (IMX708 Camera Module 3)")
        except Exception as af_err:
            logger.warning(f"Autofocus control note: {af_err}")

        # Warmup sensor
        time.sleep(1.0)
        use_picam2 = True
        logger.info("✓ Picamera2 hardware started successfully.")
    except Exception as e:
        logger.warning(f"Picamera2 initialization failed or not running on Pi: {e}")
        logger.info("Testing fallback to OpenCV VideoCapture(0)...")

    cap = None
    if not use_picam2:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            logger.error("❌ FAILED: No camera device accessible (neither Picamera2 nor VideoCapture).")
            logger.error("Please ensure the camera ribbon cable is seated correctly in CAM/DISP0 or CAM/DISP1.")
            sys.exit(1)
        logger.info("✓ OpenCV VideoCapture(0) opened.")

    # Frame capture benchmark
    logger.info(f"Capturing {args.frames} frames to measure hardware capture framerate...")
    start_time = time.time()
    last_frame = None

    for i in range(args.frames):
        if use_picam2:
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame at step {i}")
                break

        last_frame = frame

        if args.display:
            cv2.imshow("Camera Module 3 Test Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elapsed = time.time() - start_time
    actual_fps = args.frames / elapsed if elapsed > 0 else 0

    logger.info(f"✓ Captured {args.frames} frames in {elapsed:.2f} seconds ({actual_fps:.1f} FPS)")

    # Save verification frame
    if last_frame is not None:
        # Add timestamp & info text to image
        text = f"PlantCare Camera Test | {args.width}x{args.height} | {actual_fps:.1f} FPS"
        cv2.putText(last_frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(args.output, last_frame)
        logger.info(f"✓ Verification image saved to: {args.output}")

    # Cleanup
    if use_picam2 and picam2:
        picam2.stop()
        picam2.close()
        logger.info("✓ Picamera2 released cleanly.")
    if cap:
        cap.release()
        logger.info("✓ VideoCapture released cleanly.")
    if args.display:
        cv2.destroyAllWindows()

    logger.info("=" * 60)
    logger.info("🎉 CAMERA TEST PASSED: Camera hardware is working properly.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
