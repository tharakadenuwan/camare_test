"""
Camera capture abstraction for Raspberry Pi 5 with Camera Module 3 and OpenCV fallback.
Supports Picamera2 with continuous autofocus (IMX708 sensor) and threaded frame capture
to eliminate frame buffering lag during inference.
"""

import time
import threading
import logging
from typing import Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CameraStream")

# Check if Picamera2 is available
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class PiCameraStream:
    """
    Threaded Camera Reader for Raspberry Pi Camera Module 3 and USB webcams.
    """
    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30, autofocus: bool = True):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.autofocus = autofocus
        self.is_picam = False
        self.picam2: Optional[object] = None
        self.cap: Optional[object] = None

        self.frame: Optional[np.ndarray] = None
        self.grabbed: bool = False
        self.running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self._initialize_camera()

    def _initialize_camera(self):
        """Initializes Picamera2 if available, otherwise falls back to OpenCV VideoCapture."""
        if PICAMERA2_AVAILABLE:
            try:
                logger.info("Initializing Raspberry Pi Camera Module 3 via Picamera2...")
                self.picam2 = Picamera2()
                # Create preview config with RGB888 format
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()

                # Enable autofocus controls for Camera Module 3 (IMX708)
                if self.autofocus:
                    try:
                        # AfMode: 2 is Continuous Autofocus, AfRange: 0 is Normal
                        self.picam2.set_controls({"AfMode": 2, "AfRange": 0})
                        logger.info("Continuous Autofocus enabled for Camera Module 3.")
                    except Exception as af_err:
                        logger.warning(f"Could not enable autofocus: {af_err}")

                # Warmup time for sensor exposure & white balance
                time.sleep(1.0)
                self.is_picam = True
                logger.info(f"Picamera2 successfully initialized at {self.width}x{self.height} @ {self.framerate} FPS.")
                return
            except Exception as e:
                logger.warning(f"Picamera2 initialization failed: {e}. Falling back to OpenCV VideoCapture...")
                if self.picam2:
                    try:
                        self.picam2.stop()
                        self.picam2.close()
                    except Exception:
                        pass
                self.picam2 = None

        # Fallback to cv2.VideoCapture
        import cv2
        logger.info("Initializing OpenCV VideoCapture (device index 0)...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.framerate)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open any camera device (neither Picamera2 nor OpenCV VideoCapture).")
        self.is_picam = False
        logger.info("OpenCV VideoCapture successfully opened.")

    def start(self) -> "PiCameraStream":
        """Starts background frame reader thread."""
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # Wait until first frame is grabbed
        for _ in range(50):
            if self.grabbed and self.frame is not None:
                break
            time.sleep(0.05)
        return self

    def _capture_loop(self):
        """Continuously grabs frames to ensure zero buffering latency."""
        import cv2
        while self.running:
            try:
                if self.is_picam and self.picam2:
                    # Picamera2 configured with 'RGB888' is delivered in BGR byte order for OpenCV
                    frame = self.picam2.capture_array()
                    with self.lock:
                        self.frame = frame
                        self.grabbed = True
                elif self.cap and self.cap.isOpened():
                    ret, bgr_frame = self.cap.read()
                    if ret:
                        with self.lock:
                            self.frame = bgr_frame
                            self.grabbed = True
                    else:
                        time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.05)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the latest grabbed frame (BGR format). Thread-safe and non-blocking."""
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        """Stops capture loop and releases hardware resources safely."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.is_picam and self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
                logger.info("Picamera2 hardware released cleanly.")
            except Exception as e:
                logger.warning(f"Error closing Picamera2: {e}")
            self.picam2 = None

        if self.cap:
            try:
                self.cap.release()
                logger.info("OpenCV VideoCapture released cleanly.")
            except Exception as e:
                logger.warning(f"Error releasing VideoCapture: {e}")
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
