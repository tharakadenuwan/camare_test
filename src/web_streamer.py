"""
Lightweight Zero-Dependency MJPEG Web Streamer for Headless Raspberry Pi 5.
Uses Python standard library (http.server and threading) to stream real-time annotated
video to any web browser over Wi-Fi / Ethernet without requiring X11 or a monitor.
"""

import time
import io
import threading
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import cv2
import numpy as np

logger = logging.getLogger("WebStreamer")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌿 PlantCare Pi 5 Live Feed</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #121417;
            color: #ecf0f1;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            text-align: center;
        }
        .header {
            margin-bottom: 16px;
        }
        h1 {
            margin: 0 0 6px 0;
            font-size: 24px;
            color: #2ecc71;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .sub {
            color: #95a5a6;
            font-size: 14px;
        }
        .stream-card {
            display: inline-block;
            background: #1e222b;
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            border: 1px solid #2c3e50;
            max-width: 95vw;
        }
        .stream-img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }
        .badges {
            margin-top: 14px;
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        .badge {
            background: #2a303c;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            border: 1px solid #3e4a5d;
        }
        .badge span {
            font-weight: bold;
            color: #2ecc71;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌿 PlantCare Live Monitor</h1>
        <div class="sub">Raspberry Pi 5 &bull; Camera Module 3 &bull; Real-time AI Inference</div>
    </div>
    <div class="stream-card">
        <img class="stream-img" src="/stream.mjpg" alt="PlantCare Live Camera Stream">
        <div class="badges">
            <div class="badge">Architecture: <span>Non-YOLO MobileNet-SSD</span></div>
            <div class="badge">Engine: <span>LiteRT / TFLite</span></div>
            <div class="badge">Stream: <span>Low Latency MJPEG</span></div>
        </div>
    </div>
</body>
</html>
"""


class StreamingOutput:
    def __init__(self):
        self.frame: bytes = b""
        self.condition = threading.Condition()

    def update(self, frame_bytes: bytes):
        with self.condition:
            self.frame = frame_bytes
            self.condition.notify_all()


class StreamingHandler(BaseHTTPRequestHandler):
    output: StreamingOutput = None

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(HTML_PAGE)))
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with self.output.condition:
                        self.output.condition.wait()
                        frame = self.output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence access logs to keep terminal clean
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class WebStreamer:
    """
    HTTP MJPEG streaming server for remote browser viewing.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.output = StreamingOutput()
        StreamingHandler.output = self.output
        self.server = None
        self.thread = None
        self.running = False

    def start(self):
        """Starts HTTP server in a background daemon thread."""
        if self.running:
            return
        self.server = ThreadedHTTPServer((self.host, self.port), StreamingHandler)
        self.running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Web Streamer running at http://{self.host}:{self.port}/ (or http://<pi-ip>:{self.port}/)")

    def update_frame(self, frame_bgr: np.ndarray, quality: int = 80):
        """Encodes frame to JPEG and notifies waiting HTTP clients."""
        if not self.running:
            return
        ret, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret:
            self.output.update(jpeg.tobytes())

    def stop(self):
        """Stops HTTP server."""
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Web Streamer stopped.")
