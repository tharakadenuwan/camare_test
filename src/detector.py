"""
Lightweight Non-YOLO Leaf Object Detector.
Supports MobileNet-SSD / SSDLite models in TensorFlow Lite / LiteRT (.tflite) and ONNX (.onnx).
Optimized for Raspberry Pi 5 with low RAM consumption and fast CPU inference.
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger("LeafDetector")


def get_tflite_interpreter(model_path: str, num_threads: int = 4):
    """
    Attempts to load a TFLite interpreter using available runtimes:
    1. ai-edge-litert (Google LiteRT - modern successor)
    2. tflite-runtime (legacy standalone runtime)
    3. tensorflow.lite (full TF fallback)
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
        logger.info("Using Google LiteRT (ai-edge-litert) runtime.")
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter
        logger.info("Using tflite-runtime.")
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        logger.info("Using tensorflow.lite interpreter fallback.")
        return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        raise ImportError(
            "No TensorFlow Lite runtime found. Please install LiteRT or ONNX Runtime:\n"
            "  pip install ai-edge-litert\n"
            "or run with ONNX Runtime: pip install onnxruntime"
        )


class LeafDetector:
    """
    Non-YOLO Leaf Object Detector.
    Detects leaves in camera frames and outputs bounding boxes [x1, y1, x2, y2]
    with confidence scores.
    """
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        num_threads: int = 4,
        allow_mock: bool = False
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_threads = num_threads
        self.allow_mock = allow_mock
        self.is_mock = False

        self.labels = ["leaf"]
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]

        self.backend = None
        self.interpreter = None
        self.ort_session = None

        self.input_shape = (300, 300)
        self.input_dtype = np.float32
        self.input_index = 0
        self.output_details = None

        self._load_model()

    def _load_model(self):
        """Loads TFLite or ONNX model depending on file extension."""
        if not os.path.exists(self.model_path):
            if self.allow_mock:
                logger.warning(
                    f"Model file '{self.model_path}' not found. Initializing in MOCK mode for hardware/pipeline verification."
                )
                self.is_mock = True
                return
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please train or export your leaf detector model to '{self.model_path}'.\n"
                f"See training/colab_train_detector.py for instructions."
            )

        ext = os.path.splitext(self.model_path)[1].lower()

        if ext == ".tflite":
            self.backend = "tflite"
            self.interpreter = get_tflite_interpreter(self.model_path, num_threads=self.num_threads)
            self.interpreter.allocate_tensors()

            input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.input_index = input_details[0]['index']
            # Shape is typically [1, height, width, channels]
            shape = input_details[0]['shape']
            self.input_shape = (int(shape[1]), int(shape[2]))
            self.input_dtype = input_details[0]['dtype']
            logger.info(f"Loaded TFLite Detector: input shape {self.input_shape}, dtype {self.input_dtype}")

        elif ext == ".onnx":
            self.backend = "onnx"
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("onnxruntime is required for .onnx models. Run: pip install onnxruntime")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.num_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])

            inputs = self.ort_session.get_inputs()
            shape = inputs[0].shape
            # Detect NCHW vs NHWC
            if len(shape) == 4 and shape[1] in [1, 3]:
                # NCHW
                self.input_shape = (int(shape[2]), int(shape[3]))
                self.is_nchw = True
            else:
                self.input_shape = (int(shape[1]), int(shape[2]))
                self.is_nchw = False
            logger.info(f"Loaded ONNX Detector: input shape {self.input_shape}, NCHW={getattr(self, 'is_nchw', False)}")
        else:
            raise ValueError(f"Unsupported model extension: {ext}. Expected .tflite or .onnx")

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Preprocesses frame for model input."""
        target_h, target_w = self.input_shape
        resized = cv2.resize(frame_bgr, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.backend == "tflite":
            if self.input_dtype == np.uint8:
                input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
            else:
                # Normalization to [0, 1] or [-1, 1]
                input_data = np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0
            return input_data

        elif self.backend == "onnx":
            if getattr(self, "is_nchw", False):
                # Transpose HWC -> CHW
                chw = np.transpose(rgb, (2, 0, 1))
                input_data = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0
            else:
                input_data = np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0
            return input_data

        return rgb

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs leaf detection on a single camera frame.
        Returns a list of dicts:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "score": float,
                    "class_id": int,
                    "label": "leaf"
                }, ...
            ]
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # Handle Mock Mode for testing
        if self.is_mock:
            # Generate sample simulated leaf box centered in the frame
            pad_w, pad_h = int(orig_w * 0.25), int(orig_h * 0.25)
            x1, y1 = pad_w, pad_h
            x2, y2 = orig_w - pad_w, orig_h - pad_h
            return [{
                "box": [x1, y1, x2, y2],
                "score": 0.88,
                "class_id": 0,
                "label": "leaf (mock)"
            }]

        input_data = self.preprocess(frame_bgr)
        detections: List[Dict[str, Any]] = []

        if self.backend == "tflite":
            self.interpreter.set_tensor(self.input_index, input_data)
            self.interpreter.invoke()

            # Typical MobileNet SSD TFLite outputs:
            # 0: Boxes [1, num_detections, 4] (ymin, xmin, ymax, xmax)
            # 1: Classes [1, num_detections]
            # 2: Scores [1, num_detections]
            # 3: Num detections [1]
            out0 = self.interpreter.get_tensor(self.output_details[0]['index'])
            out1 = self.interpreter.get_tensor(self.output_details[1]['index'])
            out2 = self.interpreter.get_tensor(self.output_details[2]['index'])
            out3 = self.interpreter.get_tensor(self.output_details[3]['index']) if len(self.output_details) > 3 else None

            # Determine which tensor corresponds to boxes and scores
            boxes = None
            classes = None
            scores = None

            for tensor in [out0, out1, out2, out3]:
                if tensor is None:
                    continue
                shape = tensor.shape
                if len(shape) == 3 and shape[-1] == 4:
                    boxes = tensor[0]
                elif len(shape) == 2 and (scores is None or classes is None):
                    # Check dtype or values to distinguish classes from scores
                    if np.issubdtype(tensor.dtype, np.floating) and np.max(tensor) <= 1.01:
                        scores = tensor[0]
                    else:
                        classes = tensor[0]

            if boxes is None or scores is None:
                # Fallback to direct positional indexing
                boxes = out0[0]
                classes = out1[0] if len(self.output_details) > 1 else np.zeros(len(boxes))
                scores = out2[0] if len(self.output_details) > 2 else np.ones(len(boxes))

            num = int(out3[0]) if out3 is not None else len(scores)

            for i in range(min(num, len(scores))):
                score = float(scores[i])
                if score < self.conf_threshold:
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                # Scale to original dimensions
                x1 = int(max(0, xmin * orig_w))
                y1 = int(max(0, ymin * orig_h))
                x2 = int(min(orig_w - 1, xmax * orig_w))
                y2 = int(min(orig_h - 1, ymax * orig_h))

                # Check box validity
                if (x2 - x1) >= 15 and (y2 - y1) >= 15:
                    class_id = int(classes[i]) if classes is not None else 0
                    label = self.labels[class_id] if class_id < len(self.labels) else "leaf"
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "score": round(score, 3),
                        "class_id": class_id,
                        "label": label
                    })

        elif self.backend == "onnx":
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: input_data})

            # Check if outputs contain pre-NMS or post-NMS results
            if len(outputs) >= 3 and len(outputs[0].shape) == 3 and outputs[0].shape[-1] == 4:
                boxes = outputs[0][0]
                classes = outputs[1][0]
                scores = outputs[2][0]

                for i in range(len(scores)):
                    score = float(scores[i])
                    if score < self.conf_threshold:
                        continue
                    ymin, xmin, ymax, xmax = boxes[i]
                    x1 = int(max(0, xmin * orig_w))
                    y1 = int(max(0, ymin * orig_h))
                    x2 = int(min(orig_w - 1, xmax * orig_w))
                    y2 = int(min(orig_h - 1, ymax * orig_h))

                    if (x2 - x1) >= 15 and (y2 - y1) >= 15:
                        class_id = int(classes[i])
                        label = self.labels[class_id] if class_id < len(self.labels) else "leaf"
                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "score": round(score, 3),
                            "class_id": class_id,
                            "label": label
                        })
            elif len(outputs) == 1:
                # Raw SSD output [1, num_anchors, 4 + num_classes] or [1, 5, num_anchors]
                pred = outputs[0][0]
                if pred.shape[0] < pred.shape[1]:
                    pred = pred.T  # Transpose to [num_anchors, features]

                boxes_xywh = pred[:, :4]
                scores_all = pred[:, 4:]

                # Find max score per anchor
                max_scores = np.max(scores_all, axis=1)
                mask = max_scores >= self.conf_threshold
                filtered_boxes = boxes_xywh[mask]
                filtered_scores = max_scores[mask]
                filtered_classes = np.argmax(scores_all[mask], axis=1)

                cv_boxes = []
                for b in filtered_boxes:
                    cx, cy, w, h = b
                    x1 = int((cx - w / 2) * orig_w)
                    y1 = int((cy - h / 2) * orig_h)
                    cv_boxes.append([x1, y1, int(w * orig_w), int(h * orig_h)])

                indices = cv2.dnn.NMSBoxes(cv_boxes, filtered_scores.tolist(), self.conf_threshold, self.iou_threshold)
                if len(indices) > 0:
                    for idx in indices.flatten():
                        bx, by, bw, bh = cv_boxes[idx]
                        x1 = max(0, bx)
                        y1 = max(0, by)
                        x2 = min(orig_w - 1, bx + bw)
                        y2 = min(orig_h - 1, by + bh)
                        cid = int(filtered_classes[idx])
                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "score": round(float(filtered_scores[idx]), 3),
                            "class_id": cid,
                            "label": self.labels[cid] if cid < len(self.labels) else "leaf"
                        })

        return detections
