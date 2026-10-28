"""
Lightweight Plant Disease Classifier for Raspberry Pi 5.
Supports exported MobileNetV3 38-class models in TFLite / LiteRT and ONNX formats.
Classifies cropped leaf regions with confidence thresholding and 'Uncertain / Unknown' fallback.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

logger = logging.getLogger("DiseaseClassifierLite")

# Severity mapping aligned with backend/app/seed_diseases.py
SEVERITY_KEYWORDS = {
    "blight": "severe",
    "rot": "severe",
    "virus": "severe",
    "greening": "severe",
    "curl": "severe",
    "mosaic": "moderate",
    "wilt": "severe",
    "rust": "moderate",
    "spot": "moderate",
    "scab": "moderate",
    "mold": "moderate",
    "mildew": "moderate",
    "scorch": "moderate",
    "mite": "moderate",
}


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes softmax probabilities stably."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class DiseaseClassifierLite:
    """
    Lightweight plant disease classifier for cropped leaf patches.
    """
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        conf_threshold: float = 0.60,
        num_threads: int = 4,
        allow_mock: bool = False
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.conf_threshold = conf_threshold
        self.num_threads = num_threads
        self.allow_mock = allow_mock
        self.is_mock = False

        self.labels: List[str] = []
        self._load_labels()

        self.backend = None
        self.interpreter = None
        self.ort_session = None
        self.input_shape = (224, 224)
        self.input_dtype = np.float32
        self.input_index = 0
        self.output_index = 0

        self._load_model()

    def _load_labels(self):
        """Loads 38 PlantCare class labels from JSON or TXT file."""
        if not os.path.exists(self.labels_path):
            logger.warning(f"Labels file not found: {self.labels_path}. Using default 38 classes.")
            return

        if self.labels_path.endswith(".json"):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(self.labels)} disease classification labels.")

    def _load_model(self):
        """Loads TFLite or ONNX classifier model."""
        if not os.path.exists(self.model_path):
            if self.allow_mock:
                logger.warning(
                    f"Classifier model '{self.model_path}' not found. Initializing in MOCK mode."
                )
                self.is_mock = True
                return
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please place your exported 'disease_classifier.tflite' or '.onnx' in '{self.model_path}'.\n"
                f"See training/export_classifier_tflite.py for export instructions."
            )

        ext = os.path.splitext(self.model_path)[1].lower()

        if ext == ".tflite":
            self.backend = "tflite"
            from src.detector import get_tflite_interpreter
            self.interpreter = get_tflite_interpreter(self.model_path, num_threads=self.num_threads)
            self.interpreter.allocate_tensors()

            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            self.input_index = input_details[0]['index']
            self.output_index = output_details[0]['index']
            shape = input_details[0]['shape']
            self.input_shape = (int(shape[1]), int(shape[2]))
            self.input_dtype = input_details[0]['dtype']
            logger.info(f"Loaded TFLite Classifier: input shape {self.input_shape}, dtype {self.input_dtype}")

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
            if len(shape) == 4 and shape[1] in [1, 3]:
                self.input_shape = (int(shape[2]), int(shape[3]))
                self.is_nchw = True
            else:
                self.input_shape = (int(shape[1]), int(shape[2]))
                self.is_nchw = False
            logger.info(f"Loaded ONNX Classifier: input shape {self.input_shape}")
        else:
            raise ValueError(f"Unsupported classifier extension: {ext}")

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Preprocesses cropped leaf patch (resize 224x224, ImageNet normalization)."""
        target_h, target_w = self.input_shape
        resized = cv2.resize(crop_bgr, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Normalization with ImageNet mean and std (matching original backend/ai_model/train.py)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std

        if self.backend == "tflite":
            if self.input_dtype == np.uint8:
                # Quantized input
                return np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), axis=0).astype(np.uint8)
            return np.expand_dims(normalized, axis=0).astype(np.float32)

        elif self.backend == "onnx":
            if getattr(self, "is_nchw", False):
                chw = np.transpose(normalized, (2, 0, 1))
                return np.expand_dims(chw, axis=0).astype(np.float32)
            return np.expand_dims(normalized, axis=0).astype(np.float32)

        return np.expand_dims(normalized, axis=0).astype(np.float32)

    def _determine_severity(self, condition: str, is_healthy: bool) -> str:
        """Categorizes disease severity for color coding."""
        if is_healthy:
            return "healthy"
        cond_lower = condition.lower()
        for kw, sev in SEVERITY_KEYWORDS.items():
            if kw in cond_lower:
                return sev
        return "moderate"

    def classify_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Classifies a single cropped leaf image.
        Returns a dictionary:
            {
                "crop": str (e.g. 'Tomato'),
                "condition": str (e.g. 'Late Blight' or 'Healthy' or 'Uncertain / Unknown'),
                "raw_label": str,
                "confidence": float,
                "severity": 'healthy' | 'moderate' | 'severe' | 'uncertain',
                "is_uncertain": bool
            }
        """
        # Edge case: degenerate crop
        if crop_bgr is None or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
            return {
                "crop": "Leaf",
                "condition": "Too Small",
                "raw_label": "unknown",
                "confidence": 0.0,
                "severity": "uncertain",
                "is_uncertain": True
            }

        # Handle Mock Mode
        if self.is_mock:
            return {
                "crop": "Plant",
                "condition": "Healthy (Demo)",
                "raw_label": "Tomato___healthy",
                "confidence": 0.92,
                "severity": "healthy",
                "is_uncertain": False
            }

        input_data = self.preprocess(crop_bgr)

        if self.backend == "tflite":
            self.interpreter.set_tensor(self.input_index, input_data)
            self.interpreter.invoke()
            logits = self.interpreter.get_tensor(self.output_index)[0]

        elif self.backend == "onnx":
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: input_data})
            logits = outputs[0][0]
        else:
            logits = np.zeros(len(self.labels))

        probs = softmax(logits)
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        # If confidence is below threshold, mark as Uncertain rather than hallucinating
        if confidence < self.conf_threshold:
            return {
                "crop": "Leaf",
                "condition": "Uncertain / Unverified",
                "raw_label": "uncertain",
                "confidence": round(confidence, 3),
                "severity": "uncertain",
                "is_uncertain": True
            }

        raw_label = self.labels[best_idx] if best_idx < len(self.labels) else f"Class_{best_idx}"

        # Parse crop and condition from standard PlantVillage format: "Crop___Condition"
        if "___" in raw_label:
            crop_part, cond_part = raw_label.split("___", 1)
            crop_name = crop_part.replace("_", " ").title()
            condition_name = cond_part.replace("_", " ").title()
        else:
            crop_name = "Plant"
            condition_name = raw_label.replace("_", " ").title()

        is_healthy = "healthy" in condition_name.lower()
        severity = self._determine_severity(condition_name, is_healthy)

        return {
            "crop": crop_name,
            "condition": condition_name,
            "raw_label": raw_label,
            "confidence": round(confidence, 3),
            "severity": severity,
            "is_uncertain": False
        }
