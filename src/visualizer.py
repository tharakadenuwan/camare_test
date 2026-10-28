"""
High-contrast Visualizer for PlantCare Leaf Detection and Disease Classification.
Renders smooth bounding boxes, color-coded severity badges, and performance HUD.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

# BGR Color definitions
COLOR_HEALTHY = (70, 200, 70)      # Vivid Green
COLOR_MODERATE = (0, 165, 255)     # Amber / Orange
COLOR_SEVERE = (50, 50, 220)       # Crimson Red
COLOR_UNCERTAIN = (210, 180, 0)    # Cyan / Goldenrod
COLOR_LEAF_ONLY = (200, 150, 50)   # Blue-Gray


def get_severity_color(severity: str) -> Tuple[int, int, int]:
    """Returns BGR color based on leaf health status."""
    if severity == "healthy":
        return COLOR_HEALTHY
    elif severity == "moderate":
        return COLOR_MODERATE
    elif severity == "severe":
        return COLOR_SEVERE
    elif severity == "uncertain":
        return COLOR_UNCERTAIN
    return COLOR_LEAF_ONLY


class PlantCareVisualizer:
    """
    Renders annotations, badges, and real-time statistics onto camera frames.
    """
    def __init__(self, font_scale: float = 0.55, thickness: int = 2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float = 0.0,
        det_time_ms: float = 0.0,
        cls_time_ms: float = 0.0,
        classification_enabled: bool = True
    ) -> np.ndarray:
        """
        Draws bounding boxes, disease diagnostics, and top HUD banner on the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw each detected leaf
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            leaf_score = item.get("score", 0.0)

            # Determine disease label and color
            disease_info = item.get("disease")
            if classification_enabled and disease_info:
                severity = disease_info.get("severity", "uncertain")
                color = get_severity_color(severity)
                crop = disease_info.get("crop", "Leaf")
                cond = disease_info.get("condition", "Unknown")
                conf = int(disease_info.get("confidence", 0.0) * 100)
                if disease_info.get("is_uncertain", False):
                    label_text = f"Leaf (Uncertain: {conf}%)"
                else:
                    label_text = f"{crop}: {cond} ({conf}%)"
            else:
                color = COLOR_LEAF_ONLY
                conf = int(leaf_score * 100)
                label_text = f"Leaf ({conf}%)"

            # Draw outer bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.thickness)

            # Draw stylish corner accents for high visibility
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            thick_accent = self.thickness + 2
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thick_accent)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thick_accent)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thick_accent)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thick_accent)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, thick_accent)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, thick_accent)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, thick_accent)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, thick_accent)

            # Draw text badge above bounding box
            (tw, th), baseline = cv2.getTextSize(label_text, self.font, self.font_scale, 1)
            badge_y1 = max(0, y1 - th - 10)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(w, x1 + tw + 10)

            # Badge background with semi-transparency
            badge_roi = annotated[badge_y1:badge_y2, badge_x1:badge_x2]
            if badge_roi.shape[0] > 0 and badge_roi.shape[1] > 0:
                dark_bg = np.full_like(badge_roi, (25, 25, 25))
                annotated[badge_y1:badge_y2, badge_x1:badge_x2] = cv2.addWeighted(badge_roi, 0.3, dark_bg, 0.7, 0)
                # Colored left-strip on badge
                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x1 + 4, badge_y2), color, -1)

            # Badge text
            cv2.putText(
                annotated,
                label_text,
                (badge_x1 + 8, badge_y2 - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 2. Draw Top HUD Bar
        hud_h = 32
        hud_overlay = annotated[0:hud_h, 0:w].copy()
        cv2.rectangle(annotated, (0, 0), (w, hud_h), (20, 20, 20), -1)
        annotated[0:hud_h, 0:w] = cv2.addWeighted(hud_overlay, 0.25, annotated[0:hud_h, 0:w], 0.75, 0)

        # HUD Text items
        # FPS status color
        fps_color = (0, 255, 0) if fps >= 20 else ((0, 200, 255) if fps >= 12 else (0, 0, 255))
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated, fps_text, (10, 22), self.font, 0.55, fps_color, 2, cv2.LINE_AA)

        leaves_text = f"Leaves: {len(detections)}"
        cv2.putText(annotated, leaves_text, (110, 22), self.font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        if det_time_ms > 0:
            lat_text = f"Det: {det_time_ms:.0f}ms"
            if classification_enabled and cls_time_ms > 0:
                lat_text += f" | Cls: {cls_time_ms:.0f}ms"
            cv2.putText(annotated, lat_text, (220, 22), self.font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Brand on right
        brand_text = "PlantCare Pi 5"
        (bw, _), _ = cv2.getTextSize(brand_text, self.font, 0.5, 1)
        cv2.putText(annotated, brand_text, (w - bw - 10, 22), self.font, 0.5, (100, 220, 100), 1, cv2.LINE_AA)

        return annotated
