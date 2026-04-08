"""
Lightweight YOLOv8n object detector.

Lazy-loads the model on first use (auto-downloads ~6MB yolov8n.pt).
Detects objects overlapping the center reticle region and routes them
to the appropriate color analysis pipeline.
"""

from __future__ import annotations
import sys
import numpy as np

# YOLO COCO class name → color analysis routing
_ROUTING_MAP: dict[str, str] = {
    # Person / body — routes to ITA skin tone detector
    "person": "skin",
    # Clothing and accessories
    "tie": "clothing",
    "backpack": "clothing",
    "umbrella": "clothing",
    "handbag": "clothing",
    "suitcase": "clothing",
    # Food and drink containers
    "cup": "food",
    "bottle": "food",
    "wine glass": "food",
    "bowl": "food",
    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",
}


class ObjectDetector:
    """
    Wraps YOLOv8n inference. Model is loaded lazily on first detect() call
    to avoid a ~1s import delay at app startup.
    """

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            print("Loading YOLO model (downloads ~6MB on first run)...", file=sys.stderr)
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
            print("YOLO model ready.", file=sys.stderr)

    def detect(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        region_size: int = 160,
    ) -> tuple[str, float] | None:
        """
        Run YOLOv8n on the frame. Return (class_name, confidence) for the
        highest-confidence COCO class whose bounding box overlaps the center
        region with >= 40% of the ROI covered, or None.

        Detections below 40% confidence are ignored.
        """
        self._load()

        half = region_size // 2
        roi_x1, roi_y1 = cx - half, cy - half
        roi_x2, roi_y2 = cx + half, cy + half
        roi_area = region_size * region_size

        results = self._model(frame, verbose=False)
        if not results or results[0].boxes is None:
            return None

        boxes = results[0].boxes
        names = results[0].names

        best_class: str | None = None
        best_conf: float = 0.0

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < 0.40:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy()
            bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Compute intersection with center ROI
            inter_x1 = max(bx1, roi_x1)
            inter_y1 = max(by1, roi_y1)
            inter_x2 = min(bx2, roi_x2)
            inter_y2 = min(by2, roi_y2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                overlap_fraction = inter_area / roi_area
                # Require at least 40% of the ROI to be covered
                if overlap_fraction >= 0.40 and conf > best_conf:
                    best_conf = conf
                    best_class = names[int(boxes.cls[i])]

        return (best_class, best_conf) if best_class is not None else None

    def get_routing(self, yolo_class: str | None) -> str:
        """Map a YOLO class name to a color analysis routing category."""
        if yolo_class is None:
            return "general"
        return _ROUTING_MAP.get(yolo_class.lower(), "general")
