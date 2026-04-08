"""
Main color detection loop.

Identification pipeline on SPACE press:
  Thread A (local, ~100-200ms):
    1. YOLOv8n detects object class in center region
    2. Routes to skin detector (ITA) or Oklab k-NN based on object type
    3. Falls back to legacy HSV if Oklab fails
    4. Posts result to thread-safe queue

  Thread B (Claude Vision, ~1-2s, optional):
    1. Uses YOLO context to enrich the Claude prompt
    2. Posts result if API is available
    3. Only overrides local result if color differs

Results are collected each frame from a queue.Queue — no data races.
Stale results from previous SPACE presses are discarded via press_id.
"""

import cv2
import logging
import numpy as np
import queue
import threading
from dataclasses import dataclass

import oklab_namer
import skin_detector
import voice_output
from color_namer import get_color_name
from object_detector import ObjectDetector
from vision_identifier import identify_color
from voice_output import speak

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("color_detector")

VOTE_REGION_SIZE = 160
RETICLE_COLOR = (0, 255, 0)
WINDOW_TITLE = "Color Identifier  |  SPACE = identify  |  Q = quit"
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
WARMUP_FRAMES = 10


@dataclass
class AppState:
    color: str = ""
    identifying: bool = False
    space_press_id: int = 0


class ColorDetector:
    def __init__(self, camera_index: int = 0):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {camera_index}. Is a camera connected?"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self._result_queue: queue.Queue[tuple[str, str, int]] = queue.Queue()
        self._object_detector = ObjectDetector()

    def run(self) -> None:
        state = AppState()
        frame_count = 0
        try:
            cv2.namedWindow(WINDOW_TITLE)
            cv2.createTrackbar("Volume", WINDOW_TITLE, 100, 100,
                               lambda v: voice_output.set_volume(v / 100.0))
            cv2.createTrackbar("Rate", WINDOW_TITLE, 150, 300,
                               lambda v: voice_output.set_rate(v))
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame_count += 1
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2

                self._collect_pending(state)
                self._draw_reticle(frame, cx, cy)

                # Capture frame here: reticle drawn, no text overlay yet
                api_frame = frame.copy()

                display = "identifying..." if state.identifying else state.color
                if display:
                    self._draw_text_overlay(frame, display)

                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(" ") and frame_count > WARMUP_FRAMES and not state.identifying:
                    state.identifying = True
                    state.color = ""
                    state.space_press_id += 1
                    self._start_pipeline(state, api_frame, cx, cy, state.space_press_id)

                elif key in (ord("q"), ord("Q")):
                    break
        finally:
            self._cap.release()
            cv2.destroyAllWindows()

    def _collect_pending(self, state: AppState) -> None:
        """Drain result queue each frame. Thread-safe — queue.Queue is thread-safe by design."""
        while not self._result_queue.empty():
            try:
                color, source, press_id = self._result_queue.get_nowait()
            except queue.Empty:
                break

            if press_id != state.space_press_id:
                _log.debug("Discarding stale %s result '%s'", source, color)
                continue

            if source == "local":
                state.color = color
                state.identifying = False
                speak(color)
                _log.info("Local result: %s", color)

            elif source == "claude":
                # Claude arrived after local — only update if result meaningfully differs
                if color and color != state.color:
                    _log.info("Claude override: '%s' → '%s'", state.color, color)
                    state.color = color
                    speak(color)

    def _start_pipeline(
        self,
        state: AppState,
        frame: np.ndarray,
        cx: int,
        cy: int,
        press_id: int,
    ) -> None:
        def local_pipeline() -> None:
            try:
                yolo_class = self._object_detector.detect(frame, cx, cy, VOTE_REGION_SIZE)
                routing = self._object_detector.get_routing(yolo_class)
                _log.info("YOLO: class=%s routing=%s", yolo_class, routing)

                # Extract center ROI
                half = VOTE_REGION_SIZE // 2
                fh, fw = frame.shape[:2]
                y1, y2 = max(0, cy - half), min(fh, cy + half)
                x1, x2 = max(0, cx - half), min(fw, cx + half)
                roi = frame[y1:y2, x1:x2]

                # Person detected but no actual skin → it's clothing
                if routing == "skin" and not skin_detector.is_skin_region(roi):
                    routing = "clothing"
                    _log.debug("Person detected, no skin in ROI — rerouting to clothing")

                if routing == "skin":
                    color = skin_detector.get_skin_tone_name(roi)
                else:
                    mean_bgr = roi.mean(axis=(0, 1)) if roi.size > 0 else np.array([128., 128., 128.])
                    b_val, g_val, r_val = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
                    try:
                        color = oklab_namer.get_oklab_color_name(r_val, g_val, b_val)
                    except Exception as e:
                        _log.warning("Oklab failed (%s), using HSV fallback", e)
                        color = get_color_name(r_val, g_val, b_val)

                self._result_queue.put((color, "local", press_id))

            except Exception as e:
                _log.error("Local pipeline error: %s", e, exc_info=True)
                state.identifying = False

        def claude_pipeline() -> None:
            try:
                yolo_class = self._object_detector.detect(frame, cx, cy, VOTE_REGION_SIZE)
                color = identify_color(frame, object_context=yolo_class)
                if color:
                    self._result_queue.put((color, "claude", press_id))
            except Exception as e:
                _log.warning("Claude pipeline error: %s", e)

        threading.Thread(target=local_pipeline, daemon=True).start()
        threading.Thread(target=claude_pipeline, daemon=True).start()

    def _draw_reticle(self, frame: np.ndarray, cx: int, cy: int) -> None:
        half = VOTE_REGION_SIZE // 2
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                      RETICLE_COLOR, 1, cv2.LINE_AA)

    def _draw_text_overlay(self, frame: np.ndarray, text: str) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        h = frame.shape[0]
        padding = 10
        x1 = padding
        y2 = h - padding
        y1 = y2 - text_h - baseline - padding * 2
        x2 = x1 + text_w + padding * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(
            frame, text,
            (x1 + padding, y2 - padding - baseline),
            FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
        )
