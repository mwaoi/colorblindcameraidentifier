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
from color_memory import ColorMemory, REJECTED
from color_namer import get_color_name
from object_detector import ObjectDetector
from vision_identifier import identify_color
from voice_output import speak

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("color_detector")

VOTE_REGION_SIZE = 160
RETICLE_COLOR = (0, 255, 0)
WINDOW_TITLE = "Color Identifier  |  SPACE = identify  |  Y = correct  |  N = wrong  |  Q = quit"
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
WARMUP_FRAMES = 10


def _highlight_robust_mean(roi: np.ndarray) -> np.ndarray:
    """
    Compute the mean BGR of an ROI after masking out specular highlights.

    Specular highlights are identified as pixels that are BOTH very bright
    (HSV V > 200) AND near-colorless (HSV S < 40). This specifically targets
    white glare from light sources — not legitimate bright colors like yellow
    (which has high V but also high S) or white objects (which are uniformly
    bright across the whole ROI, not just a small glare spot).

    If >80% of pixels are masked (e.g. pointing at a legitimately white object),
    falls back to the full mean so white/cream colors are still identified correctly.
    """
    if roi.size == 0:
        return np.array([128.0, 128.0, 128.0])

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Highlight pixels: very bright (V > 200) AND near-white (S < 40)
    highlight = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 40)
    non_highlight = ~highlight

    # Only use masked mean if enough real-surface pixels remain (> 20% of ROI)
    if non_highlight.sum() > roi.shape[0] * roi.shape[1] * 0.20:
        return roi[non_highlight].astype(np.float64).mean(axis=0)
    else:
        # Whole ROI is bright + desaturated → legitimately white/pale object
        return roi.mean(axis=(0, 1))


@dataclass
class AppState:
    color: str = ""
    identifying: bool = False
    space_press_id: int = 0
    last_oklab: np.ndarray | None = None  # Oklab of most recent local result


class ColorDetector:
    def __init__(self, camera_index: int = 0):
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {camera_index}. Is a camera connected?"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self._result_queue: queue.Queue[tuple[str, str, int, np.ndarray | None]] = queue.Queue()
        self._object_detector = ObjectDetector()
        self._memory = ColorMemory()

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

                elif key in (ord("y"), ord("Y")) and state.color and state.last_oklab is not None:
                    self._memory.confirm(state.last_oklab, state.color)
                    _log.info("User confirmed: '%s' (memory samples: %d)",
                              state.color, self._memory.sample_count())

                elif key in (ord("n"), ord("N")) and state.color and state.last_oklab is not None:
                    self._memory.reject(state.last_oklab, state.color)
                    _log.info("User rejected: '%s' (memory samples: %d)",
                              state.color, self._memory.sample_count())
                    state.color = "?"  # clear the wrong answer visually

                elif key in (ord("q"), ord("Q")):
                    break
        finally:
            self._cap.release()
            cv2.destroyAllWindows()

    def _collect_pending(self, state: AppState) -> None:
        """Drain result queue each frame. Thread-safe — queue.Queue is thread-safe by design."""
        while not self._result_queue.empty():
            try:
                color, source, press_id, oklab = self._result_queue.get_nowait()
            except queue.Empty:
                break

            if press_id != state.space_press_id:
                _log.debug("Discarding stale %s result '%s'", source, color)
                continue

            if source == "local":
                state.color = color
                state.last_oklab = oklab
                state.identifying = False
                speak(color)
                _log.info("Local result: %s", color)

            elif source == "claude":
                # Claude arrived after local — only update if result meaningfully differs
                if color and color != state.color:
                    _log.info("Claude override: '%s' → '%s'", state.color, color)
                    # Store Claude's correction in memory
                    if state.last_oklab is not None:
                        self._memory.add_correction(state.last_oklab, color)
                    state.color = color
                    speak(color)
                elif color and color == state.color and state.last_oklab is not None:
                    # Local and Claude agree — reinforce memory
                    self._memory.confirm(state.last_oklab, color)

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
                detection = self._object_detector.detect(frame, cx, cy, VOTE_REGION_SIZE)
                yolo_class = detection[0] if detection else None
                yolo_conf = detection[1] if detection else 0.0
                # Skin routing requires high-confidence person detection (≥0.65)
                routing = self._object_detector.get_routing(yolo_class)
                if routing == "skin" and yolo_conf < 0.65:
                    routing = "general"
                    _log.debug("Person at low conf (%.2f) — routing to general", yolo_conf)
                _log.info("YOLO: class=%s conf=%.2f routing=%s", yolo_class, yolo_conf, routing)

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

                # Compute ROI mean and Oklab coords (needed for memory check on all routes)
                mean_bgr = _highlight_robust_mean(roi)
                b_val, g_val, r_val = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
                try:
                    oklab = oklab_namer.rgb_to_oklab(r_val, g_val, b_val)
                except Exception:
                    oklab = None

                # Check memory: a prior rejection overrides skin routing
                if oklab is not None and routing == "skin":
                    mem = self._memory.predict(oklab)
                    if mem == REJECTED:
                        routing = "general"
                        _log.debug("Memory rejection — overriding skin route to general")

                if routing == "skin":
                    color = skin_detector.get_skin_tone_name(roi)
                    self._result_queue.put((color, "local", press_id, oklab))
                else:
                    # Check memory for a trusted positive hit
                    if oklab is not None:
                        mem = self._memory.predict(oklab)
                        if mem and mem != REJECTED:
                            _log.info("Memory hit: '%s'", mem)
                            self._result_queue.put((mem, "local", press_id, oklab))
                            return

                    try:
                        color = oklab_namer.get_oklab_color_name(r_val, g_val, b_val)
                    except Exception as e:
                        _log.warning("Oklab failed (%s), using HSV fallback", e)
                        color = get_color_name(r_val, g_val, b_val)
                        oklab = None

                    self._result_queue.put((color, "local", press_id, oklab))

            except Exception as e:
                _log.error("Local pipeline error: %s", e, exc_info=True)
                state.identifying = False

        def claude_pipeline() -> None:
            try:
                detection = self._object_detector.detect(frame, cx, cy, VOTE_REGION_SIZE)
                yolo_class = detection[0] if detection else None
                color = identify_color(frame, object_context=yolo_class)
                if color:
                    self._result_queue.put((color, "claude", press_id, None))
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
