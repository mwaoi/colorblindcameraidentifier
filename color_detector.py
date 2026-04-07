import cv2
import numpy as np
import threading
import voice_output
from color_namer import get_color_name
from voice_output import speak
from vision_identifier import identify_color

VOTE_REGION_SIZE = 160   # px, square region around crosshair
VOTE_GRID = 9            # 9x9 = 81 sample points
RETICLE_COLOR = (0, 255, 0)
WINDOW_TITLE = "Color Identifier  |  SPACE = identify  |  Q = quit"
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
WARMUP_FRAMES = 10


class ColorDetector:
    def __init__(self, camera_index: int = 0):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {camera_index}. Is a camera connected?"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # disable hardware AWB if supported

    def run(self) -> None:
        state = {"color": "", "identifying": False, "pending": None}
        frame_count = 0
        try:
            cv2.namedWindow(WINDOW_TITLE)
            cv2.createTrackbar("Volume", WINDOW_TITLE, 100, 100,
                               lambda v: voice_output.set_volume(v / 100.0))
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                frame_count += 1
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2

                # Pick up completed vision result from background thread
                if state["pending"] is not None:
                    state["color"] = state["pending"]
                    state["pending"] = None
                    state["identifying"] = False

                self._draw_reticle(frame, cx, cy)

                # Capture here: reticle is drawn (gives Claude context),
                # but no text overlay yet (prevents "identifying..." from confusing the model)
                api_frame = frame.copy()

                display = "identifying..." if state["identifying"] else state["color"]
                if display:
                    self._draw_text_overlay(frame, display)

                cv2.imshow(WINDOW_TITLE, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and frame_count > WARMUP_FRAMES and not state["identifying"]:
                    state["identifying"] = True
                    state["color"] = ""
                    frame_copy = api_frame
                    cx_c, cy_c = cx, cy

                    def _identify(f=frame_copy, cx=cx_c, cy=cy_c):
                        color = identify_color(f)           # Claude Vision API
                        if not color:
                            color = self._vote_region(f, cx, cy)  # HSV fallback
                        speak(color)
                        state["pending"] = color

                    threading.Thread(target=_identify, daemon=True).start()

                elif key in (ord("q"), ord("Q")):
                    break
        finally:
            self._cap.release()
            cv2.destroyAllWindows()

    def _vote_region(self, frame: np.ndarray, cx: int, cy: int) -> str:
        """Sample a 5x5 grid across a 100x100 region and return the plurality color name."""
        half = VOTE_REGION_SIZE // 2
        step = VOTE_REGION_SIZE // (VOTE_GRID - 1)
        votes: dict[str, int] = {}
        for row in range(VOTE_GRID):
            for col in range(VOTE_GRID):
                x = cx - half + col * step
                y = cy - half + row * step
                x = max(1, min(frame.shape[1] - 2, x))
                y = max(1, min(frame.shape[0] - 2, y))
                # 3x3 patch average per point reduces single-pixel noise
                patch = frame[y - 1:y + 2, x - 1:x + 2].astype(np.float32)
                px_bgr = patch.mean(axis=(0, 1))
                b, g, r = px_bgr
                name = get_color_name(int(r), int(g), int(b))
                votes[name] = votes.get(name, 0) + 1
        return max(votes, key=votes.get)

    def _draw_reticle(self, frame: np.ndarray, cx: int, cy: int) -> None:
        half = VOTE_REGION_SIZE // 2
        cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), RETICLE_COLOR, 1, cv2.LINE_AA)

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
            frame,
            text,
            (x1 + padding, y2 - padding - baseline),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
