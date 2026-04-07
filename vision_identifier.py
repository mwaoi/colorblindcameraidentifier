import base64

import cv2
import numpy as np

_PROMPT = """\
There is a green rectangle in the center of this image marking the area of interest.

First, identify what the main subject or object is inside that green rectangle \
(e.g. "human face", "coffee in mug", "blue shirt", "wooden table"). \
If you genuinely cannot tell, write "unknown".

Then, using that object identity as context, determine its true natural color — \
ignore lighting glare, specular highlights, and shadows. \
For a human face, give a skin tone (e.g. olive, brown, peach, dark brown, tan). \
For coffee or tea, say dark brown or brown. \
For clothing, give the fabric color ignoring folds/shadows.

Reply in exactly this format (two lines, nothing else):
object: <what it is>
color: <1-3 word color name>"""


def identify_color(frame: np.ndarray) -> str | None:
    """
    Send frame to Claude Vision API.
    Returns a color name string (e.g. 'olive', 'dark brown'), or None on failure.
    """
    try:
        import anthropic
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.standard_b64encode(buf).decode("utf-8")
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=40,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                }},
                {"type": "text", "text": _PROMPT},
            ]}],
        )
        return _parse_response(msg.content[0].text.strip().lower())
    except Exception:
        return None


def _parse_response(text: str) -> str:
    """Extract color from structured 'object: ...\ncolor: ...' response."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("color:"):
            color = line[len("color:"):].strip().rstrip(".")
            if color:
                return color
    # Fallback: if Claude didn't follow the format, use the whole response as-is
    # (strip any "object:" lines first)
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("object:")]
    return lines[0] if lines else None
