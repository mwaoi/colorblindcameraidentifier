"""
Skin detection and tone classification.

- is_skin_region(): HSV + YCbCr dual-mask approach for robust skin presence detection
- get_skin_tone_name(): ITA (Individual Typology Angle) formula on CIELAB gives
  8-level Monk Skin Tone Scale names — medically validated and inclusive across
  all human skin tones.

ITA formula: ITA = arctan2(L* - 50, b*) * (180/π)
where L*, b* are CIELAB coordinates. Higher ITA = lighter skin tone.
"""

import cv2
import numpy as np


def is_skin_region(roi_bgr: np.ndarray) -> bool:
    """
    Returns True if the ROI likely contains human skin.
    Requires HSV and YCbCr masks to agree on at least 22% of pixels.

    Key exclusions:
    - Very bright pixels (V > 215) are masked out before testing — white/light
      fabric and pale walls are bright but not skin.
    - Minimum HSV saturation raised to 28 — white shirts and cream walls have
      near-zero saturation and are excluded.
    - Minimum YCbCr Cr raised to 138 — skin has more red tone than neutral surfaces.
    """
    if roi_bgr.size == 0:
        return False

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Exclude very bright pixels (white shirts, pale walls) before any skin test
    brightness_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([180, 255, 215], dtype=np.uint8),  # V <= 215 only
    )

    # HSV skin range — raised saturation minimum (28) to exclude low-chroma surfaces
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 28, 50], dtype=np.uint8),
        np.array([25, 180, 215], dtype=np.uint8),
    )

    # YCbCr skin range — raised Cr minimum (138) to require meaningful red tone
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 138, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    # All three must agree: not too bright, in HSV skin range, in YCbCr skin range
    combined = cv2.bitwise_and(cv2.bitwise_and(brightness_mask, hsv_mask), ycrcb_mask)
    skin_fraction = np.count_nonzero(combined) / combined.size
    return skin_fraction >= 0.22


def get_skin_tone_name(roi_bgr: np.ndarray) -> str:
    """
    Compute mean CIELAB of the ROI, derive ITA angle, and return a
    descriptive Monk-scale skin tone name.

    IMPORTANT: OpenCV's cv2.COLOR_BGR2Lab outputs L in [0,255], a/b in [0,255]
    centered at 128. This is NOT the same as standard CIELAB — rescaling is required.
    """
    if roi_bgr.size == 0:
        return "skin"

    lab_cv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab)
    mean_lab = lab_cv.mean(axis=(0, 1)).astype(np.float64)

    # Rescale to true CIELAB coordinates
    L_star = mean_lab[0] * 100.0 / 255.0   # [0, 100]
    b_star = mean_lab[2] - 128.0            # [-128, 127]

    # ITA formula — arctan2 avoids divide-by-zero when b* ≈ 0
    ita = np.degrees(np.arctan2(L_star - 50.0, b_star))

    if ita > 55:
        return "very light skin"
    elif ita > 41:
        return "light skin"
    elif ita > 28:
        return "medium light skin"
    elif ita > 10:
        return "medium skin"
    elif ita > -10:
        return "medium brown skin"
    elif ita > -30:
        return "brown skin"
    elif ita > -50:
        return "dark brown skin"
    else:
        return "very dark skin"
