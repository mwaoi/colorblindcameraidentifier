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
    Requires HSV and YCbCr masks to agree on at least 15% of pixels.
    """
    if roi_bgr.size == 0:
        return False

    # HSV skin range (empirically validated across diverse skin tones)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 15, 50], dtype=np.uint8),
        np.array([25, 180, 255], dtype=np.uint8),
    )

    # YCbCr skin range (illumination-robust)
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    skin_fraction = np.count_nonzero(combined) / combined.size
    return skin_fraction >= 0.15


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
