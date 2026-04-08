"""
Perceptual color naming using the XKCD color dataset and Oklab color space.

Oklab (Björn Ottosson, 2020) is designed so that Euclidean distance equals
perceptual difference — unlike HSV, where equal distances are not equally
perceptible. This eliminates false "forest green" identifications for olive/
warm-toned colors that lie near the green cluster in HSV space.

Pipeline: BGR uint8 → sRGB [0,1] → Linear RGB → XYZ (D65) → LMS → Oklab
"""

import numpy as np
from scipy.spatial import cKDTree

from xkcd_colors import XKCD_COLORS

# ---------- Oklab conversion matrices ----------

# Standard D65 illuminant RGB → XYZ matrix (IEC 61966-2-1)
_M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# Björn Ottosson's M1: XYZ → LMS
_M_XYZ_TO_LMS = np.array([
    [ 0.8189330101,  0.3618667424, -0.1288597137],
    [ 0.0329845436,  0.9293118715,  0.0361456387],
    [ 0.0482003018,  0.2643662691,  0.6338517070],
], dtype=np.float64)

# Björn Ottosson's M2: LMS_ → Lab
_M_LMS_TO_LAB = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float64)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Gamma-decode sRGB [0,1] to linear light values."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _rgb_to_oklab(r: float, g: float, b: float) -> np.ndarray:
    """Convert a single linear-light RGB triple (0–1) to Oklab [L, a, b]."""
    srgb = np.array([r, g, b], dtype=np.float64)
    linear = _srgb_to_linear(srgb)
    xyz = _M_RGB_TO_XYZ @ linear
    lms = _M_XYZ_TO_LMS @ xyz
    lms_ = np.cbrt(lms)
    return _M_LMS_TO_LAB @ lms_


def _hex_to_oklab_batch(hex_list: list[str]) -> np.ndarray:
    """Convert a list of '#rrggbb' hex strings to Oklab coordinates, shape (N, 3)."""
    srgb = np.array(
        [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex_list],
        dtype=np.float64,
    ) / 255.0  # shape (N, 3), RGB order
    linear = _srgb_to_linear(srgb)          # (N, 3)
    xyz = linear @ _M_RGB_TO_XYZ.T          # (N, 3)
    lms = xyz @ _M_XYZ_TO_LMS.T            # (N, 3)
    lms_ = np.cbrt(lms)                     # (N, 3)
    return lms_ @ _M_LMS_TO_LAB.T          # (N, 3)  → Oklab


# ---------- Module-level init (runs once on import) ----------

_NAMES: list[str] = list(XKCD_COLORS.keys())
_HEX: list[str] = list(XKCD_COLORS.values())
_LAB_ARRAY: np.ndarray = _hex_to_oklab_batch(_HEX)   # shape (949, 3)
_TREE: cKDTree = cKDTree(_LAB_ARRAY)


# ---------- Public API ----------

def rgb_to_oklab(r: int, g: int, b: int) -> np.ndarray:
    """Convert uint8 RGB (0–255 each) to Oklab [L, a, b] coordinates."""
    return _rgb_to_oklab(r / 255.0, g / 255.0, b / 255.0)


def get_oklab_color_name(r: int, g: int, b: int) -> str:
    """
    Return the XKCD color name perceptually closest to the given RGB (0–255 each).
    Uses nearest-neighbor search in Oklab space.
    """
    lab = _rgb_to_oklab(r / 255.0, g / 255.0, b / 255.0)
    _, idx = _TREE.query(lab)
    return _NAMES[idx]
