I am severely colorblind. I made this color detector in Python to help me naviagte the world and for fun.

---

## What it does

Point your webcam at anything to check the color. The model also learns from corrections over time, so repeated mistakes get fixed permanently.

## Architecture

### Dual identification pipeline

Every Space press launches two threads in parallel:

**Local pipeline (~100–200ms):**
1. YOLOv8n detects what object is in the center reticle
2. Routes to the appropriate analyzer based on object type (see routing below)
3. Result is spoken immediately

**Claude Vision pipeline (~1–2s, optional):**
1. Sends the frame to the Claude Haiku API with YOLO object context injected into the prompt
2. If the result differs from the local result, it overrides and speaks again
3. Disagreements are stored as corrections in persistent memory (see learning below)
4. Silently skipped if no API key or network is unavailable

Stale results from prior Space presses are discarded via a `press_id` counter — no data races.

### Color space: Oklab

All color comparisons are done in **Oklab** (Björn Ottosson, 2020), not HSV. HSV is not perceptually uniform — equal HSV distances do not correspond to equal perceptual differences, which causes warm-toned objects like olive or tan to be misidentified as green.

Oklab is designed so that Euclidean distance equals perceptual difference. The conversion pipeline:

```
BGR uint8
  → sRGB [0,1]  (divide by 255)
  → Linear RGB  (sRGB gamma decode: c ≤ 0.04045 → c/12.92, else ((c+0.055)/1.055)^2.4)
  → XYZ D65     (standard IEC 61966-2-1 matrix)
  → LMS         (Björn Ottosson M1 matrix)
  → LMS^(1/3)   (cube root)
  → Oklab       (Björn Ottosson M2 matrix)
```

### Color naming: XKCD k-NN

954 real-world color names from the [XKCD color survey](https://xkcd.com/color/rgb.txt) are pre-converted to Oklab at import time and indexed in a `scipy.spatial.cKDTree`. Each query is a single nearest-neighbor lookup — O(log n), ~0.1ms.

### Object routing (YOLOv8n)

Detection requires ≥40% confidence and ≥40% overlap with the 160×160px center ROI to filter out partial detections.

| YOLO class | Route |
|---|---|
| `person` (conf ≥ 0.65) | skin detector |
| `tie`, `backpack`, `handbag`, `suitcase`, `umbrella` | general color |
| `cup`, `bottle`, `bowl`, food items | general color |
| anything else | general color |

YOLO has no "shirt" class — shirts are detected as `person`. If YOLO returns `person` but the ROI contains no skin pixels, routing falls back to general color.

### Skin detection

Two-step approach:

**Presence check (`is_skin_region`):** Triple mask — HSV skin range AND YCbCr skin range AND brightness gate (V ≤ 215, to exclude white fabric and pale walls). All three must agree on ≥22% of ROI pixels.

**Tone classification (`get_skin_tone_name`):** Uses the **ITA (Individual Typology Angle)** formula on CIELAB:

```
ITA = arctan2(L* − 50, b*) × (180/π)
```

Maps to 8 levels of the Monk Skin Tone Scale (medically validated, inclusive across all human skin tones).

| ITA | Name |
|---|---|
| > 55° | very light skin |
| 41–55° | light skin |
| 28–40° | medium light skin |
| 10–27° | medium skin |
| −10–9° | medium brown skin |
| −30 to −11° | brown skin |
| −50 to −31° | dark brown skin |
| < −50° | very dark skin |

### Specular highlight rejection

Before computing the ROI mean color, pixels that are both very bright (HSV V > 200) and near-colorless (HSV S < 40) are masked out. This removes light-source glare without affecting legitimate bright colors like yellow (high V but also high S) or white objects (uniformly bright, not glare-spotted). If >80% of pixels are masked, the mask is discarded and the full mean is used — so white objects are still identified correctly.

### Online learning (persistent Oklab k-NN memory)

Every identification is stored in `~/.colorblindcam/memory.json` as an `(Oklab coords, color name, count)` sample. On the next identification, memory is checked first: if a trusted nearby sample exists, its label is returned without running the full pipeline.

**Claude acts as teacher:** When Claude disagrees with the local result, the correction is stored automatically.

**Keys:**
- `Y` — confirm the result is correct (reinforces the sample)
- `N` — mark the result as wrong (stores a rejection for that Oklab region)

A sample is trusted after ≥2 confirmations. A rejected region is remembered permanently and will not be identified as that color again. If a skin tone is rejected (e.g. a wall being called "medium skin"), future identifications of that region bypass skin routing entirely.

Rejections can be cleared by pressing Y if the model later gets the same region right.

## Controls

| Key | Action |
|---|---|
| `Space` | Identify color under reticle |
| `Y` | Confirm result is correct |
| `N` | Mark result as wrong |
| `Q` | Quit |

Volume and speech rate are adjustable via trackbars in the window.

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
pyttsx3>=2.90
ultralytics>=8.0.0
anthropic>=0.25.0
```

```bash
pip install -r requirements.txt
```

The YOLOv8n model (~6MB) downloads automatically on first run.

Claude Vision requires an `ANTHROPIC_API_KEY` environment variable. The app works without it — local identification still runs.

## File overview

| File | Role |
|---|---|
| `main.py` | Entry point |
| `color_detector.py` | Main loop, threading, UI, pipeline orchestration |
| `oklab_namer.py` | Oklab conversion + XKCD k-NN color naming |
| `skin_detector.py` | Skin presence detection + ITA skin tone classification |
| `object_detector.py` | YOLOv8n wrapper with routing logic |
| `vision_identifier.py` | Claude Vision API integration |
| `color_memory.py` | Persistent online learning database |
| `voice_output.py` | Text-to-speech with volume/rate control |
| `color_namer.py` | Legacy HSV fallback (used only if Oklab fails) |
| `xkcd_colors.py` | 954 XKCD color name → hex mappings |
