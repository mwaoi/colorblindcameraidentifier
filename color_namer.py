import colorsys

ACHROMATIC_S_THRESHOLD = 0.20


def get_color_name(r: int, g: int, b: int) -> str:
    """Return a real-world color name for the given RGB values (0-255 each)."""
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_f, g_f, b_f)
    h_deg = h * 360.0

    # Very dark: hue is meaningless under tinted lighting
    if v < 0.20:
        return "black" if v < 0.10 else "charcoal"

    # Achromatic (low saturation)
    if s < ACHROMATIC_S_THRESHOLD:
        if v < 0.35:
            return "dark gray"
        if v < 0.60:
            return "gray"
        if v < 0.80:
            return "silver"
        return "white"

    # --- Chromatic decision tree ---

    if h_deg < 10 or h_deg >= 350:   # red family
        if v < 0.45:
            return "maroon"
        if v > 0.70 and s < 0.65:
            return "coral"
        return "red"

    if h_deg < 35:                    # orange / brown / peach
        if v < 0.25:
            return "dark brown"
        if v < 0.45:
            return "brown"
        if v > 0.80 and s < 0.50:
            return "peach"
        return "orange"

    if h_deg < 65:                    # yellow / olive / gold
        if v < 0.50:
            return "olive"
        if v < 0.80 and s > 0.60:
            return "gold"
        return "yellow"

    if h_deg < 90:                    # lime / yellow-green
        if v < 0.45:
            return "olive"
        return "lime"

    if h_deg < 150:                   # green family
        if s < 0.30:                  # low-sat green hue = olive, not forest green
            return "dark green" if v < 0.35 else "olive"
        if v < 0.30:
            return "dark green"
        if v < 0.55:
            return "forest green"
        return "green"

    if h_deg < 195:                   # teal / turquoise
        if v > 0.55 and s > 0.55:
            return "turquoise"
        return "teal"

    if h_deg < 225:                   # sky blue
        return "sky blue"

    if h_deg < 265:                   # blue / navy
        if v < 0.35:
            return "navy"
        if v < 0.60:
            return "dark blue"
        return "blue"

    if h_deg < 285:                   # indigo / purple
        return "indigo" if v < 0.45 else "purple"

    if h_deg < 330:                   # purple / lavender / magenta
        if v > 0.75 and s < 0.40:
            return "lavender"
        if v > 0.65 and s > 0.55:
            return "magenta"
        return "purple"

    # 330-350: hot pink / pink / maroon
    if v > 0.70 and s > 0.50:
        return "hot pink"
    if v > 0.50:
        return "pink"
    return "maroon"
