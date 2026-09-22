"""Generate XCal logo candidates in the mbirtorch/mbirjax house style:
bold sans wordmark, black base, colored accent, glossy reflection.

The logos are written into docs/source/_static/, where the docs use
them, so running this script updates the real assets in place.
"""

import os

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# The docs static directory, resolved from this script's own location so
# the output goes to the same place no matter where the script is run.
STATIC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "docs", "source", "_static"))

FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 260)
CANVAS = (1500, 560)
BASELINE_Y = 60


def text_layer(text, color_fn):
    """Render text; color_fn(mask_size) returns a fill image."""
    img = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    x = 60
    for piece, color in text:
        mask = Image.new("L", CANVAS, 0)
        d = ImageDraw.Draw(mask)
        d.text((x, BASELINE_Y), piece, font=FONT, fill=255)
        bbox = d.textbbox((x, BASELINE_Y), piece, font=FONT)
        fill = color_fn(color)
        img.paste(fill, (0, 0), mask)
        x = bbox[2] + 6
    return img


def solid(color):
    return Image.new("RGBA", CANVAS, color)


def vertical_gradient(stops):
    """stops: list of (frac, (r,g,b)) top to bottom over the glyph band."""
    img = Image.new("RGBA", CANVAS)
    top, bottom = BASELINE_Y + 30, BASELINE_Y + 260
    px = img.load()
    for y in range(CANVAS[1]):
        f = min(max((y - top) / (bottom - top), 0.0), 1.0)
        for i in range(len(stops) - 1):
            f0, c0 = stops[i]
            f1, c1 = stops[i + 1]
            if f0 <= f <= f1:
                t = (f - f0) / (f1 - f0)
                c = tuple(int(c0[k] + t * (c1[k] - c0[k])) for k in range(3))
                break
        for x in range(CANVAS[0]):
            px[x, y] = c + (255,)
    return img


def add_reflection(img):
    """Mirror the wordmark below with a fading, slightly blurred copy."""
    from PIL import ImageChops
    bbox = img.getbbox()
    word = img.crop(bbox)
    refl = word.transpose(Image.FLIP_TOP_BOTTOM)
    refl = refl.resize((refl.width, int(refl.height * 0.85)))
    refl = refl.filter(ImageFilter.GaussianBlur(1.5))
    fade = Image.new("L", refl.size, 0)
    fp = fade.load()
    for y in range(refl.height):
        a = max(0, int(150 * (1 - y / (refl.height * 0.85))))
        for x in range(refl.width):
            fp[x, y] = a
    a = refl.split()[3]
    refl.putalpha(ImageChops.multiply(a, fade))
    margin = 40
    out = Image.new("RGBA",
                    (word.width + 2 * margin,
                     word.height + refl.height + margin + 20),
                    (0, 0, 0, 0))
    out.paste(word, (margin, 10), word)
    out.paste(refl, (margin, 10 + word.height + 8), refl)
    return out


BLACK = (20, 20, 20)
LIGHT = (235, 235, 235)
TORCH_RED = (238, 42, 74)

# X in torch red ties xcal to the mbirtorch family.  "Cal" is dark
# for the light-background logo and light for the dark-background
# logo, so it stays visible in both documentation themes.
variants = {
    "logo": [("X", ("solid", TORCH_RED)), ("Cal", ("solid", BLACK))],
    "logo_dark": [("X", ("solid", TORCH_RED)),
                  ("Cal", ("solid", LIGHT))],
}

for name, spec in variants.items():
    def color_fn(color):
        kind, val = color
        if kind == "solid":
            return solid(val + (255,))
        return vertical_gradient(val)
    img = text_layer(spec, color_fn)
    img = add_reflection(img)
    out = os.path.join(STATIC, f"{name}.png")
    img.save(out)
    print("wrote", out)
