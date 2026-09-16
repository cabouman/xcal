"""Generate the xcal social-preview card (Open Graph / GitHub).

The card is 1280x640, the size GitHub and the Open Graph protocol both
accept, so one image serves the GitHub repository preview and the
documentation link preview.  It shows the xCal wordmark in the
mbirtorch/mbirjax house style over a glowing X-ray tube spectrum, the
quantity xcal estimates.  The image renders at twice the final size and
is downsampled once, which anti-aliases the text and the curve.

The output is written to docs/source/_static/og_card.png, the location
docs/source/conf.py points the og:image meta tag at.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SCALE = 2                                   # supersample, then downsample
W, H = 1280 * SCALE, 640 * SCALE

TORCH_RED = (238, 42, 74)
LIGHT = (238, 238, 240)
MUTED = (150, 152, 160)
BG_TOP = (17, 18, 22)
BG_BOTTOM = (9, 9, 12)

WORD_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 300 * SCALE)
TAG_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial.ttf", 52 * SCALE)
URL_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34 * SCALE)


def background():
    """Returns the dark vertical-gradient background."""
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        f = y / (H - 1)
        c = tuple(int(BG_TOP[k] + f * (BG_BOTTOM[k] - BG_TOP[k]))
                  for k in range(3))
        for x in range(W):
            px[x, y] = c
    return img


def spectrum_curve(n):
    """Returns n samples of a synthetic X-ray tube spectrum in [0, 1]:
    a bremsstrahlung continuum shaped by a low-energy filter cutoff,
    with two characteristic emission lines."""
    e = np.linspace(0.0, 1.0, n)
    e_max = 0.92
    continuum = np.clip(e_max - e, 0, None) * (1 - np.exp(-e / 0.09))
    lines = (0.55 * np.exp(-((e - 0.58) / 0.012) ** 2)
             + 0.9 * np.exp(-((e - 0.64) / 0.012) ** 2))
    y = continuum / continuum.max() * 0.72 + lines
    return np.clip(y / y.max(), 0, 1)


def draw_spectrum(img):
    """Draws the glowing spectrum as a bottom skyline: a soft red fill
    under the curve plus a bright stroke, blurred for a glow."""
    left, right = 70 * SCALE, W - 70 * SCALE
    base_y = H - 34 * SCALE
    top_y = 448 * SCALE
    n = right - left
    y = spectrum_curve(n)
    xs = np.arange(left, right)
    ys = base_y - y * (base_y - top_y)

    # Filled area under the curve, red fading downward.
    fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill)
    for i in range(n):
        x = int(xs[i])
        yt = int(ys[i])
        for yy in range(yt, base_y):
            f = (yy - yt) / max(base_y - yt, 1)
            a = int(150 * (1 - f) ** 1.6)
            fd.point((x, yy), TORCH_RED + (a,))
    fill = fill.filter(ImageFilter.GaussianBlur(3 * SCALE))
    img.paste(fill, (0, 0), fill)

    # Bright stroke with a blurred copy behind it for the glow.
    stroke = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stroke)
    pts = list(zip(xs.tolist(), ys.tolist()))
    sd.line(pts, fill=(255, 120, 140, 255), width=5 * SCALE, joint="curve")
    glow = stroke.filter(ImageFilter.GaussianBlur(7 * SCALE))
    img.paste(glow, (0, 0), glow)
    img.paste(stroke, (0, 0), stroke)


def draw_wordmark(img):
    """Draws 'XCal' centered in the upper area: X in torch red, Cal in
    light, with a soft red glow behind the whole word."""
    pieces = [("X", TORCH_RED), ("Cal", LIGHT)]
    d = ImageDraw.Draw(img)
    widths = [d.textbbox((0, 0), t, font=WORD_FONT)[2] for t, _ in pieces]
    total = sum(widths)
    x = (W - total) // 2
    y = 62 * SCALE

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text((x, y), "XCal", font=WORD_FONT, fill=TORCH_RED + (110,))
    glow = glow.filter(ImageFilter.GaussianBlur(18 * SCALE))
    img.paste(glow, (0, 0), glow)

    for (t, color), w in zip(pieces, widths):
        d.text((x, y), t, font=WORD_FONT, fill=color)
        x += w


def draw_text(img):
    """Draws the tagline and the documentation URL, both centered."""
    d = ImageDraw.Draw(img)
    tag = "Model-based X-ray CT spectral calibration"
    tw = d.textbbox((0, 0), tag, font=TAG_FONT)[2]
    d.text(((W - tw) // 2, 372 * SCALE), tag, font=TAG_FONT, fill=MUTED)

    url = "xcal.readthedocs.io"
    uw = d.textbbox((0, 0), url, font=URL_FONT)[2]
    d.text((W - 70 * SCALE - uw, 556 * SCALE), url, font=URL_FONT,
           fill=TORCH_RED)


def main():
    img = background().convert("RGBA")
    draw_spectrum(img)
    draw_wordmark(img)
    draw_text(img)
    img = img.convert("RGB").resize((W // SCALE, H // SCALE),
                                    Image.LANCZOS)
    out = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "..", "docs", "source", "_static", "og_card.png")
    img.save(os.path.normpath(out))
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
