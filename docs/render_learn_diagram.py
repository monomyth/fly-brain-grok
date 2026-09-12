#!/usr/bin/env python3
"""Explanatory diagram of Learn-mode nets. Exact labels via Pillow."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 980
BG = (18, 22, 26)
CARD = (28, 34, 40)
CARD2 = (36, 44, 52)
LIME = (180, 220, 70)
CYAN = (80, 200, 210)
GOLD = (240, 170, 50)
WHITE = (236, 238, 240)
MUTED = (150, 158, 166)
RED = (220, 90, 80)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box, fill, r=18):
    draw.rounded_rectangle(box, radius=r, fill=fill)


def arrow(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, color=LIME, w=4):
    draw.line((x0, y0, x1, y1), fill=color, width=w)
    # head
    if abs(x1 - x0) >= abs(y1 - y0):
        s = 1 if x1 > x0 else -1
        draw.polygon([(x1, y1), (x1 - 14 * s, y1 - 8), (x1 - 14 * s, y1 + 8)], fill=color)
    else:
        s = 1 if y1 > y0 else -1
        draw.polygon([(x1, y1), (x1 - 8, y1 - 14 * s), (x1 + 8, y1 - 14 * s)], fill=color)


def center_text(draw, xy, text, f, fill=WHITE):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x - tw / 2, y - th / 2), text, font=f, fill=fill)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title = font(36, True)
    h1 = font(22, True)
    body = font(16)
    small = font(14)
    tiny = font(12)

    d.text((56, 36), "Learn mode — what is trained", font=title, fill=WHITE)
    d.text(
        (56, 86),
        "Scripted arm pick. The fly wiring stays frozen except 4,184 existing KC → MBON11 synapses.",
        font=body,
        fill=MUTED,
    )

    # Camera
    rounded(d, (56, 150, 280, 340), CARD)
    center_text(d, (168, 185), "Eyes", h1, CYAN)
    d.rectangle((96, 210, 240, 300), outline=CYAN, width=2)
    d.rectangle((120, 230, 160, 270), fill=(60, 80, 90))
    d.rectangle((176, 230, 216, 270), fill=(200, 90, 40))
    center_text(d, (168, 318), "Top JPEG 160×120", small, MUTED)

    arrow(d, 280, 245, 340, 245)

    # Photoreceptors
    rounded(d, (340, 150, 620, 340), CARD)
    center_text(d, (480, 180), "Optical neurons", h1, CYAN)
    center_text(d, (480, 220), "R1–R6  ~3,377", body, WHITE)
    center_text(d, (480, 248), "R8  chroma", body, WHITE)
    center_text(d, (480, 286), "FROZEN", h1, MUTED)
    center_text(d, (480, 318), "brightness grid, not fly optics", tiny, MUTED)

    arrow(d, 620, 245, 680, 245)

    # Frozen CNS
    rounded(d, (680, 130, 1120, 360), CARD2)
    center_text(d, (900, 165), "MaleCNS v1.0  (frozen)", h1, WHITE)
    center_text(d, (900, 210), "166,700 LIF cells", body, WHITE)
    center_text(d, (900, 240), "25,582,938 signed synapses", body, WHITE)
    center_text(d, (900, 280), "Not trained. Not backprop.", body, MUTED)
    center_text(d, (900, 320), "DNs exist but do not move the arm in Learn.", small, MUTED)

    # KC row
    rounded(d, (56, 420, 520, 640), CARD)
    center_text(d, (288, 450), "Kenyon cells (KC)", h1, GOLD)
    center_text(d, (288, 490), "Each KC sees one cell of an 8×8 luma grid", body, WHITE)
    # mini grid
    gx0, gy0 = 168, 520
    for i in range(8):
        for j in range(8):
            v = 40 + ((i * 8 + j) % 5) * 30
            d.rectangle((gx0 + j * 14, gy0 + i * 10, gx0 + j * 14 + 12, gy0 + i * 10 + 8), fill=(v, int(v * 0.7), 40))
    center_text(d, (400, 560), "Engineered PN→KC drive", small, MUTED)
    center_text(d, (400, 588), "(16 hops from R1 never reach KC)", tiny, MUTED)
    center_text(d, (288, 620), "Spatial pattern — not one mean current", small, GOLD)

    arrow(d, 520, 530, 600, 530)

    # Plastic synapses
    rounded(d, (600, 400, 980, 660), (48, 38, 22))
    d.rounded_rectangle((600, 400, 980, 660), radius=18, outline=GOLD, width=3)
    center_text(d, (790, 435), "THE ONLY TRAINED NET", h1, GOLD)
    center_text(d, (790, 480), "4,184 existing KC → MBON11", body, WHITE)
    center_text(d, (790, 512), "synapses  (already in MaleCNS)", body, WHITE)
    # dots
    for i in range(6):
        d.ellipse((680 + i * 36, 545, 696 + i * 36, 561), fill=GOLD)
        d.ellipse((692 + i * 36, 580, 704 + i * 36, 592), fill=CYAN)
        d.line((688 + i * 36, 561, 698 + i * 36, 580), fill=WHITE, width=2)
    center_text(d, (790, 630), "Hebbian × dopamine gate", small, GOLD)

    arrow(d, 980, 530, 1060, 530)

    # MBON / DA
    rounded(d, (1060, 400, 1544, 560), CARD)
    center_text(d, (1302, 435), "MBON11  +  PPL101", h1, CYAN)
    center_text(d, (1302, 478), "Mushroom-body output  ·  dopamine", body, WHITE)
    center_text(d, (1302, 518), "PPL101 pulsed when the SCRIPTED pick is correct", small, MUTED)

    rounded(d, (1060, 580, 1544, 660), (42, 32, 28))
    center_text(d, (1302, 620), "Reward: hover · pads · attach · lift · hold", body, GOLD)

    # Arm - not trained
    rounded(d, (56, 700, 1544, 920), CARD)
    center_text(d, (800, 735), "Arm  —  not trained by the brain", h1, LIME)
    steps = [
        ("1 Folded", "idle"),
        ("2 Ready", "unfold"),
        ("3 Hover", "z = 48 mm"),
        ("4 Pads", "+35 mm tool X"),
        ("5 Close", "grip 20 mm"),
        ("6 Lift", "z = 160 mm"),
        ("7 Hold 5 s", "then open"),
    ]
    for i, (a, b) in enumerate(steps):
        x = 90 + i * 205
        rounded(d, (x, 770, x + 190, 880), CARD2, r=12)
        center_text(d, (x + 95, 805), a, h1, LIME)
        center_text(d, (x + 95, 845), b, small, MUTED)
        if i < len(steps) - 1:
            arrow(d, x + 190, 825, x + 205, 825, LIME, 3)

    d.text((56, 938), "Grok ReBot lab  ·  MaleCNS v1.0 frozen  ·  Learn = scripted pick + visual KC + DA on KC→MBON11", font=tiny, fill=MUTED)

    out = Path(__file__).with_name("learn-network.png")
    img.save(out, "PNG")
    print(out)


if __name__ == "__main__":
    main()
