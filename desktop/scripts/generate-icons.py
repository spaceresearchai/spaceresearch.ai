#!/usr/bin/env python3
"""Generate crisp anti-aliased Space AI desktop icons (PNG, ICO, and macOS tray template)."""

from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
ASSETS_DIR = ROOT / "assets"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def draw_rotated_ellipse(
    size: int,
    rx: float,
    ry: float,
    angle_deg: float,
    color: tuple[int, int, int, int],
    width: int,
) -> Image.Image:
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = cy = size / 2.0
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, outline=color, width=width)
    return layer.rotate(angle_deg, resample=Image.Resampling.BICUBIC)


def render_app_icon(target_size: int = 512) -> Image.Image:
    scale = 2
    s = target_size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # Rounded macOS-style obsidian squircle background
    bg = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    pad = int(s * 0.04)
    radius = int(s * 0.22)
    bg_draw.rounded_rectangle([pad, pad, s - pad, s - pad], radius=radius, fill=(8, 8, 11, 255))

    # Subtle radial warm-brass core glow inside the squircle
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow)
    cx = cy = s / 2.0
    for r_frac, alpha in ((0.38, 24), (0.26, 38), (0.15, 62)):
        r = s * r_frac
        g_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(226, 176, 104, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=s * 0.06))

    # Mask glow to squircle
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, s - pad, s - pad], radius=radius, fill=255)
    img.paste(bg, (0, 0), mask)
    img = Image.alpha_composite(img, Image.composite(glow, Image.new("RGBA", (s, s), (0, 0, 0, 0)), mask))

    # Subtle metallic border
    border = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(border).rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=radius,
        outline=(226, 176, 104, 115),
        width=max(2, int(s * 0.012)),
    )
    img = Image.alpha_composite(img, border)

    # Subtle starlight dots
    stars_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(stars_layer)
    for fx, fy, fr, fa in (
        (0.22, 0.24, 0.008, 160),
        (0.79, 0.21, 0.009, 190),
        (0.19, 0.75, 0.007, 130),
        (0.81, 0.77, 0.008, 170),
        (0.68, 0.30, 0.006, 120),
        (0.30, 0.71, 0.006, 125),
    ):
        px, py, pr = s * fx, s * fy, s * fr
        s_draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(244, 244, 248, fa))
    img = Image.alpha_composite(img, stars_layer)

    # Space AI Dual Orbital Rings (warm brass)
    ring_b = draw_rotated_ellipse(
        s,
        rx=s * 0.32,
        ry=s * 0.13,
        angle_deg=-52,
        color=(226, 176, 104, 145),
        width=max(4, int(s * 0.022)),
    )
    ring_a = draw_rotated_ellipse(
        s,
        rx=s * 0.34,
        ry=s * 0.136,
        angle_deg=24,
        color=(238, 194, 126, 242),
        width=max(5, int(s * 0.027)),
    )
    img = Image.alpha_composite(img, ring_b)
    img = Image.alpha_composite(img, ring_a)

    # Center luminous brass-white core
    core = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    c_draw = ImageDraw.Draw(core)
    r_outer = s * 0.095
    c_draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=(226, 176, 104, 255))
    r_inner = s * 0.052
    c_draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(255, 247, 232, 255))
    img = Image.alpha_composite(img, core)

    return img.resize((target_size, target_size), Image.Resampling.LANCZOS)


def render_tray_icon(target_size: int, monochrome: bool = False) -> Image.Image:
    s = target_size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    fg = (0, 0, 0, 255) if monochrome else (226, 176, 104, 255)
    fg_soft = (0, 0, 0, 155) if monochrome else (226, 176, 104, 165)

    ring_b = draw_rotated_ellipse(s, rx=s * 0.40, ry=s * 0.16, angle_deg=-52, color=fg_soft, width=max(4, int(s * 0.065)))
    ring_a = draw_rotated_ellipse(s, rx=s * 0.42, ry=s * 0.17, angle_deg=24, color=fg, width=max(5, int(s * 0.075)))
    img = Image.alpha_composite(img, ring_b)
    img = Image.alpha_composite(img, ring_a)

    core = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    cx = cy = s / 2.0
    r = s * 0.135
    ImageDraw.Draw(core).ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    img = Image.alpha_composite(img, core)

    return img.resize((target_size, target_size), Image.Resampling.LANCZOS)


def main() -> None:
    icon_512 = render_app_icon(512)
    icon_512.save(BUILD_DIR / "icon.png", format="PNG")
    icon_512.save(
        BUILD_DIR / "icon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    icon_512.resize((128, 128), Image.Resampling.LANCZOS).save(ASSETS_DIR / "icon-128.png", format="PNG")

    render_tray_icon(32, monochrome=False).save(ASSETS_DIR / "tray-icon.png", format="PNG")
    render_tray_icon(22, monochrome=True).save(ASSETS_DIR / "tray-iconTemplate.png", format="PNG")
    render_tray_icon(44, monochrome=True).save(ASSETS_DIR / "tray-iconTemplate@2x.png", format="PNG")
    print("Generated Space AI desktop icons successfully.")


if __name__ == "__main__":
    main()
