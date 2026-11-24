#!/usr/bin/env python3
"""Generate crisp anti-aliased Space AI desktop icons matching mark.svg (octahedral prism)."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
ASSETS_DIR = ROOT / "assets"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _scale_pts(pts: list[tuple[float, float]], size: int, pad_frac: float = 0.08) -> list[tuple[float, float]]:
    """Map coordinates from mark.svg's 64x64 viewBox into target pixel canvas."""
    usable = size * (1.0 - 2.0 * pad_frac)
    offset = size * pad_frac
    return [(offset + (x / 64.0) * usable, offset + (y / 64.0) * usable) for x, y in pts]


def _draw_mark_geometry(canvas: Image.Image, pad_frac: float = 0.08, monochrome: bool = False) -> Image.Image:
    s = canvas.width
    lw_outer = max(2, int(round(s * 0.0145)))
    lw_inner = max(2, int(round(s * 0.0115)))

    # Exact frame-0 3D perspective vertices from mark.svg (viewBox 0 0 64 64)
    v_top = (27.6, 7.7)
    v_bot = (35.5, 51.1)
    v_right = (54.1, 28.0)
    v_front = (34.3, 44.7)
    v_left = (9.9, 36.0)
    v_back = (30.5, 24.0)

    # Inner wireframe cube vertices from mark.svg
    c_bl = (21.4, 40.3)
    c_br = (42.6, 40.3)
    c_tr = (42.1, 20.1)
    c_tl = (21.9, 20.1)
    i_bl = (23.5, 42.1)
    i_br = (40.5, 42.1)
    i_tr = (40.2, 25.6)
    i_tl = (23.8, 25.6)

    if monochrome:
        fg = (0, 0, 0, 255)
        fg_mid = (0, 0, 0, 185)
        layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        edges = [
            (v_top, v_right), (v_top, v_front), (v_top, v_left),
            (v_bot, v_right), (v_bot, v_front), (v_bot, v_left),
            (v_left, v_front), (v_front, v_right),
        ]
        for a, b in edges:
            d.line(_scale_pts([a, b], s, pad_frac), fill=fg, width=max(3, int(s * 0.055)))
        for a, b in [(v_top, v_back), (v_left, v_back), (v_right, v_back), (v_bot, v_back)]:
            d.line(_scale_pts([a, b], s, pad_frac), fill=fg_mid, width=max(2, int(s * 0.036)))
        # Center star dot
        cx, cy = _scale_pts([(32.0, 32.0)], s, pad_frac)[0]
        r = s * 0.08
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
        return Image.alpha_composite(canvas, layer)

    # 1. Core radial glow (white -> cyan -> violet, matching #gCore in mark.svg)
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = _scale_pts([(32.0, 32.0)], s, pad_frac)[0]
    for frac, col in (
        (0.28, (168, 85, 247, 72)),
        (0.20, (56, 189, 248, 128)),
        (0.11, (255, 255, 255, 210)),
    ):
        r = s * frac
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=s * 0.055))
    canvas = Image.alpha_composite(canvas, glow)

    # 2. Translucent prismatic faces (matching mark.svg polygons)
    faces = [
        ([v_top, v_left, v_back], (52, 211, 153, 56), (52, 211, 153, 215)),      # emerald/cyan
        ([v_top, v_back, v_right], (251, 191, 36, 56), (251, 191, 36, 215)),     # amber/rose
        ([v_bot, v_back, v_left], (96, 165, 250, 50), (96, 165, 250, 210)),      # cyan/blue
        ([v_bot, v_right, v_back], (45, 212, 191, 50), (45, 212, 191, 210)),     # teal
        ([v_top, v_right, v_front], (56, 189, 248, 44), (56, 189, 248, 240)),    # cyan/blue front-right
        ([v_top, v_front, v_left], (192, 132, 252, 46), (192, 132, 252, 240)),   # violet/pink front-left
        ([v_bot, v_front, v_right], (236, 72, 153, 44), (236, 72, 153, 235)),    # pink bottom-right
        ([v_bot, v_left, v_front], (244, 63, 94, 44), (244, 63, 94, 235)),       # rose bottom-left
    ]
    for poly_pts, fill_col, stroke_col in faces:
        layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pts = _scale_pts(poly_pts, s, pad_frac)
        d.polygon(pts, fill=fill_col)
        d.line(pts + [pts[0]], fill=stroke_col, width=lw_outer, joint="curve")
        canvas = Image.alpha_composite(canvas, layer)

    # 3. Inner crystalline tesseract/cube lines (matching lines in mark.svg)
    inner_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    idraw = ImageDraw.Draw(inner_layer)
    inner_lines = [
        (i_bl, i_br, (255, 255, 255, 145)),
        (i_br, i_tr, (103, 232, 249, 140)),
        (i_tr, i_tl, (232, 121, 249, 135)),
        (i_tl, i_bl, (253, 230, 138, 140)),
        (c_bl, c_br, (255, 255, 255, 220)),
        (c_br, c_tr, (103, 232, 249, 210)),
        (c_tr, c_tl, (232, 121, 249, 200)),
        (c_tl, c_bl, (253, 230, 138, 210)),
        (i_bl, c_bl, (255, 255, 255, 185)),
        (i_br, c_br, (103, 232, 249, 185)),
        (i_tr, c_tr, (232, 121, 249, 175)),
        (i_tl, c_tl, (253, 230, 138, 175)),
    ]
    for a, b, col in inner_lines:
        idraw.line(_scale_pts([a, b], s, pad_frac), fill=col, width=lw_inner)
    canvas = Image.alpha_composite(canvas, inner_layer)

    # 4. Luminous vertex nodes (matching circles in mark.svg)
    nodes_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    nd = ImageDraw.Draw(nodes_layer)
    vertices = [
        (v_top, 2.5, (255, 255, 255, 245)),
        (v_right, 2.3, (56, 189, 248, 235)),
        (v_front, 2.6, (192, 132, 252, 248)),
        (v_left, 2.3, (52, 211, 153, 235)),
        (v_back, 2.0, (251, 191, 36, 210)),
        (v_bot, 2.2, (244, 114, 182, 230)),
    ]
    for pt, rad_units, col in vertices:
        px, py = _scale_pts([pt], s, pad_frac)[0]
        r = (rad_units / 64.0) * s * (1.0 - 2.0 * pad_frac)
        nd.ellipse([px - r, py - r, px + r, py + r], fill=col)
    canvas = Image.alpha_composite(canvas, nodes_layer)

    return canvas


def render_app_icon(target_size: int = 512) -> Image.Image:
    s = target_size * 2
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # Pure-black macOS squircle background
    bg = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    pad = int(s * 0.04)
    radius = int(s * 0.22)
    ImageDraw.Draw(bg).rounded_rectangle([pad, pad, s - pad, s - pad], radius=radius, fill=(5, 5, 8, 255))

    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, s - pad, s - pad], radius=radius, fill=255)
    img.paste(bg, (0, 0), mask)

    # Subtle platinum border
    border = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(border).rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=radius,
        outline=(255, 255, 255, 48),
        width=max(2, int(s * 0.01)),
    )
    img = Image.alpha_composite(img, border)

    # Prismatic Space AI octahedral crystal mark
    mark_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    mark_layer = _draw_mark_geometry(mark_layer, pad_frac=0.13, monochrome=False)
    img = Image.alpha_composite(img, Image.composite(mark_layer, Image.new("RGBA", (s, s), (0, 0, 0, 0)), mask))

    return img.resize((target_size, target_size), Image.Resampling.LANCZOS)


def render_tray_icon(target_size: int, monochrome: bool = False) -> Image.Image:
    s = target_size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    img = _draw_mark_geometry(img, pad_frac=0.06, monochrome=monochrome)
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
    print("Generated prismatic Space AI desktop icons successfully.")


if __name__ == "__main__":
    main()
