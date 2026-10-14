import html
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "figures"
SVG_PATH = OUTPUT_DIR / "etg_intro_motivation.svg"
PNG_PATH = OUTPUT_DIR / "etg_intro_motivation.png"

WIDTH = 2200
HEIGHT = 820

BLUE = "#6FA3EF"
BLUE_DARK = "#456990"
ORANGE = "#F6B66B"
ORANGE_DARK = "#C9842B"
GREEN = "#67B87F"
RED = "#E46F6F"
GRAY = "#C9CED6"
GRAY_DARK = "#6F7782"
TEAL = "#58B7B2"
PANEL_BG = "#F7F9FC"
EDGE = "#5C6470"
TEXT = "#2C3440"
LIGHT_BLUE_BG = "#EDF4FF"
LIGHT_ORANGE_BG = "#FFF5EA"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def rect(x, y, w, h, fill, stroke=EDGE, sw=2, rx=18, dashed=False):
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>'
    )


def circle(cx, cy, r, fill, stroke="none", sw=0):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def line(x1, y1, x2, y2, stroke, sw=3, dashed=False, marker_end=False, opacity=1.0):
    dash = ' stroke-dasharray="8 8"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if marker_end else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" stroke-linecap="round"{dash}{marker} opacity="{opacity}"/>'
    )


def path(d, stroke, sw=3, fill="none", marker_end=False, opacity=1.0):
    marker = ' marker-end="url(#arrow)"' if marker_end else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round"{marker} opacity="{opacity}"/>'
    )


def text(x, y, s, size=28, fill=TEXT, weight="400", anchor="start", family="Arial, Helvetica, sans-serif"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'font-family="{family}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def multiline_text(x, y, lines, size=26, fill=TEXT, weight="400", anchor="start", line_gap=1.25):
    tspans = []
    dy = size * line_gap
    for idx, line_text in enumerate(lines):
        tspans.append(
            f'<tspan x="{x}" y="{y + idx * dy}">{esc(line_text)}</tspan>'
        )
    return (
        f'<text font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}">'
        + "".join(tspans)
        + "</text>"
    )


def people_group(x, y, color, scale=1.0, n=3):
    parts = []
    spacing = 40 * scale
    for idx in range(n):
        cx = x + idx * spacing
        parts.append(circle(cx, y, 12 * scale, color))
        parts.append(
            rect(
                cx - 13 * scale,
                y + 12 * scale,
                26 * scale,
                26 * scale,
                color,
                stroke="none",
                sw=0,
                rx=8 * scale,
            )
        )
    return "".join(parts)


def mini_gain_chart(x, y, w, values):
    parts = [line(x, y, x + w, y, "#AEB6C2", sw=2, dashed=True)]
    n = len(values)
    gap = w / (n * 1.55 + 0.3)
    bar_w = gap * 0.82
    scale = 72
    for idx, value in enumerate(values):
        bx = x + gap * (0.25 + idx * 1.55)
        if value > 0.08:
            color = GREEN
        elif value < -0.08:
            color = RED
        else:
            color = GRAY
        if value >= 0:
            bh = max(value * scale, 7)
            by = y - bh
        else:
            bh = max(-value * scale, 7)
            by = y
        parts.append(rect(bx, by, bar_w, bh, color, stroke="white", sw=1, rx=4))
    return "".join(parts)


def source_nodes(x, y):
    parts = []
    nodes = [
        (x + 0, y + 12),
        (x + 52, y - 20),
        (x + 98, y + 18),
        (x + 76, y + 54),
    ]
    edges = [(0, 1), (1, 2), (1, 3)]
    for i, j in edges:
        parts.append(line(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1], "#8FB6F3", sw=4, opacity=0.95))
    for nx, ny in nodes:
        parts.append(circle(nx, ny, 14, BLUE, stroke="white", sw=2))
    return "".join(parts)


def badge(x, y, w, h, label, fill, stroke):
    return rect(x, y, w, h, fill, stroke=stroke, sw=2, rx=18) + text(x + w / 2, y + h * 0.62, label, size=21, anchor="middle")


def build_svg():
    left_x, panel_y, panel_w, panel_h = 20, 20, 1050, 780
    right_x = 1130
    parts = []

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    parts.append(
        """
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#456990"/>
          </marker>
        </defs>
        """
    )
    parts.append(rect(0, 0, WIDTH, HEIGHT, "white", stroke="none", sw=0, rx=0))

    # Left panel
    parts.append(rect(left_x, panel_y, panel_w, panel_h, PANEL_BG, stroke="#D7DEE8", sw=2, rx=28))
    parts.append(text(left_x + 30, 82, "Standard CDR: transfer gains are uneven", size=34, weight="700"))
    parts.append(text(left_x + 30, 124, "One-shot transfer improves some users much more than others.", size=22, fill=GRAY_DARK))

    parts.append(rect(left_x + 35, 220, 220, 330, LIGHT_BLUE_BG, stroke="#C7D8F6", sw=2, rx=22))
    parts.append(multiline_text(left_x + 145, 302, ["Source", "Domain"], size=31, fill=BLUE_DARK, weight="700", anchor="middle"))
    parts.append(multiline_text(left_x + 145, 394, ["rich auxiliary", "knowledge"], size=22, fill=GRAY_DARK, anchor="middle"))
    parts.append(source_nodes(left_x + 88, 455))

    parts.append(line(left_x + 258, 385, left_x + 405, 385, BLUE_DARK, sw=7, marker_end=True))
    parts.append(text(left_x + 288, 345, "standard transfer", size=22, fill=BLUE_DARK))

    parts.append(rect(left_x + 410, 145, 600, 520, "white", stroke="#D7DEE8", sw=2, rx=22))
    parts.append(text(left_x + 710, 215, "Target-domain users", size=29, weight="700", anchor="middle"))

    parts.append(rect(left_x + 450, 310, 500, 145, LIGHT_BLUE_BG, stroke="#D0DEF5", sw=2, rx=18))
    parts.append(people_group(left_x + 505, 365, BLUE_DARK, scale=1.0, n=3))
    parts.append(text(left_x + 620, 360, "Overlapping users", size=26, fill=BLUE_DARK, weight="700"))
    parts.append(text(left_x + 620, 398, "mostly receive helpful transfer", size=21, fill=GRAY_DARK))
    parts.append(text(left_x + 845, 330, "transfer gain", size=20, fill=GRAY_DARK, anchor="middle"))
    parts.append(mini_gain_chart(left_x + 782, 392, 118, [0.78, 0.64, 0.88, 0.58]))

    parts.append(rect(left_x + 450, 485, 500, 145, LIGHT_ORANGE_BG, stroke="#F1D7B7", sw=2, rx=18))
    parts.append(people_group(left_x + 505, 540, ORANGE_DARK, scale=1.0, n=3))
    parts.append(text(left_x + 620, 535, "Non-overlapping users", size=26, fill=ORANGE_DARK, weight="700"))
    parts.append(text(left_x + 620, 573, "limited gain with neutral / harmful cases", size=21, fill=GRAY_DARK))
    parts.append(text(left_x + 845, 505, "transfer gain", size=20, fill=GRAY_DARK, anchor="middle"))
    parts.append(mini_gain_chart(left_x + 782, 567, 118, [0.18, -0.26, 0.05, -0.16]))

    parts.append(rect(left_x + 515, 690, 370, 65, "#FFF0F0", stroke="#F1B9B9", sw=2, rx=18))
    parts.append(multiline_text(left_x + 700, 717, ["Transfer benefits concentrate on overlap users,", "leaving more harmful transfer on under-benefited users."], size=18, fill="#A24B4B", anchor="middle"))

    # Right panel
    parts.append(rect(right_x, panel_y, panel_w, panel_h, PANEL_BG, stroke="#D7DEE8", sw=2, rx=28))
    parts.append(text(right_x + 30, 82, "ETG: close the intervention gap", size=34, weight="700"))
    parts.append(text(right_x + 30, 124, "Prediction-level gating keeps helpful transfer and suppresses harmful transfer.", size=22, fill=GRAY_DARK))

    parts.append(rect(right_x + 40, 255, 150, 125, LIGHT_BLUE_BG, stroke="#C7D8F6", sw=2, rx=18))
    parts.append(multiline_text(right_x + 115, 310, ["Source", "signal"], size=26, fill=BLUE_DARK, weight="700", anchor="middle"))

    parts.append(rect(right_x + 40, 470, 150, 125, "#F5F7FA", stroke="#D5DCE6", sw=2, rx=18))
    parts.append(multiline_text(right_x + 115, 525, ["Target pair", "(u, i)"], size=24, fill=TEXT, weight="700", anchor="middle"))

    parts.append(rect(right_x + 275, 265, 190, 110, "#EEF5FF", stroke="#C7D8F6", sw=2, rx=18))
    parts.append(multiline_text(right_x + 370, 313, ["Transfer-on", "view"], size=25, fill=BLUE_DARK, weight="700", anchor="middle"))

    parts.append(rect(right_x + 275, 480, 190, 110, "#F7F8FA", stroke="#D5DCE6", sw=2, rx=18))
    parts.append(multiline_text(right_x + 370, 528, ["Target-only", "view"], size=25, fill=TEXT, weight="700", anchor="middle"))

    parts.append(line(right_x + 192, 318, right_x + 275, 318, BLUE_DARK, sw=6, marker_end=True))
    parts.append(line(right_x + 192, 533, right_x + 275, 533, GRAY_DARK, sw=6, marker_end=True))
    parts.append(path(f"M {right_x + 140} 470 C {right_x + 190} 430, {right_x + 230} 390, {right_x + 280} 360", "#9AA5B5", sw=4, marker_end=True))

    parts.append(rect(right_x + 540, 340, 185, 210, "#EAF8F7", stroke="#ABDCD8", sw=2, rx=22))
    parts.append(text(right_x + 632, 395, "ETG gate", size=30, fill="#227A76", weight="700", anchor="middle"))
    parts.append(multiline_text(right_x + 632, 448, ["estimate", "helpful / neutral / harmful"], size=19, fill=GRAY_DARK, anchor="middle"))
    parts.append(multiline_text(right_x + 632, 515, ["use uncertainty as", "reliability control"], size=19, fill=GRAY_DARK, anchor="middle"))

    parts.append(path(f"M {right_x + 466} 320 C {right_x + 500} 338, {right_x + 515} 355, {right_x + 540} 380", BLUE_DARK, sw=5, marker_end=True))
    parts.append(path(f"M {right_x + 466} 534 C {right_x + 500} 520, {right_x + 515} 500, {right_x + 540} 470", GRAY_DARK, sw=5, marker_end=True))

    parts.append(rect(right_x + 800, 270, 175, 115, "#F6FBF9", stroke="#C9E6E2", sw=2, rx=18))
    parts.append(multiline_text(right_x + 885, 318, ["Final", "prediction"], size=27, fill=TEXT, weight="700", anchor="middle"))
    parts.append(line(right_x + 726, 438, right_x + 800, 328, TEAL, sw=7, marker_end=True))

    parts.append(badge(right_x + 720, 425, 104, 48, "Helpful", "#EAF8EE", "#B9DFC3"))
    parts.append(badge(right_x + 836, 425, 104, 48, "Neutral", "#F1F3F6", "#D4D9E0"))
    parts.append(badge(right_x + 952, 425, 104, 48, "Harmful", "#FFF0F0", "#F1C0C0"))

    parts.append(rect(right_x + 590, 602, 395, 132, "white", stroke="#D7DEE8", sw=2, rx=20))
    parts.append(text(right_x + 788, 644, "After gating: smaller transfer-gain gap", size=21, weight="700", anchor="middle"))
    parts.append(text(right_x + 620, 685, "Overlap", size=21, fill=BLUE_DARK, weight="700"))
    parts.append(mini_gain_chart(right_x + 745, 689, 110, [0.56, 0.49, 0.60, 0.44]))
    parts.append(text(right_x + 620, 724, "Non-overlap", size=21, fill=ORANGE_DARK, weight="700"))
    parts.append(mini_gain_chart(right_x + 745, 728, 110, [0.38, 0.28, 0.33, 0.26]))
    parts.append(multiline_text(right_x + 880, 677, ["more reliable", "positive gain"], size=17, fill=GRAY_DARK))
    parts.append(multiline_text(right_x + 880, 716, ["fewer harmful", "cases"], size=17, fill=GRAY_DARK))

    parts.append("</svg>")
    return "".join(parts)


def render_png_from_svg():
    firefox = shutil.which("firefox")
    if not firefox:
        return False
    subprocess.run(
        [
            firefox,
            "--headless",
            "--window-size",
            f"{WIDTH},{HEIGHT}",
            "--screenshot",
            str(PNG_PATH),
            SVG_PATH.as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    SVG_PATH.write_text(svg, encoding="utf-8")
    png_ok = render_png_from_svg()
    print(SVG_PATH)
    if png_ok:
        print(PNG_PATH)


if __name__ == "__main__":
    main()
