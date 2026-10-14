from pathlib import Path
import html
import re
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "figures"
W, H = 1600, 880


COLORS = {
    "ink": "#1F2937",
    "muted": "#5B677A",
    "line": "#4A5568",
    "blue": "#DCEBFF",
    "blue_edge": "#4D7FE8",
    "teal": "#DDF7F5",
    "teal_edge": "#00A0A8",
    "green": "#E1F6E8",
    "green_edge": "#2F9E5C",
    "amber": "#FFF3C6",
    "amber_edge": "#D99A00",
    "orange": "#FFE1C2",
    "orange_edge": "#E47B2B",
    "rose": "#FFE3E6",
    "rose_edge": "#E35A6F",
    "lav": "#EFE7FF",
    "lav_edge": "#7C63E6",
    "gray_edge": "#A6B0BF",
}


def esc(s):
    return html.escape(str(s), quote=True)


def sy(y):
    return H - y


def ry(y, h):
    return H - y - h


def flip_path_d(d):
    tokens = re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?", d)
    out = []
    coord_idx = 0
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z]", tok):
            out.append(tok)
            coord_idx = 0
            continue
        val = float(tok)
        if coord_idx % 2 == 1:
            val = sy(val)
        out.append(f"{val:g}")
        coord_idx += 1
    return " ".join(out)


def attrs(**kw):
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        out.append(f'{k.replace("_", "-")}="{esc(v)}"')
    return " ".join(out)


def rect(x, y, w, h, fill, stroke, lw=2, rx=10, dash=None, extra=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{ry(y, h)}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"{dash_attr} {extra}/>'
    )


def circle(x, y, r, fill, stroke=None, lw=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{lw}"' if stroke else ""
    return f'<circle cx="{x}" cy="{sy(y)}" r="{r}" fill="{fill}"{stroke_attr}/>'


def line(x1, y1, x2, y2, color=None, lw=2, dash=None, marker=True):
    color = color or COLORS["line"]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#arrow-{marker_key(color)})"' if marker else ""
    return (
        f'<line x1="{x1}" y1="{sy(y1)}" x2="{x2}" y2="{sy(y2)}" '
        f'stroke="{color}" stroke-width="{lw}" fill="none"{dash_attr}{marker_attr}/>'
    )


def path(d, color=None, lw=2, dash=None, marker=True):
    color = color or COLORS["line"]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#arrow-{marker_key(color)})"' if marker else ""
    return f'<path d="{flip_path_d(d)}" stroke="{color}" stroke-width="{lw}" fill="none"{dash_attr}{marker_attr}/>'


def marker_key(color):
    return color.replace("#", "").lower()


def text(x, y, content, size=16, weight="400", anchor="middle", color=None, italic=False, leading=1.18):
    color = color or COLORS["ink"]
    lines = str(content).split("\n")
    style = "italic" if italic else "normal"
    parts = [
        f'<text x="{x}" y="{sy(y)}" font-size="{size}" font-family="Times New Roman, Nimbus Roman, serif" '
        f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" fill="{color}">'
    ]
    if len(lines) == 1:
        parts.append(esc(lines[0]))
    else:
        for i, ln in enumerate(lines):
            dy = 0 if i == 0 else size * leading
            parts.append(f'<tspan x="{x}" dy="{dy}">{esc(ln)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def user_icon(x, y, color, scale=1.0):
    head_r = 7.5 * scale
    body_w, body_h = 27 * scale, 16 * scale
    return "\n".join(
        [
            circle(x, y + 13 * scale, head_r, color),
            rect(x - body_w / 2, y - 4 * scale, body_w, body_h, color, color, lw=0, rx=6 * scale),
        ]
    )


def mini_users(x, y, colors, scale=1.0, gap=38):
    return "\n".join(user_icon(x + i * gap * scale, y, c, scale) for i, c in enumerate(colors))


def vector_bar(x, y, w, h, fill, edge, n=4):
    parts = [rect(x, y, w, h, "#FFFFFF", edge, lw=1.4, rx=4)]
    gap = 3
    seg_w = (w - gap * (n + 1)) / n
    for i in range(n):
        parts.append(rect(x + gap + i * (seg_w + gap), y + gap, seg_w, h - 2 * gap, fill, edge, lw=1, rx=0))
    return "\n".join(parts)


def draw_mlp(x, y):
    xs = [x, x + 32, x + 64]
    ys = [[y + 15, y + 42, y + 69], [y + 28, y + 55], [y + 15, y + 42, y + 69]]
    parts = []
    for layer in range(2):
        for yy in ys[layer]:
            for yy2 in ys[layer + 1]:
                parts.append(line(xs[layer], yy, xs[layer + 1], yy2, "#A7B3C2", lw=1, marker=False))
    for layer_x, layer_y in zip(xs, ys):
        for yy in layer_y:
            parts.append(circle(layer_x, yy, 6, "#FFFFFF", COLORS["teal_edge"], lw=1.2))
    return "\n".join(parts)


def gate_curve(x, y, w, h):
    pts = []
    for i in range(80):
        t = -4 + 8 * i / 79
        g = 1 / (1 + pow(2.718281828, -t))
        pts.append((x + (t + 4) / 8 * w, y + g * h))
    d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
    fill_d = f"M {x} {y+h} " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts) + f" L {x+w} {y+h} Z"
    return "\n".join(
        [
            f'<path d="{flip_path_d(fill_d)}" fill="#FFE9D6" opacity="0.55"/>',
            line(x, y + h, x + w, y + h, COLORS["line"], lw=1.4, marker=False),
            line(x, y + h, x, y, COLORS["line"], lw=1.4, marker=False),
            f'<path d="{flip_path_d(d)}" stroke="{COLORS["orange_edge"]}" stroke-width="3" fill="none"/>',
            text(x + w - 6, y + h + 17, "R", size=13),
            text(x - 10, y - 3, "lambda", size=13),
        ]
    )


def ranking_box(x, y, w, h):
    parts = [rect(x, y, w, h, "#FFFFFF", COLORS["rose_edge"], lw=1.4, rx=9)]
    parts.append(text(x + w / 2, y + h - 22, "Target Ranking", size=13, weight="700"))
    widths = [0.74, 0.60, 0.47, 0.33]
    for i, ww in enumerate(widths):
        yy = y + h - 48 - i * 19
        parts.append(text(x + 11, yy + 10, str(i + 1), size=10, color=COLORS["muted"]))
        parts.append(
            rect(x + 22, yy, w * ww, 10, "#FFC9D2", COLORS["rose_edge"], lw=0.8, rx=0)
        )
    return "\n".join(parts)


def build_svg():
    arrow_defs = []
    for c in sorted({COLORS[k] for k in COLORS} | {"#A7B3C2"}):
        key = marker_key(c)
        arrow_defs.append(
            f'<marker id="arrow-{key}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
            f'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 Z" fill="{c}"/></marker>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        "<filter id=\"softShadow\" x=\"-10%\" y=\"-10%\" width=\"120%\" height=\"120%\">",
        "<feDropShadow dx=\"0\" dy=\"2\" stdDeviation=\"2\" flood-color=\"#1F2937\" flood-opacity=\"0.12\"/>",
        "</filter>",
        "\n".join(arrow_defs),
        "</defs>",
        '<rect width="1600" height="880" fill="#FFFFFF"/>',
    ]

    # Background regions.
    parts.append(rect(35, 235, 1530, 600, "#FAFCFF", "#B9C9E3", lw=1.5, rx=24, dash="6 6"))
    parts.append(rect(35, 35, 1530, 175, "#FFFCF0", "#E3B654", lw=1.5, rx=22, dash="6 6"))
    parts.append(text(55, 808, "Inference / Forward Path", size=15, anchor="start", color=COLORS["muted"]))
    parts.append(text(55, 183, "Training-only Signals", size=15, anchor="start", color=COLORS["muted"]))

    # Input domain panel.
    parts.append(rect(65, 345, 175, 400, "#FFFFFF", COLORS["gray_edge"], lw=2, rx=16, extra='filter="url(#softShadow)"'))
    parts.append(rect(85, 705, 135, 32, "#D7E5FA", "#D7E5FA", lw=0, rx=6))
    parts.append(text(152, 726, "Source Domain", size=14, weight="700"))
    parts.append(mini_users(105, 647, ["#9EC3E8", "#9EC3E8", "#9EC3E8"], scale=1.1, gap=36))
    parts.append(line(95, 605, 200, 574, "#92A3B5", lw=1.2, marker=False))
    for px, py in [(95, 592), (128, 582), (162, 572), (195, 562)]:
        parts.append(circle(px, py, 5, "#F0B45B"))
    parts.append(rect(85, 500, 135, 32, "#D7E5FA", "#D7E5FA", lw=0, rx=6))
    parts.append(text(152, 521, "Target Domain", size=14, weight="700"))
    parts.append(text(152, 467, "overlap", size=13, weight="700", color=COLORS["muted"]))
    parts.append(mini_users(105, 432, ["#9EC3E8", "#9EC3E8"], scale=1.05, gap=42))
    parts.append(text(152, 396, "non-overlap", size=13, weight="700", color=COLORS["muted"]))
    parts.append(mini_users(100, 363, ["#A9D382", "#A9D382", "#A9D382"], scale=0.95, gap=37))

    # Backbone.
    parts.append(rect(275, 435, 175, 235, COLORS["blue"], COLORS["blue_edge"], lw=2, rx=18, extra='filter="url(#softShadow)"'))
    parts.append(text(362, 575, "CDR Backbone", size=17, weight="700"))
    parts.append(text(362, 548, "base recommender", size=13, color=COLORS["muted"]))
    for i in range(4):
        parts.append(rect(308 + i * 18, 478, 11, 92, "#BBD4FF", COLORS["blue_edge"], lw=0.8, rx=0))
    for i in range(4):
        parts.append(rect(378 + i * 13, 492, 8, 65, "#A9C3F3", COLORS["blue_edge"], lw=0.8, rx=0))
    parts.append(line(238, 545, 275, 545, COLORS["line"], lw=2))
    parts.append(path("M 238 460 C 255 465, 262 485, 275 492", COLORS["line"], lw=2))

    # ETG envelope.
    parts.append(rect(490, 262, 1035, 545, "#FFFFFF", "#7E8DA3", lw=2.4, rx=20, extra='filter="url(#softShadow)"'))
    parts.append(rect(518, 768, 215, 34, "#EDF4FF", "#EDF4FF", lw=0, rx=7))
    parts.append(text(626, 790, "ETG Module", size=17, weight="700"))

    # Stage A.
    parts.append(rect(518, 428, 225, 325, "#F7FBFF", COLORS["teal_edge"], lw=1.7, rx=14))
    parts.append(rect(542, 716, 175, 34, "#FFFFFF", COLORS["teal_edge"], lw=1.2, rx=7))
    parts.append(text(630, 738, "A. Reliability Feature", size=13, weight="700"))
    parts.append(text(631, 695, "Construction", size=13, weight="700"))
    parts.append(rect(548, 618, 168, 56, COLORS["green"], COLORS["green_edge"], lw=1.5, rx=9))
    parts.append(text(632, 657, "Transfer-on Branch", size=13, weight="700"))
    parts.append(text(632, 632, "z_tr", size=13))
    parts.append(rect(548, 500, 168, 62, "#EDF4FF", COLORS["blue_edge"], lw=1.5, rx=9))
    parts.append(text(632, 548, "Target-only", size=13, weight="700"))
    parts.append(text(632, 524, "Reference Branch  z_to", size=12))
    parts.append(text(631, 489, "x = [z_to, z_tr, z_tr - z_to]", size=13))
    parts.append(vector_bar(548, 448, 58, 22, "#D7E5FA", COLORS["blue_edge"], n=3))
    parts.append(vector_bar(614, 448, 58, 22, "#DFF3E7", COLORS["green_edge"], n=3))
    parts.append(vector_bar(680, 448, 58, 22, "#FFE1C2", COLORS["orange_edge"], n=3))
    parts.append(path("M 450 610 C 480 618, 503 635, 548 646", COLORS["green_edge"], lw=2))
    parts.append(path("M 450 490 C 483 497, 510 520, 548 531", COLORS["blue_edge"], lw=2))

    # Stage B.
    parts.append(rect(772, 428, 230, 325, "#F9FEFC", COLORS["green_edge"], lw=1.7, rx=14))
    parts.append(rect(797, 716, 175, 34, "#FFFFFF", COLORS["green_edge"], lw=1.2, rx=7))
    parts.append(text(885, 738, "B. Evidential Transfer Head", size=11.2, weight="700"))
    parts.append(draw_mlp(803, 605))
    parts.append(text(837, 586, "e = softplus(MLP(x))", size=12.5))
    parts.append(rect(895, 620, 72, 58, "#FFFFFF", COLORS["green_edge"], lw=1.3, rx=7))
    parts.append(text(931, 668, "Evidence", size=12, weight="700"))
    parts.append(text(931, 645, "Dirichlet", size=12, color=COLORS["muted"]))
    parts.append(rect(805, 500, 78, 66, "#FFFFFF", COLORS["green_edge"], lw=1.3, rx=7))
    parts.append(text(844, 552, "State", size=12, weight="700"))
    parts.append(text(844, 528, "Posterior", size=12, weight="700"))
    parts.append(text(912, 570, "p_helpful", size=12, anchor="start"))
    parts.append(text(912, 547, "p_neutral", size=12, anchor="start"))
    parts.append(text(912, 524, "p_harmful", size=12, anchor="start"))
    parts.append(rect(805, 445, 162, 32, COLORS["lav"], COLORS["lav_edge"], lw=1.2, rx=7))
    parts.append(text(886, 467, "Uncertainty  U = K / S", size=12.5, weight="700"))
    parts.append(line(743, 585, 772, 585, COLORS["teal_edge"], lw=2))

    # Stage C.
    parts.append(rect(1028, 428, 192, 325, "#FBFAFF", COLORS["lav_edge"], lw=1.7, rx=14))
    parts.append(rect(1036, 716, 175, 34, "#FFFFFF", COLORS["lav_edge"], lw=1.2, rx=7))
    parts.append(text(1124, 738, "C. Signed Reliability", size=13, weight="700"))
    parts.append(text(1124, 657, "R = (p_helpful - p_harmful)(1 - U)", size=12))
    parts.append(line(1055, 555, 1190, 555, COLORS["line"], lw=1.6, marker=False))
    parts.append(circle(1125, 555, 11, COLORS["lav_edge"], "#FFFFFF", lw=1.2))
    parts.append(text(1058, 585, "harmful", size=12, color=COLORS["muted"]))
    parts.append(text(1184, 585, "helpful", size=12, color=COLORS["muted"]))
    parts.append(text(1125, 640, "state direction", size=12.5, color=COLORS["muted"]))
    parts.append(text(1125, 477, "uncertainty attenuates\nstate direction", size=12, color=COLORS["muted"]))
    parts.append(line(1002, 572, 1028, 572, COLORS["green_edge"], lw=2))

    # Stage D.
    parts.append(rect(1246, 428, 182, 325, "#FFF9F2", COLORS["orange_edge"], lw=1.7, rx=14))
    parts.append(rect(1255, 716, 175, 34, "#FFFFFF", COLORS["orange_edge"], lw=1.2, rx=7))
    parts.append(text(1342, 738, "D. Transfer Gate", size=13, weight="700"))
    parts.append(gate_curve(1278, 552, 102, 82))
    parts.append(text(1337, 510, "lambda = G(R, p_harmful)", size=12.5))
    parts.append(text(1337, 475, "continuous\ntransfer usage", size=12, color=COLORS["muted"]))
    parts.append(line(1220, 572, 1246, 572, COLORS["lav_edge"], lw=2))

    # Prediction fusion.
    parts.append(rect(1445, 502, 96, 178, COLORS["rose"], COLORS["rose_edge"], lw=1.7, rx=13))
    parts.append(text(1493, 655, "Prediction", size=12.5, weight="700"))
    parts.append(text(1493, 631, "Fusion", size=12.5, weight="700"))
    parts.append(text(1493, 588, "z_final", size=14))
    parts.append(text(1493, 553, "(1-lambda) z_to", size=11.5))
    parts.append(text(1493, 530, "+ lambda z_tr", size=11.5))
    parts.append(ranking_box(1445, 332, 96, 126))
    parts.append(line(1428, 572, 1445, 572, COLORS["orange_edge"], lw=2))
    parts.append(line(1493, 502, 1493, 458, COLORS["rose_edge"], lw=1.8))
    parts.append(path("M 695 618 C 875 710, 1260 693, 1445 612", COLORS["green_edge"], lw=1.5, dash="6 5"))
    parts.append(path("M 695 500 C 885 435, 1270 474, 1445 538", COLORS["blue_edge"], lw=1.5, dash="6 5"))

    # Training-only signals.
    parts.append(rect(495, 72, 265, 92, "#FFFFFF", COLORS["amber_edge"], lw=1.5, rx=11))
    parts.append(text(628, 148, "Delta Pseudo-labels", size=13.5, weight="700"))
    parts.append(text(628, 119, "Delta = loss(z_to,y) - loss(z_tr,y)", size=12))
    parts.append(text(628, 94, "helpful / neutral / harmful", size=12, color=COLORS["muted"]))
    parts.append(rect(805, 78, 142, 80, "#FFFFFF", COLORS["amber_edge"], lw=1.5, rx=11))
    parts.append(text(876, 132, "State Estimation Loss", size=11.4, weight="700"))
    parts.append(text(876, 104, "L_EDL", size=14))
    parts.append(rect(1012, 78, 165, 80, "#FFFFFF", COLORS["amber_edge"], lw=1.5, rx=11))
    parts.append(text(1094, 132, "TGF-aware Loss", size=12.5, weight="700"))
    parts.append(text(1094, 104, "transfer gain gap", size=12, color=COLORS["muted"]))
    parts.append(mini_users(1042, 151, ["#9EC3E8", "#9EC3E8"], scale=0.44, gap=32))
    parts.append(mini_users(1106, 151, ["#A9D382", "#A9D382"], scale=0.44, gap=32))
    parts.append(rect(1245, 72, 300, 92, "#FFFFFF", COLORS["rose_edge"], lw=1.5, rx=11))
    parts.append(text(1395, 148, "Training Objective", size=13.5, weight="700"))
    parts.append(text(1395, 119, "L = L_base + lambda_edl L_EDL + lambda_tgf L_TGF", size=12.3))
    parts.append(line(628, 428, 628, 164, COLORS["amber_edge"], lw=1.6, dash="5 5"))
    parts.append(line(760, 118, 805, 118, COLORS["amber_edge"], lw=1.6, dash="5 5"))
    parts.append(line(947, 118, 1012, 118, COLORS["amber_edge"], lw=1.6, dash="5 5"))
    parts.append(line(1177, 118, 1245, 118, COLORS["amber_edge"], lw=1.6, dash="5 5"))
    parts.append(path("M 1492 502 C 1515 375, 1520 245, 1492 164", COLORS["amber_edge"], lw=1.6, dash="5 5"))

    # Small explanatory callout.
    parts.append(rect(75, 68, 272, 58, "#FFFFFF", "#CAD2DD", lw=1.3, rx=8))
    parts.append(text(211, 112, "ETG estimates whether transfer is helpful,", size=12, color=COLORS["muted"]))
    parts.append(text(211, 91, "neutral, or harmful, then gates transfer usage.", size=12, color=COLORS["muted"]))

    parts.append("</svg>")
    return "\n".join(parts)


def convert_with_libreoffice(svg_path):
    if not shutil.which("libreoffice"):
        return []
    created = []
    for target in ("png", "pdf"):
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", target, "--outdir", str(OUT_DIR), str(svg_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        candidate = OUT_DIR / f"{svg_path.stem}.{target}"
        if candidate.exists():
            created.append(candidate)
    return created


def main():
    svg_path = OUT_DIR / "ETG_method_framework_strict_v3.svg"
    svg_path.write_text(build_svg(), encoding="utf-8")
    convert_with_libreoffice(svg_path)
    print(svg_path)


if __name__ == "__main__":
    main()
