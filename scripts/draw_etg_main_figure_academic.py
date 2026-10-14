from pathlib import Path
import html
import re
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "figures"
W, H = 1850, 980


COLORS = {
    "ink": "#1F2937",
    "muted": "#5B677A",
    "line": "#2E3440",
    "light_line": "#94A3B8",
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
    "gray": "#F7F9FC",
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


def rect(x, y, w, h, fill, stroke, lw=2, rx=10, dash=None, extra=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{ry(y, h)}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"{dash_attr} {extra}/>'
    )


def circle(x, y, r, fill, stroke=None, lw=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{lw}"' if stroke else ""
    return f'<circle cx="{x}" cy="{sy(y)}" r="{r}" fill="{fill}"{stroke_attr}/>'


def text(x, y, content, size=16, weight="400", anchor="middle", color=None, italic=False, leading=1.18):
    color = color or COLORS["ink"]
    style = "italic" if italic else "normal"
    lines = str(content).split("\n")
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


def line(x1, y1, x2, y2, color=None, lw=2, dash=None, marker=True):
    color = color or COLORS["line"]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#arrow-{marker_key(color)})"' if marker else ""
    return (
        f'<line x1="{x1}" y1="{sy(y1)}" x2="{x2}" y2="{sy(y2)}" '
        f'stroke="{color}" stroke-width="{lw}" fill="none"{dash_attr}{marker_attr}/>'
    )


def path(d, color=None, lw=2, dash=None, marker=True, fill="none", opacity=None):
    color = color or COLORS["line"]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#arrow-{marker_key(color)})"' if marker else ""
    opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
    return (
        f'<path d="{flip_path_d(d)}" stroke="{color}" stroke-width="{lw}" fill="{fill}"'
        f'{dash_attr}{marker_attr}{opacity_attr}/>'
    )


def marker_key(color):
    return color.replace("#", "").lower()


def user_icon(x, y, color, scale=1.0):
    head_r = 8 * scale
    body_w, body_h = 30 * scale, 17 * scale
    return "\n".join(
        [
            circle(x, y + 12 * scale, head_r, color),
            rect(x - body_w / 2, y - 6 * scale, body_w, body_h, color, color, lw=0, rx=6 * scale),
        ]
    )


def mini_users(x, y, colors, scale=1.0, gap=40):
    return "\n".join(user_icon(x + i * gap * scale, y, c, scale) for i, c in enumerate(colors))


def item_tokens(x, y):
    parts = [
        circle(x, y, 6, "#0F766E"),
        f'<polygon points="{x+28},{sy(y+9)} {x+41},{sy(y+3)} {x+34},{sy(y-9)} {x+22},{sy(y-3)}" fill="#D1B892" stroke="#8B6B4A" stroke-width="1"/>',
        rect(x + 55, y - 8, 16, 16, "#374151", "#374151", lw=0, rx=2),
        rect(x + 82, y - 10, 12, 20, "#FFFFFF", "#6B7280", lw=1.3, rx=2),
        line(x + 88, y + 10, x + 88, y + 20, "#6B7280", lw=1.2, marker=False),
    ]
    return "\n".join(parts)


def tiny_network(x, y, scale=1.0, stroke="#7C9BEA"):
    xs = [x, x + 38 * scale, x + 76 * scale]
    ys = [[y + 12, y + 42, y + 72], [y + 26, y + 56], [y + 12, y + 42, y + 72]]
    parts = []
    for i in range(2):
        for yy in ys[i]:
            for yy2 in ys[i + 1]:
                parts.append(line(xs[i], yy, xs[i + 1], yy2, stroke, lw=1.2, marker=False))
    for xx, layer_y in zip(xs, ys):
        for yy in layer_y:
            parts.append(circle(xx, yy, 7 * scale, "#EAF2FF", stroke, lw=1.2))
    return "\n".join(parts)


def cylinder(x, y, w, h, fill, stroke, title):
    parts = [
        f'<ellipse cx="{x + w / 2}" cy="{sy(y + h - 10)}" rx="{w / 2}" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        rect(x, y + 10, w, h - 20, fill, stroke, lw=1.5, rx=0),
        f'<ellipse cx="{x + w / 2}" cy="{sy(y + 10)}" rx="{w / 2}" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        text(x + w / 2, y + h / 2 + 5, title, size=13, weight="700"),
    ]
    return "\n".join(parts)


def score_bar(x, y, w, h, fill, edge, n=5):
    parts = [rect(x, y, w, h, "#FFFFFF", edge, lw=1.4, rx=0)]
    seg_h = (h - 2) / n
    for i in range(n):
        parts.append(rect(x + 2, y + 2 + i * seg_h, w - 4, seg_h - 2, fill if i < n - 1 else "#FFFFFF", edge, lw=0.8, rx=0))
    return "\n".join(parts)


def vector_triplet(x, y):
    return "\n".join(
        [
            rect(x, y, 58, 20, "#FFFFFF", COLORS["blue_edge"], lw=1.2, rx=4),
            rect(x + 70, y, 58, 20, "#FFFFFF", COLORS["green_edge"], lw=1.2, rx=4),
            rect(x + 140, y, 58, 20, "#FFFFFF", COLORS["orange_edge"], lw=1.2, rx=4),
            rect(x + 4, y + 4, 14, 12, "#DCEBFF", COLORS["blue_edge"], lw=0.8, rx=0),
            rect(x + 22, y + 4, 14, 12, "#DCEBFF", COLORS["blue_edge"], lw=0.8, rx=0),
            rect(x + 40, y + 4, 14, 12, "#DCEBFF", COLORS["blue_edge"], lw=0.8, rx=0),
            rect(x + 74, y + 4, 14, 12, "#E1F6E8", COLORS["green_edge"], lw=0.8, rx=0),
            rect(x + 92, y + 4, 14, 12, "#E1F6E8", COLORS["green_edge"], lw=0.8, rx=0),
            rect(x + 110, y + 4, 14, 12, "#E1F6E8", COLORS["green_edge"], lw=0.8, rx=0),
            rect(x + 144, y + 4, 14, 12, "#FFE1C2", COLORS["orange_edge"], lw=0.8, rx=0),
            rect(x + 162, y + 4, 14, 12, "#FFE1C2", COLORS["orange_edge"], lw=0.8, rx=0),
            rect(x + 180, y + 4, 14, 12, "#FFE1C2", COLORS["orange_edge"], lw=0.8, rx=0),
        ]
    )


def evidence_bars(x, y):
    parts = []
    labels = [("e_helpful", "#9BD08F"), ("e_neutral", "#D1D5DB"), ("e_harmful", "#FF8A8A")]
    for i, (lab, fill) in enumerate(labels):
        yy = y - i * 33
        parts.append(text(x, yy + 10, lab, size=12, anchor="start"))
        for j in range(5):
            parts.append(rect(x + 64 + j * 20, yy, 18, 14, fill if j < 4 else "#FFFFFF", "#64748B", lw=0.9, rx=0))
    return "\n".join(parts)


def posterior_pills(x, y):
    pills = [("p_helpful", "#E7F6DE"), ("p_neutral", "#F1F5F9"), ("p_harmful", "#FFE2E2")]
    parts = []
    for i, (lab, fill) in enumerate(pills):
        yy = y - i * 42
        parts.append(rect(x, yy, 86, 22, fill, "#D1D5DB", lw=0.8, rx=8))
        parts.append(text(x + 43, yy + 15, lab, size=12, weight="700"))
    return "\n".join(parts)


def gauge(x, y, w=120, h=58):
    parts = [
        path(f"M {x} {y} C {x + 24} {y + h}, {x + w - 24} {y + h}, {x + w} {y}", COLORS["line"], lw=2, marker=False),
        line(x + w / 2, y + 6, x + w / 2 + 24, y + 28, "#111827", lw=2, marker=False),
        circle(x + w / 2, y + 6, 5, "#111827"),
        text(x, y - 10, "Low", size=11, anchor="start"),
        text(x + w, y - 10, "High", size=11, anchor="end"),
    ]
    return "\n".join(parts)


def plus_node(x, y):
    return "\n".join(
        [
            circle(x, y, 22, "#FFFFFF", COLORS["line"], lw=1.8),
            line(x - 10, y, x + 10, y, COLORS["line"], lw=1.8, marker=False),
            line(x, y - 10, x, y + 10, COLORS["line"], lw=1.8, marker=False),
        ]
    )


def gate_curve_panel(x, y):
    parts = [
        line(x, y, x, y + 130, COLORS["line"], lw=1.4, marker=False),
        line(x, y, x + 108, y, COLORS["line"], lw=1.4, marker=False),
        line(x, y + 65, x + 108, y + 65, "#9CA3AF", lw=1.0, dash="4 4", marker=False),
        text(x - 8, y + 2, "0", size=11),
        text(x - 8, y + 65, "0.5", size=11),
        text(x - 8, y + 130, "1", size=11),
        text(x - 3, y + 145, "-1", size=11),
        text(x + 108, y + 145, "1", size=11),
    ]
    pts = []
    for i in range(80):
        t = -4 + 8 * i / 79
        g = 1 / (1 + pow(2.718281828, -t))
        px = x + (t + 4) / 8 * 108
        py = y + g * 130
        pts.append((px, py))
    d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
    parts.append(path(d, COLORS["orange_edge"], lw=3, marker=False))
    return "\n".join(parts)


def legend_box(x, y):
    parts = [
        rect(x, y, 165, 72, "#FFFFFF", COLORS["gray_edge"], lw=1.3, rx=8),
        line(x + 18, y + 48, x + 68, y + 48, COLORS["line"], lw=1.6),
        text(x + 82, y + 51, "inference flow", size=12, anchor="start"),
        line(x + 18, y + 24, x + 68, y + 24, COLORS["amber_edge"], lw=1.6, dash="5 5"),
        text(x + 82, y + 27, "training signal", size=12, anchor="start"),
    ]
    return "\n".join(parts)


def build_svg():
    arrow_defs = []
    for c in sorted({COLORS[k] for k in COLORS} | {"#111827"}):
        arrow_defs.append(
            f'<marker id="arrow-{marker_key(c)}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
            f'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 Z" fill="{c}"/></marker>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        "<filter id=\"softShadow\" x=\"-15%\" y=\"-15%\" width=\"130%\" height=\"130%\">",
        "<feDropShadow dx=\"0\" dy=\"2\" stdDeviation=\"2.4\" flood-color=\"#111827\" flood-opacity=\"0.13\"/>",
        "</filter>",
        "\n".join(arrow_defs),
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
    ]

    # Title.
    parts.append(text(W / 2, 948, "ETG: Evidential Transfer Gating", size=30, weight="700"))

    # Regions.
    parts.append(rect(20, 290, 1810, 610, "#FBFCFF", "#BDD0EF", lw=1.5, rx=26, dash="6 6"))
    parts.append(rect(20, 30, 1810, 210, "#FFFCEF", "#E4A11B", lw=1.5, rx=22, dash="6 6"))
    parts.append(text(46, 876, "Inference / Forward Path", size=16, anchor="start", color=COLORS["muted"]))
    parts.append(text(46, 210, "Training-only signals (dashed)", size=16, anchor="start", color=COLORS["muted"]))

    # Left input panel.
    parts.append(rect(40, 655, 175, 130, "#FFFFFF", COLORS["gray_edge"], lw=1.8, rx=16, extra='filter="url(#softShadow)"'))
    parts.append(text(128, 742, "Target user-item", size=17, weight="700"))
    parts.append(text(128, 718, "pair (u_t, i_t)", size=17, weight="700"))
    parts.append(user_icon(72, 685, "#0F766E", scale=1.2))
    parts.append(item_tokens(110, 684))
    parts.append(text(195, 688, "...", size=24, anchor="start"))

    # Backbone.
    parts.append(rect(40, 405, 175, 205, "#EAF2FF", COLORS["gray_edge"], lw=1.8, rx=16, extra='filter="url(#softShadow)"'))
    parts.append(text(128, 565, "CDR Backbone", size=18, weight="700", color="#1F3A8A"))
    parts.append(tiny_network(72, 445, 1.15, stroke="#7394DD"))

    # User flow arrows to backbone and branches.
    parts.append(line(128, 655, 128, 610, COLORS["line"], lw=1.9))
    parts.append(path("M 215 720 C 252 720, 248 720, 280 720", COLORS["line"], lw=1.9))
    parts.append(path("M 215 512 C 250 512, 250 510, 280 510", COLORS["line"], lw=1.9))
    parts.append(path("M 215 450 C 248 450, 247 480, 280 500", COLORS["line"], lw=1.9))

    # Branch panels.
    parts.append(rect(280, 620, 320, 160, "#F7FBFF", COLORS["blue_edge"], lw=1.8, rx=16, extra='filter="url(#softShadow)"'))
    parts.append(text(440, 748, "Transfer-on Branch", size=16.5, weight="700", color="#1E3A8A"))
    parts.append(text(440, 725, "transfer enabled", size=14, color=COLORS["muted"]))
    parts.append(cylinder(300, 646, 70, 74, "#EEF5FF", "#5B84D7", "Source\nDomain"))
    parts.append(rect(388, 652, 120, 62, "#FFFFFF", COLORS["blue_edge"], lw=1.5, rx=10))
    parts.append(text(448, 686, "CDR Model", size=16, weight="700"))
    parts.append(text(448, 664, "transfer-on", size=14))
    parts.append(line(370, 683, 388, 683, COLORS["line"], lw=1.6))
    parts.append(line(508, 683, 536, 683, COLORS["line"], lw=1.6))
    parts.append(score_bar(536, 642, 22, 112, "#A7C4FF", COLORS["line"], n=6))
    parts.append(text(547, 758, "z_tr", size=15))

    parts.append(rect(280, 415, 320, 160, "#F7FFFD", COLORS["teal_edge"], lw=1.8, rx=16, extra='filter="url(#softShadow)"'))
    parts.append(text(440, 543, "Target-only Reference Branch", size=16, weight="700", color="#0F766E"))
    parts.append(text(440, 520, "no transfer", size=14, color=COLORS["muted"]))
    parts.append(cylinder(300, 441, 70, 74, "#EEFFFB", "#1CA8AE", "Target\nDomain"))
    parts.append(rect(388, 447, 120, 62, "#FFFFFF", COLORS["teal_edge"], lw=1.5, rx=10))
    parts.append(text(448, 481, "CDR Model", size=16, weight="700"))
    parts.append(text(448, 459, "target-only", size=14))
    parts.append(line(370, 478, 388, 478, COLORS["line"], lw=1.6))
    parts.append(line(508, 478, 536, 478, COLORS["line"], lw=1.6))
    parts.append(score_bar(536, 437, 22, 112, "#AAEFE6", COLORS["line"], n=6))
    parts.append(text(547, 553, "z_to", size=15))

    # ETG outer module.
    parts.append(rect(648, 342, 840, 535, "#FFFFFF", "#7A879A", lw=2.4, rx=22, extra='filter="url(#softShadow)"'))
    parts.append(text(1068, 845, "ETG Module", size=19, weight="700"))

    # Stage A.
    parts.append(rect(672, 390, 195, 430, "#FAFCFF", COLORS["blue_edge"], lw=1.4, rx=14))
    parts.append(rect(686, 770, 167, 52, "#FFFFFF", COLORS["blue_edge"], lw=1.2, rx=10))
    parts.append(text(770, 805, "A. Reliability Feature", size=14.5, weight="700", color="#1E3A8A"))
    parts.append(text(770, 783, "Construction", size=14.5, weight="700", color="#1E3A8A"))
    parts.append(text(770, 724, "Reliability Features", size=15, weight="700"))
    parts.append(rect(704, 645, 132, 54, "#FFFFFF", COLORS["blue_edge"], lw=1.2, rx=10))
    parts.append(text(770, 679, "z_to", size=16, weight="700"))
    parts.append(text(770, 658, "target-only score", size=13))
    parts.append(rect(704, 548, 132, 54, "#FFFFFF", COLORS["blue_edge"], lw=1.2, rx=10))
    parts.append(text(770, 582, "z_tr", size=16, weight="700"))
    parts.append(text(770, 561, "transfer-on score", size=13))
    parts.append(rect(704, 451, 132, 54, "#FFFFFF", COLORS["lav_edge"], lw=1.2, rx=10))
    parts.append(text(770, 485, "z_tr - z_to", size=16, weight="700"))
    parts.append(text(770, 464, "transfer contrast", size=13))
    parts.append(text(770, 418, "x = [z_to, z_tr, z_tr - z_to]", size=15, weight="700"))

    # Stage B.
    parts.append(rect(890, 390, 205, 430, "#FCFFFD", COLORS["green_edge"], lw=1.4, rx=14))
    parts.append(rect(905, 770, 175, 52, "#FFFFFF", COLORS["green_edge"], lw=1.2, rx=10))
    parts.append(text(992, 805, "B. Evidential Transfer Head", size=13.8, weight="700", color="#14532D"))
    parts.append(text(992, 728, "MLP / Evidential Head", size=15, weight="700"))
    parts.append(tiny_network(942, 642, 1.2, stroke="#6B88C8"))
    parts.append(text(992, 610, "Evidence (Dirichlet)", size=15, weight="700"))
    parts.append(evidence_bars(912, 566))
    parts.append(rect(908, 430, 175, 98, "#FFFFFF", COLORS["green_edge"], lw=1.2, rx=10))
    parts.append(text(996, 500, "Uncertainty", size=15, weight="700"))
    parts.append(text(996, 478, "U = K / S", size=15))
    parts.append(gauge(936, 442, 120, 56))

    # Stage C.
    parts.append(rect(1118, 390, 195, 430, "#FEFBFF", COLORS["lav_edge"], lw=1.4, rx=14))
    parts.append(rect(1132, 770, 167, 52, "#FFFFFF", COLORS["lav_edge"], lw=1.2, rx=10))
    parts.append(text(1216, 796, "C. Signed Reliability", size=14.5, weight="700", color="#4C1D95"))
    parts.append(rect(1142, 682, 150, 48, "#FFFFFF", COLORS["lav_edge"], lw=1.2, rx=10))
    parts.append(text(1217, 712, "R = (p_helpful - p_harmful)(1 - U)", size=14.2, weight="700"))
    parts.append(text(1217, 650, "direction from posterior;", size=14))
    parts.append(text(1217, 630, "confidence from evidence", size=14))
    parts.append(rect(1142, 454, 150, 120, "#FFFFFF", COLORS["lav_edge"], lw=1.2, rx=10))
    parts.append(text(1217, 548, "State Posterior", size=15, weight="700"))
    parts.append(text(1217, 528, "from evidential head", size=13))
    parts.append(posterior_pills(1160, 480))
    parts.append(text(1217, 428, "p_helpful + p_neutral + p_harmful = 1", size=13.5))

    # Stage D.
    parts.append(rect(1332, 390, 145, 430, "#FFF9F3", COLORS["orange_edge"], lw=1.4, rx=14))
    parts.append(rect(1342, 770, 126, 52, "#FFFFFF", COLORS["orange_edge"], lw=1.2, rx=10))
    parts.append(text(1405, 805, "D. Transfer Gate", size=14, weight="700", color="#9A3412"))
    parts.append(text(1405, 724, "lambda = G(R, p_harmful)", size=15.5, weight="700"))
    parts.append(gate_curve_panel(1366, 580))
    parts.append(text(1405, 528, "small lambda:", size=13.5, weight="700", color="#9A3412"))
    parts.append(text(1405, 508, "rely on z_to", size=13.5, color="#9A3412"))
    parts.append(text(1405, 470, "large lambda:", size=13.5, weight="700", color="#9A3412"))
    parts.append(text(1405, 450, "trust z_tr", size=13.5, color="#9A3412"))

    # ETG arrows.
    parts.append(line(558, 683, 672, 683, COLORS["line"], lw=2))
    parts.append(line(558, 478, 672, 478, COLORS["line"], lw=2))
    parts.append(line(867, 605, 890, 605, COLORS["line"], lw=2))
    parts.append(line(1095, 605, 1118, 605, COLORS["line"], lw=2))
    parts.append(line(1313, 605, 1332, 605, COLORS["line"], lw=2))

    # Fusion block to the right.
    parts.append(text(1546, 676, "z_tr", size=15))
    parts.append(text(1546, 542, "z_to", size=15))
    parts.append(line(1546, 660, 1546, 627, COLORS["line"], lw=1.8))
    parts.append(line(1546, 558, 1546, 593, COLORS["line"], lw=1.8))
    parts.append(plus_node(1560, 610))
    parts.append(text(1592, 592, "lambda", size=17, color="#D97706", weight="700"))
    parts.append(line(1477, 605, 1538, 605, COLORS["line"], lw=2))
    parts.append(line(1582, 610, 1612, 610, COLORS["line"], lw=2))
    parts.append(rect(1618, 576, 196, 92, "#FFF7F8", COLORS["rose_edge"], lw=1.6, rx=12, extra='filter="url(#softShadow)"'))
    parts.append(text(1716, 636, "Final Prediction", size=18, weight="700", color="#B91C1C"))
    parts.append(text(1716, 607, "z_final = (1 - lambda) z_to + lambda z_tr", size=13.6, weight="700", color="#B91C1C"))
    parts.append(line(1716, 576, 1716, 538, COLORS["line"], lw=1.7))
    parts.append(rect(1628, 398, 176, 120, "#FFFFFF", COLORS["gray_edge"], lw=1.4, rx=12, extra='filter="url(#softShadow)"'))
    parts.append(text(1716, 492, "Target Ranking", size=17, weight="700"))
    for i, ww in enumerate([74, 60, 46, 32, 18]):
        yy = 430 - i * 16
        parts.append(rect(1668, yy, ww, 12, "#FFD4DB", COLORS["rose_edge"], lw=0.8, rx=0))
    parts.append(text(1648, 447, "Score", size=13))

    # Training-only area.
    parts.append(rect(250, 90, 330, 112, "#FFFFFF", COLORS["amber_edge"], lw=1.4, rx=12, extra='filter="url(#softShadow)"'))
    parts.append(text(415, 172, "Delta Pseudo-labels", size=17, weight="700", color="#9A5B00"))
    parts.append(text(415, 142, "Delta = loss(z_to, y) - loss(z_tr, y)", size=15.5))
    parts.append(text(415, 112, "helpful / neutral / harmful", size=15))
    parts.append(circle(332, 100, 8, "#9BD08F", "#64748B", lw=0.8))
    parts.append(circle(415, 100, 8, "#D1D5DB", "#64748B", lw=0.8))
    parts.append(circle(498, 100, 8, "#FF6B6B", "#64748B", lw=0.8))

    parts.append(rect(650, 106, 240, 88, "#FFFFFF", COLORS["amber_edge"], lw=1.4, rx=12, extra='filter="url(#softShadow)"'))
    parts.append(text(770, 164, "State Estimation Loss", size=16, weight="700", color="#7C2D12"))
    parts.append(text(770, 128, "L_EDL", size=20))

    parts.append(rect(930, 106, 360, 88, "#FFFFFF", COLORS["amber_edge"], lw=1.4, rx=12, extra='filter="url(#softShadow)"'))
    parts.append(text(1110, 164, "TGF-aware Loss", size=16, weight="700", color="#7C2D12"))
    parts.append(text(1110, 132, "transfer gain gap between overlap and non-overlap users", size=13.8))
    parts.append(mini_users(1044, 112, ["#5B8DEF", "#5B8DEF"], scale=0.5, gap=28))
    parts.append(mini_users(1180, 112, ["#F59E0B", "#F59E0B"], scale=0.5, gap=28))
    parts.append(line(1108, 102, 1118, 102, COLORS["line"], lw=1.2, marker=False))
    parts.append(line(1118, 102, 1128, 102, COLORS["line"], lw=1.2, marker=True))
    parts.append(text(1118, 90, "gap", size=11))

    parts.append(rect(1245, 44, 390, 128, "#FFFFFF", COLORS["rose_edge"], lw=1.4, rx=12, extra='filter="url(#softShadow)"'))
    parts.append(text(1440, 140, "Training Objective", size=17, weight="700", color="#1F2937"))
    parts.append(text(1440, 102, "L = L_base + lambda_edl L_EDL +", size=18.5, weight="700"))
    parts.append(text(1440, 78, "lambda_tgf L_TGF", size=18.5, weight="700"))

    # Dashed training arrows.
    parts.append(path("M 440 415 C 440 325, 410 270, 410 202", COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(path("M 548 548 C 655 315, 785 265, 785 202", COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(line(580, 146, 650, 146, COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(path("M 785 106 C 785 278, 900 348, 994 390", COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(line(890, 146, 930, 146, COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(path("M 1716 398 C 1716 258, 1250 230, 1110 194", COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(path("M 1110 106 C 1110 74, 1220 74, 1245 92", COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(path("M 770 106 C 770 74, 1225 74, 1245 92", COLORS["amber_edge"], lw=1.7, dash="6 5"))
    parts.append(path("M 415 90 C 415 58, 1225 58, 1245 92", COLORS["amber_edge"], lw=1.7, dash="6 5"))

    # Legend and note.
    parts.append(rect(40, 68, 180, 62, "#FFFFFF", COLORS["gray_edge"], lw=1.2, rx=10))
    parts.append(text(130, 103, "ETG decides whether transfer", size=12.5, color=COLORS["muted"]))
    parts.append(text(130, 83, "should be trusted for each prediction.", size=12.5, color=COLORS["muted"]))
    parts.append(legend_box(1645, 44))

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
    svg_path = OUT_DIR / "ETG_main_figure_academic_v1.svg"
    svg_path.write_text(build_svg(), encoding="utf-8")
    convert_with_libreoffice(svg_path)
    print(svg_path)


if __name__ == "__main__":
    main()
