#!/usr/bin/env python3
"""Generate the computer-architecture guide's SVG figures, tuned to the htmler blue theme.

Same house style as the compiler-optimization guide: because the figures are
inlined as static base64 images (no page CSS reaches them), every colour is
chosen to work on BOTH the dark (#0b0d12) and light (#ffffff) themes at once.
A mid-slate around luminance ~0.2 gives roughly 4.3:1 contrast three ways —
white text on the fill, and the same colour as ink on either background.

  * slate blue  #6B7B94  (neutral boxes, connectors, axes, labels)
  * blue        #3E7CC0  (highlighted / "after" boxes)         + dark #2F5F98
  * teal        #1F918C  (positive "result" accent)
  * amber       #D9922B  (warning / spill; dark text on fill)
  * red         #D65A5F  (problem callouts)
  * muted       #9AA0B4  (captions)
  * white       #FFFFFF  (text inside dark fills)
  * 1.5pt wide rules, Aptos / system sans font stack

Run:  python3 scripts/gen_figures.py
Output: figures/*.svg  (referenced from the chapter markdown at the repo root)
"""
import base64
import io
import math
import os

# ── House-style constants (htmler blue theme, dual light/dark legible) ───────
GREY = "#6B7B94"
GREY_D = "#55637A"
BLUE = "#3E7CC0"
BLUE_D = "#2F5F98"
TEAL = "#1F918C"
AMBER = "#D9922B"
RED = "#D65A5F"
WHITE = "#FFFFFF"
LIGHT = "#9AA0B4"
INK_DARK = "#1F2433"  # text on light (amber) fills
# Hand-drawn Excalidraw look: "Virgil" is embedded per-figure as a subsetted
# woff2 data URI (external font URLs are blocked for base64-inlined <img>).
FONT = ("'Virgil','Segoe Print','Bradley Hand','Comic Sans MS',"
        "'Segoe UI',system-ui,-apple-system,sans-serif")
MONO = ("'Virgil','SFMono-Regular',ui-monospace,'JetBrains Mono',Consolas,"
        "monospace")
RULE = 1.5  # pt wide rules

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Virgil.woff2")

# Populated by esc() as figures are built; used to subset the embedded font.
USED_CHARS = set()
# The <style> block (with the base64 @font-face) injected into every SVG.
FONT_STYLE = ""


# ── Primitive builders ──────────────────────────────────────────────────────
def esc(s):
    USED_CHARS.update(str(s))
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def defs():
    """Arrowhead markers in each ink colour."""
    marks = []
    for name, col in (("g", GREY), ("p", BLUE), ("t", TEAL),
                      ("r", RED), ("a", AMBER), ("l", LIGHT)):
        marks.append(
            f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="4.5" markerHeight="4.5" '
            f'orient="auto-start-reverse">'
            f'<path d="M0 1L9 5L0 9z" fill="{col}"/></marker>')
    return "<defs>" + "".join(marks) + "</defs>"


def rrect(x, y, w, h, fill, rx=9, stroke=None, sw=RULE, dash=None, opacity=None):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
         f'fill="{fill}"')
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        s += f' stroke-dasharray="{dash}"'
    if opacity is not None:
        s += f' opacity="{opacity}"'
    return s + "/>"


def tspan_lines(x, cy, lines, fill, size, weight, lh, mono=False):
    """Vertically centred multiline <text>."""
    fam = MONO if mono else FONT
    n = len(lines)
    y0 = cy - (n - 1) * lh / 2.0
    out = [f'<text x="{x}" y="{y0}" fill="{fill}" font-family="{fam}" '
           f'font-size="{size}" font-weight="{weight}" text-anchor="middle" '
           f'dominant-baseline="central">']
    for i, ln in enumerate(lines):
        dy = 0 if i == 0 else lh
        out.append(f'<tspan x="{x}" dy="{dy}">{esc(ln)}</tspan>')
    out.append("</text>")
    return "".join(out)


def box(x, y, w, h, lines, fill=GREY, tcol=WHITE, size=13, weight=600,
        rx=9, lh=16, stroke=None, sw=RULE, dash=None, mono=False):
    if isinstance(lines, str):
        lines = lines.split("\n")
    r = rrect(x, y, w, h, fill, rx=rx, stroke=stroke, sw=sw, dash=dash)
    t = tspan_lines(x + w / 2.0, y + h / 2.0, lines, tcol, size, weight, lh, mono)
    return r + t


def obox(x, y, w, h, lines, stroke=GREY, tcol=GREY, size=13, weight=600,
         rx=9, lh=16, sw=RULE, dash=None, fill="none", mono=False):
    """Outlined box (transparent fill) with coloured text."""
    r = rrect(x, y, w, h, fill, rx=rx, stroke=stroke, sw=sw, dash=dash)
    t = tspan_lines(x + w / 2.0, y + h / 2.0, lines if isinstance(lines, list)
                    else [lines], tcol, size, weight, lh, mono)
    return r + t


def text(x, y, s, fill=GREY, size=13, weight=600, anchor="middle",
         italic=False, mono=False):
    fam = MONO if mono else FONT
    st = ""  # italics disabled: the hand-drawn font is hard to read slanted
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-family="{fam}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"'
            f'{st} dominant-baseline="central">{esc(s)}</text>')


def line(x1, y1, x2, y2, col=GREY, sw=RULE, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
            f'stroke-width="{sw}"{d}/>')


def _mk(col):
    return {GREY: "g", BLUE: "p", TEAL: "t", RED: "r", AMBER: "a",
            LIGHT: "l"}.get(col, "g")


def arrow(x1, y1, x2, y2, col=GREY, sw=RULE, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
            f'stroke-width="{sw}" marker-end="url(#ah-{_mk(col)})"{d}/>')


def path(d, col=GREY, sw=RULE, dash=None, arrow_end=False, fill="none"):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ah-{_mk(col)})"' if arrow_end else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}"'
            f'{dd}{m}/>')


def circle(cx, cy, r, fill, stroke=None, sw=RULE):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{st}/>'


def cylinder(x, y, w, h, fill=GREY, tcol=WHITE, lines=None, size=12,
             stroke=None, sw=RULE):
    """Database / memory cylinder."""
    ry = min(h * 0.16, 14)
    st = (f' stroke="{stroke}" stroke-width="{sw}"') if stroke else ""
    body = (f'<path d="M{x} {y+ry} A{w/2} {ry} 0 0 0 {x+w} {y+ry} '
            f'L{x+w} {y+h-ry} A{w/2} {ry} 0 0 1 {x} {y+h-ry} Z" '
            f'fill="{fill}"{st}/>')
    top = (f'<ellipse cx="{x+w/2}" cy="{y+ry}" rx="{w/2}" ry="{ry}" '
           f'fill="{fill}"{st}/>')
    lip = (f'<path d="M{x} {y+ry} A{w/2} {ry} 0 0 0 {x+w} {y+ry}" '
           f'fill="none" stroke="{WHITE}" stroke-width="1" opacity="0.35"/>')
    t = ""
    if lines:
        t = tspan_lines(x + w / 2.0, y + h / 2.0 + ry / 2, lines, tcol, size,
                        600, 15)
    return body + top + lip + t


def svg(w, h, body, title=""):
    t = f"<title>{esc(title)}</title>" if title else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{FONT}">{t}{FONT_STYLE}'
            f'{defs()}{body}</svg>\n')


def write(rel_path, content):
    full = os.path.join(REPO_ROOT, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    print("wrote", rel_path, f"({len(content)} bytes)")


def core_grid(x, y, cols, rows, cell=13, gap=4, col=BLUE):
    """A small grid of tiny squares (visual shorthand for 'many cores')."""
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(rrect(x + c * (cell + gap), y + r * (cell + gap),
                             cell, cell, col, rx=2))
    return "".join(out)


# ══ FIGURES ══

# ── 01 basics ───────────────────────────────────────────────────────────────
def fig_von_neumann():
    W, H = 760, 380
    b = [text(W / 2, 26, "Von Neumann architecture: one memory, one bus", GREY,
              16, 700)]
    b.append(box(60, 56, 640, 48, ["System Bus", "(Address \u00b7 Data \u00b7 "
                 "Control)"], GREY_D, size=12, lh=17, rx=9))
    units = [(80, "CPU", BLUE, 170), (300, "Memory", TEAL, 120),
             (450, "Input", GREY, 120), (600, "Output", GREY, 120)]
    for x, name, col, w in units:
        b.append(arrow(x + w / 2, 104, x + w / 2, 150, GREY))
        if name == "CPU":
            b.append(box(x, 150, w, 150, "", col, rx=12))
            b.append(text(x + w / 2, 168, "CPU", WHITE, 13, 700))
            b.append(box(x + 20, 186, w - 40, 44, "ALU", BLUE_D, size=12, rx=8))
            b.append(box(x + 20, 240, w - 40, 44, "Control Unit", BLUE_D,
                         size=11, rx=8))
        else:
            b.append(box(x, 150, w, 90, name, col, size=12, rx=10))
    b.append(text(W / 2, 336, "single memory for code + data \u2192 the "
                  "CPU\u2013memory bandwidth \"Von Neumann bottleneck\"", LIGHT,
                  11, 500, italic=True))
    write("figures/von-neumann.svg", svg(W, H, "".join(b), "Von Neumann"))


def fig_harvard():
    W, H = 680, 360
    b = [text(W / 2, 26, "Harvard architecture: separate code & data memories",
              GREY, 15, 700)]
    b.append(box(90, 60, 200, 70, ["Instruction", "Memory"], BLUE, size=12,
                 lh=17, rx=10))
    b.append(box(390, 60, 200, 70, ["Data", "Memory"], TEAL, size=12, lh=17,
                 rx=10))
    b.append(box(270, 240, 160, 84, ["CPU", "ALU + CU"], GREY, size=12, lh=18,
                 rx=12))
    b.append(arrow(190, 130, 320, 240, GREY))
    b.append(text(190, 190, "Instruction Bus", LIGHT, 10, 600))
    b.append(arrow(490, 130, 380, 240, GREY))
    b.append(text(500, 190, "Data Bus", LIGHT, 10, 600))
    b.append(text(W / 2, H - 18, "parallel fetch of code and data \u2014 no Von "
                  "Neumann bottleneck (DSPs, embedded)", LIGHT, 11, 500,
                  italic=True))
    write("figures/harvard.svg", svg(W, H, "".join(b), "Harvard"))


def fig_ieee754():
    W, H = 840, 300
    b = [text(W / 2, 26, "IEEE 754 single precision (32-bit)", GREY, 16, 700)]
    x0 = 70
    fields = [("S", 40, BLUE, "sign", "1 bit"),
              ("Exponent", 200, TEAL, "biased +127", "8 bits"),
              ("Mantissa (fraction)", 480, GREY, "implicit leading 1",
               "23 bits")]
    bits = ["0", "10000010", "10010100000000000000000"]
    x = x0
    for i, (name, w, col, sub, width) in enumerate(fields):
        b.append(box(x, 110, w, 52, name, col, size=12, rx=8))
        b.append(text(x + w / 2, 92, bits[i], GREY, 10, 700, mono=True))
        b.append(text(x + w / 2, 176, width, LIGHT, 10, 700))
        b.append(text(x + w / 2, 192, sub, LIGHT, 9, 500))
        x += w + 6
    b.append(text(W / 2, 236, "example  12.625 = 1.100101 \u00d7 2\u00b3  \u2192"
                  "  0 \u00b7 10000010 \u00b7 10010100\u2026", GREY, 12, 600,
                  mono=True))
    b.append(text(W / 2, 268, "exp all-1s + mantissa 0 = \u00b1\u221e \u00b7 "
                  "all-1s + non-0 = NaN \u00b7 exp 0 = zero/denormal", LIGHT,
                  11, 500, italic=True))
    write("figures/ieee754.svg", svg(W, H, "".join(b), "IEEE 754"))


def fig_logic_gates():
    W, H = 840, 360
    b = [text(W / 2, 26, "Logic gates (IEC symbols)", GREY, 16, 700)]

    def gate(x, y, lab, inv, expr, two=True):
        gw, gh = 60, 56
        b.append(rrect(x, y, gw, gh, "none", rx=6, stroke=GREY, sw=1.6))
        b.append(text(x + gw / 2, y + gh / 2, lab, GREY, 15, 700))
        if two:
            b.append(line(x - 30, y + 16, x, y + 16, GREY))
            b.append(line(x - 30, y + 40, x, y + 40, GREY))
            b.append(text(x - 38, y + 16, "A", GREY, 11, 600, anchor="end"))
            b.append(text(x - 38, y + 40, "B", GREY, 11, 600, anchor="end"))
        else:
            b.append(line(x - 30, y + 28, x, y + 28, GREY))
            b.append(text(x - 38, y + 28, "A", GREY, 11, 600, anchor="end"))
        ox = x + gw
        if inv:
            b.append(circle(ox + 6, y + 28, 5, WHITE, stroke=GREY, sw=1.4))
            b.append(line(ox + 11, y + 28, ox + 34, y + 28, GREY))
        else:
            b.append(line(ox, y + 28, ox + 34, y + 28, GREY))
        b.append(text(ox + 40, y + 28, expr, GREY, 12, 700, anchor="start"))
    gates = [(120, 70, "&", False, "A\u00b7B", True),
             (400, 70, "\u22651", False, "A+B", True),
             (680, 70, "1", True, "\u0100", False),
             (120, 220, "&", True, "(A\u00b7B)'", True),
             (400, 220, "\u22651", True, "(A+B)'", True),
             (680, 220, "=1", False, "A\u2295B", True)]
    names = ["AND", "OR", "NOT", "NAND", "NOR", "XOR"]
    for (x, y, lab, inv, expr, two), nm in zip(gates, names):
        b.append(text(x + 30, y - 14, nm, LIGHT, 11, 700))
        gate(x, y, lab, inv, expr, two)
    write("figures/logic-gates.svg", svg(W, H, "".join(b), "Logic gates"))


# ── 02 CPU ──────────────────────────────────────────────────────────────────
def fig_cpu_overview():
    W, H = 620, 420
    b = [text(W / 2, 26, "Inside the CPU", GREY, 16, 700)]
    b.append(obox(50, 50, 520, 300, "", GREY, GREY, rx=14))
    b.append(text(310, 70, "CPU", GREY, 12, 700))
    b.append(box(90, 92, 170, 76, ["Control Unit", "(CU)"], BLUE, size=12,
                 lh=17, rx=10))
    b.append(box(350, 92, 180, 76, ["Registers", "(PC, IR, \u2026)"], GREY_D,
                 size=12, lh=17, rx=10))
    b.append(box(90, 226, 170, 76, ["ALU", "arithmetic + logic"], BLUE_D,
                 size=12, lh=17, rx=10))
    b.append(box(350, 226, 180, 76, ["Cache", "(L1 / L2)"], TEAL, size=12,
                 lh=17, rx=10))
    b.append(arrow(350, 130, 260, 130, GREY))
    b.append(arrow(175, 168, 175, 226, GREY))
    b.append(arrow(350, 264, 260, 264, GREY))
    b.append(arrow(310, 350, 310, 386, GREY))
    b.append(text(310, 402, "System Bus  (Data \u00b7 Address \u00b7 Control)",
                  LIGHT, 11, 600))
    write("figures/cpu-overview.svg", svg(W, H, "".join(b), "CPU overview"))


def fig_instruction_cycle():
    W, H = 780, 360
    b = [text(W / 2, 26, "The instruction cycle", GREY, 16, 700)]
    phases = [(90, "FETCH", ["MAR \u2190 PC", "MDR \u2190 Mem[MAR]",
               "IR \u2190 MDR", "PC += size"]),
              (320, "DECODE", ["decode opcode", "identify operands",
               "read operands", "gen control"]),
              (550, "EXECUTE", ["ALU operation", "store result",
               "update flags", "handle IRQ"])]
    for x, name, steps in phases:
        b.append(box(x, 64, 150, 46, name, BLUE, size=13, rx=9))
        b.append(obox(x, 128, 150, 118, "", GREY, GREY, rx=9))
        for i, s in enumerate(steps):
            b.append(text(x + 12, 150 + i * 24, s, GREY, 10, 500,
                          anchor="start", mono=True))
    b.append(arrow(240, 87, 320, 87, GREY, 2))
    b.append(arrow(470, 87, 550, 87, GREY, 2))
    b.append(path("M620 110 C640 300 200 300 165 112", GREY, RULE,
                  arrow_end=True))
    b.append(text(W / 2, 300, "repeat for the next instruction", LIGHT, 11, 500,
                  italic=True))
    write("figures/instruction-cycle.svg",
          svg(W, H, "".join(b), "Instruction cycle"))


def fig_branch_predictor():
    W, H = 620, 440
    b = [text(W / 2, 26, "2-bit saturating counter (branch predictor)", GREY,
              15, 700)]
    states = [("Strongly Taken", TEAL, "predict TAKEN"),
              ("Weakly Taken", TEAL, "predict TAKEN"),
              ("Weakly Not Taken", GREY, "predict NOT taken"),
              ("Strongly Not Taken", GREY, "predict NOT taken")]
    x, w = 200, 220
    ys = []
    for i, (name, col, pred) in enumerate(states):
        y = 60 + i * 92
        ys.append(y)
        b.append(box(x, y, w, 52, name, col, size=12, rx=10))
        b.append(text(x + w + 16, y + 26, pred, LIGHT, 10, 600, anchor="start"))
    for i in range(3):
        b.append(arrow(x + 60, ys[i] + 52, x + 60, ys[i + 1], RED))
        b.append(text(x + 30, ys[i] + 72, "\u00ac T", RED, 10, 700))
        b.append(arrow(x + w - 60, ys[i + 1], x + w - 60, ys[i] + 52, TEAL))
        b.append(text(x + w - 28, ys[i] + 72, "T", TEAL, 10, 700))
    b.append(path(f"M{x + w} {ys[0] + 14} C{x + w + 40} {ys[0]} {x + w + 40} "
                  f"{ys[0] + 40} {x + w} {ys[0] + 40}", TEAL, RULE,
                  arrow_end=True))
    b.append(text(x + w + 30, ys[0] + 54, "T", TEAL, 10, 700))
    b.append(path(f"M{x} {ys[3] + 38} C{x - 40} {ys[3] + 52} {x - 40} "
                  f"{ys[3] + 12} {x} {ys[3] + 12}", RED, RULE, arrow_end=True))
    b.append(text(x - 30, ys[3] + 4, "\u00ac T", RED, 10, 700))
    b.append(text(W / 2, H - 14, "T = branch taken \u00b7 \u00acT = not taken "
                  "\u00b7 one miss won't flip the prediction", LIGHT, 11, 500,
                  italic=True))
    write("figures/branch-predictor.svg",
          svg(W, H, "".join(b), "Branch predictor"))


def fig_datapath():
    W, H = 880, 340
    b = [text(W / 2, 26, "Single-cycle datapath", GREY, 16, 700)]
    b.append(box(50, 150, 60, 60, "PC", GREY, size=12, rx=9))
    b.append(box(140, 140, 120, 84, ["Instruction", "Memory"], BLUE, size=11,
                 lh=15, rx=10))
    b.append(box(300, 130, 130, 104, ["Register", "File"], GREY_D, size=12,
                 lh=16, rx=10))
    # ALU trapezoid
    ax = 490
    b.append(path(f"M{ax} 130 L{ax + 90} 150 L{ax + 90} 200 L{ax} 220 "
                  f"L{ax} 185 L{ax + 15} 175 L{ax} 165 Z", BLUE_D, RULE,
                  fill=BLUE_D))
    b.append(text(ax + 52, 176, "ALU", WHITE, 12, 700))
    b.append(box(640, 140, 120, 84, ["Data", "Memory"], TEAL, size=11, lh=15,
                 rx=10))
    b.append(box(800, 150, 60, 60, "WB", GREY, size=11, rx=9))
    b.append(arrow(110, 180, 140, 180, GREY))
    b.append(arrow(260, 180, 300, 180, GREY))
    b.append(arrow(430, 175, 490, 175, GREY))
    b.append(arrow(580, 175, 640, 175, GREY))
    b.append(arrow(760, 180, 800, 180, GREY))
    b.append(path("M830 210 C830 290 365 300 365 236", GREY, RULE, dash="5 5",
                  arrow_end=True))
    b.append(text(W / 2, 312, "fetch \u2192 decode/read \u2192 execute \u2192 "
                  "memory \u2192 write-back, all in one clock", LIGHT, 11, 500,
                  italic=True))
    write("figures/datapath.svg", svg(W, H, "".join(b), "Datapath"))


def fig_alu():
    W, H = 560, 400
    b = [text(W / 2, 26, "Arithmetic Logic Unit (ALU)", GREY, 16, 700)]
    cx = 280
    b.append(path(f"M{cx - 120} 96 L{cx - 15} 96 L{cx} 118 L{cx + 15} 96 "
                  f"L{cx + 120} 96 L{cx + 70} 276 L{cx - 70} 276 Z", BLUE,
                  RULE, fill=BLUE))
    b.append(text(cx, 190, "ALU", WHITE, 18, 800))
    b.append(arrow(cx - 70, 66, cx - 70, 96, GREY))
    b.append(text(cx - 70, 56, "A", GREY, 12, 700))
    b.append(arrow(cx + 70, 66, cx + 70, 96, GREY))
    b.append(text(cx + 70, 56, "B", GREY, 12, 700))
    b.append(arrow(cx - 175, 176, cx - 118, 176, GREY))
    b.append(text(cx - 190, 176, "opcode", GREY, 11, 700, anchor="end"))
    b.append(arrow(cx, 276, cx, 326, GREY, 2))
    b.append(text(cx, 342, "Result", GREY, 12, 700))
    b.append(arrow(cx + 72, 271, cx + 150, 271, AMBER))
    b.append(text(cx + 160, 271, "flags", AMBER, 11, 700, anchor="start"))
    b.append(text(cx + 165, 288, "Z C N V", LIGHT, 9, 600, anchor="start"))
    b.append(text(W / 2, H - 16, "opcode selects the operation; flags report "
                  "zero, carry, negative, overflow", LIGHT, 11, 500,
                  italic=True))
    write("figures/alu.svg", svg(W, H, "".join(b), "ALU"))


# ── 03 memory hierarchy ─────────────────────────────────────────────────────
def fig_memory_hierarchy():
    W, H = 780, 470
    b = [text(W / 2, 26, "The memory hierarchy", GREY, 16, 700)]
    cx = 360
    levels = [("Registers", "< 1 KB \u00b7 0.25 ns", TEAL),
              ("L1 Cache", "64 KB \u00b7 1 ns", BLUE),
              ("L2 Cache", "512 KB \u00b7 3\u201310 ns", BLUE),
              ("L3 Cache", "2\u201364 MB \u00b7 10\u201320 ns", BLUE_D),
              ("Main Memory (RAM)", "4\u2013128 GB \u00b7 50\u2013100 ns", BLUE_D),
              ("SSD / HDD", "TBs \u00b7 \u00b5s\u2013ms", GREY),
              ("Tape / Cloud", "\u221e \u00b7 seconds", GREY_D)]
    y = 62
    for i, (name, spec, col) in enumerate(levels):
        w = 150 + i * 80
        x = cx - w / 2
        b.append(box(x, y, w, 44, name, col, size=12, rx=8))
        b.append(text(cx + w / 2 + 14, y + 22, spec, LIGHT, 10, 600,
                      anchor="start"))
        y += 52
    b.append(arrow(60, 380, 60, 90, TEAL, 2))
    b.append(text(48, 235, "faster", TEAL, 11, 700, anchor="end"))
    b.append(text(48, 251, "smaller", TEAL, 11, 700, anchor="end"))
    b.append(text(48, 267, "costlier", TEAL, 11, 700, anchor="end"))
    b.append(text(W / 2, H - 14, "each level caches the one below it; locality "
                  "keeps hot data near the top", LIGHT, 11, 500, italic=True))
    write("figures/memory-hierarchy.svg",
          svg(W, H, "".join(b), "Memory hierarchy"))


def fig_locality():
    W, H = 780, 300
    b = [text(W / 2, 26, "Two kinds of locality", GREY, 16, 700)]
    b.append(obox(40, 52, 330, 200, "", TEAL, TEAL, rx=12))
    b.append(text(205, 74, "Temporal", TEAL, 13, 700))
    b.append(text(205, 94, "reuse the SAME address soon", LIGHT, 10, 500))
    for i in range(3):
        b.append(box(90 + i * 80, 130, 60, 44, "X", GREY, size=13, rx=8))
        if i < 2:
            b.append(arrow(150 + i * 80, 152, 170 + i * 80, 152, GREY))
    b.append(text(205, 208, "loop counter, accumulator", LIGHT, 10, 500,
                  italic=True))
    b.append(obox(410, 52, 330, 200, "", BLUE, BLUE, rx=12))
    b.append(text(575, 74, "Spatial", BLUE, 13, 700))
    b.append(text(575, 94, "access NEARBY addresses soon", LIGHT, 10, 500))
    for i in range(3):
        b.append(box(460 + i * 80, 130, 60, 44, f"X+{i}", BLUE_D, size=12,
                     rx=8))
    b.append(text(575, 190, "one cache line serves them all", LIGHT, 10, 500,
                  italic=True))
    b.append(text(575, 208, "array walk, struct fields", LIGHT, 10, 500,
                  italic=True))
    write("figures/locality.svg", svg(W, H, "".join(b), "Locality"))


def fig_cache_mapping():
    W, H = 880, 430
    b = [text(W / 2, 26, "Cache placement: direct, set-associative, fully "
              "associative", GREY, 15, 700)]
    panels = [
        (20, "Direct-mapped", BLUE, "one fixed line", [("Tag", 150), ("Index",
         70), ("Off", 40)], 1),
        (310, "2-way Set-assoc.", TEAL, "any line in its set", [("Tag", 130),
         ("Set", 70), ("Off", 40)], 2),
        (600, "Fully associative", GREY_D, "any line at all", [("Tag", 200),
         ("Off", 40)], 3),
    ]
    for x, name, col, sub, addr, mode in panels:
        b.append(obox(x, 52, 260, 300, "", col, col, rx=12))
        b.append(text(x + 130, 74, name, col, 13, 700))
        b.append(box(x + 20, 92, 90, 40, "block B", GREY, size=11, rx=8))
        # cache lines
        lx = x + 150
        lines = [f"line {i}" for i in range(4)]
        for i, ln in enumerate(lines):
            hot = ((mode == 1 and i == 1) or (mode == 2 and i in (1, 2))
                   or mode == 3)
            b.append(box(lx, 92 + i * 44, 90, 34,
                         ln, col if hot else "none",
                         tcol=WHITE if hot else col, size=10, rx=6,
                         stroke=col, sw=1.2))
            if hot:
                b.append(arrow(x + 110, 112, lx, 92 + i * 44 + 17, col, RULE))
        if mode == 2:
            b.append(rrect(lx - 6, 86, 102, 96, "none", rx=8, stroke=col,
                           sw=1.2, dash="4 4"))
            b.append(text(lx + 45, 300, "set 0", LIGHT, 9, 600))
        b.append(text(x + 130, 328, sub, LIGHT, 10, 600, italic=True))
        # address bar
        ax = x + 14
        for lab, aw in addr:
            aw2 = aw * 0.9
            b.append(box(ax, 356, aw2, 26, lab, GREY_D, size=9, rx=5))
            ax += aw2 + 2
    write("figures/cache-mapping.svg", svg(W, H, "".join(b), "Cache mapping"))


def fig_paging():
    W, H = 760, 380
    b = [text(W / 2, 26, "Paging: virtual \u2192 physical address translation",
              GREY, 15, 700)]
    b.append(text(150, 66, "Virtual Address", LIGHT, 11, 700, anchor="start"))
    b.append(box(150, 80, 320, 44, "Page Number (VPN)", BLUE, size=12, rx=8))
    b.append(box(470, 80, 140, 44, "Offset", GREY, size=12, rx=8))
    b.append(box(230, 176, 200, 80, ["Page Table", "VPN \u2192 PFN"], TEAL,
                 size=12, lh=18, rx=10))
    b.append(arrow(310, 124, 330, 176, GREY))
    b.append(text(150, 300, "Physical Address", LIGHT, 11, 700, anchor="start"))
    b.append(box(150, 314, 320, 44, "Frame Number (PFN)", BLUE_D, size=12,
                 rx=8))
    b.append(box(470, 314, 140, 44, "Offset", GREY, size=12, rx=8))
    b.append(arrow(330, 256, 310, 314, GREY))
    b.append(path("M540 124 C560 210 560 240 540 314", GREY, RULE, dash="5 5",
                  arrow_end=True))
    b.append(text(600, 220, "offset", LIGHT, 10, 600, anchor="start"))
    b.append(text(600, 236, "unchanged", LIGHT, 10, 600, anchor="start"))
    b.append(text(W / 2, H - 12, "the page table maps the high bits; the "
                  "low-order offset passes straight through", LIGHT, 11, 500,
                  italic=True))
    write("figures/paging.svg", svg(W, H, "".join(b), "Paging"))


def fig_tlb():
    W, H = 780, 340
    b = [text(W / 2, 26, "The TLB caches recent address translations", GREY,
              15, 700)]
    b.append(box(50, 140, 90, 50, "VA", GREY, size=13, rx=9))
    b.append(box(190, 128, 150, 74, ["TLB", "(translation cache)"], BLUE,
                 size=12, lh=16, rx=10))
    b.append(arrow(140, 165, 190, 165, GREY))
    b.append(box(600, 70, 130, 50, "PA (fast!)", TEAL, size=12, rx=10))
    b.append(path("M340 150 C450 110 500 100 600 95", TEAL, RULE,
                  arrow_end=True))
    b.append(text(460, 118, "hit", TEAL, 11, 700))
    b.append(box(420, 210, 180, 60, ["Page Table walk", "then fill the TLB"],
                 GREY_D, size=11, lh=16, rx=10))
    b.append(path("M300 202 C340 240 380 240 420 240", RED, RULE,
                  arrow_end=True))
    b.append(text(360, 224, "miss", RED, 11, 700))
    b.append(path("M600 235 C660 220 700 190 700 130", GREY, RULE, dash="5 5",
                  arrow_end=True))
    b.append(text(690, 175, "PA", GREY, 11, 700, anchor="end"))
    b.append(text(W / 2, H - 14, "a hit avoids the multi-level page-table walk "
                  "\u2014 translation in ~1 cycle", LIGHT, 11, 500,
                  italic=True))
    write("figures/tlb.svg", svg(W, H, "".join(b), "TLB"))


def fig_write_policies():
    W, H = 780, 320
    b = [text(W / 2, 26, "Write-through vs write-back", GREY, 16, 700)]
    # write-through
    b.append(obox(40, 54, 330, 210, "", BLUE, BLUE, rx=12))
    b.append(text(205, 76, "Write-Through", BLUE, 13, 700))
    b.append(box(70, 100, 90, 44, "CPU", GREY, size=12, rx=9))
    b.append(box(190, 100, 90, 44, "Cache", BLUE, size=12, rx=9))
    b.append(box(190, 190, 90, 44, "Memory", BLUE_D, size=11, rx=9))
    b.append(arrow(160, 122, 190, 122, GREY))
    b.append(arrow(235, 144, 235, 190, BLUE))
    b.append(text(320, 167, "every", LIGHT, 9, 600, anchor="start"))
    b.append(text(320, 181, "write", LIGHT, 9, 600, anchor="start"))
    b.append(text(205, 250, "simple, always consistent, more traffic", LIGHT,
                  9, 500, italic=True))
    # write-back
    b.append(obox(410, 54, 330, 210, "", TEAL, TEAL, rx=12))
    b.append(text(575, 76, "Write-Back", TEAL, 13, 700))
    b.append(box(440, 100, 90, 44, "CPU", GREY, size=12, rx=9))
    b.append(box(560, 100, 90, 44, ["Cache", "dirty bit"], TEAL, size=10,
                 lh=13, rx=9))
    b.append(box(560, 190, 90, 44, "Memory", BLUE_D, size=11, rx=9))
    b.append(arrow(530, 122, 560, 122, GREY))
    b.append(arrow(605, 144, 605, 190, GREY, dash="5 5"))
    b.append(text(690, 160, "only on", LIGHT, 9, 600, anchor="start"))
    b.append(text(690, 174, "eviction", LIGHT, 9, 600, anchor="start"))
    b.append(text(575, 250, "less traffic, needs dirty tracking", LIGHT, 9,
                  500, italic=True))
    write("figures/write-policies.svg",
          svg(W, H, "".join(b), "Write policies"))


# ── 04 instruction sets ─────────────────────────────────────────────────────
def fig_cisc_vs_risc():
    W, H = 700, 470
    b = [text(W / 2, 26, "CISC vs RISC", GREY, 16, 700)]
    rows = [("Instructions", "Many (100s)", "Few"),
            ("Instruction size", "Variable", "Fixed"),
            ("Addressing modes", "Many", "Few"),
            ("Execution time", "Variable", "Uniform"),
            ("Code density", "High", "Lower"),
            ("Pipeline", "Complex", "Simple"),
            ("Registers", "Fewer", "More"),
            ("Memory access", "Any instruction", "Load / Store only"),
            ("Decode", "Complex", "Simple")]
    fx, cx, rx = 40, 260, 470
    fw, cw = 210, 190
    b.append(box(fx, 54, fw, 34, "Feature", GREY_D, size=12, rx=8))
    b.append(box(cx, 54, cw, 34, "CISC  (x86)", BLUE, size=12, rx=8))
    b.append(box(rx, 54, cw, 34, "RISC  (ARM)", TEAL, size=12, rx=8))
    for i, (f, c, r) in enumerate(rows):
        y = 94 + i * 38
        b.append(box(fx, y, fw, 32, f, "none", tcol=GREY, size=11, rx=6,
                     stroke=GREY, sw=1))
        b.append(box(cx, y, cw, 32, c, "none", tcol=BLUE, size=11, rx=6,
                     stroke=BLUE, sw=1))
        b.append(box(rx, y, cw, 32, r, "none", tcol=TEAL, size=11, rx=6,
                     stroke=TEAL, sw=1))
    b.append(text(W / 2, H - 14, "modern reality: x86 decodes to RISC-like "
                  "micro-ops; the line is blurred", LIGHT, 11, 500,
                  italic=True))
    write("figures/cisc-vs-risc.svg", svg(W, H, "".join(b), "CISC vs RISC"))


def fig_addressing_modes():
    W, H = 800, 420
    b = [text(W / 2, 26, "Common addressing modes", GREY, 16, 700)]
    modes = [
        ("Immediate", "operand is IN the instruction", "ADD R1, #5", TEAL),
        ("Register", "operand is in a register", "ADD R1, R2", BLUE),
        ("Direct", "instruction holds the address", "LD R1, (1000)", BLUE),
        ("Register indirect", "register holds the address", "LD R1, (R2)",
         BLUE_D),
        ("Indexed", "address = base + index", "LD R1, (R2+R3)", GREY_D),
        ("PC-relative", "address = PC + offset", "BEQ label", GREY),
    ]
    tw, th = 240, 96
    for i, (name, how, ex, col) in enumerate(modes):
        r, c = divmod(i, 3)
        x = 30 + c * 256
        y = 58 + r * 116
        b.append(obox(x, y, tw, th, "", col, col, rx=12))
        b.append(text(x + 120, y + 22, name, col, 13, 700))
        b.append(text(x + 120, y + 46, how, GREY, 10, 500))
        b.append(box(x + 40, y + 62, tw - 80, 26, ex, col, size=10, rx=6,
                     mono=True))
    b.append(text(W / 2, H - 12, "the mode decides HOW the CPU computes each "
                  "operand's location", LIGHT, 11, 500, italic=True))
    write("figures/addressing-modes.svg",
          svg(W, H, "".join(b), "Addressing modes"))


def fig_instruction_encoding():
    W, H = 820, 340
    b = [text(W / 2, 26, "Fixed-length (RISC) vs variable-length (CISC) "
              "encoding", GREY, 15, 700)]
    # fixed
    b.append(text(210, 62, "Fixed (RISC): every instruction 32 bits", TEAL, 11,
                  700))
    fields = [("op", 60), ("rs", 55), ("rt", 55), ("rd", 55), ("sh", 45),
              ("funct", 70)]
    for row in range(3):
        x = 40
        y = 84 + row * 46
        for lab, w in fields:
            b.append(box(x, y, w, 34, lab if row == 0 else "", TEAL,
                         tcol=WHITE, size=9, rx=4))
            x += w + 2
    b.append(text(210, 232, "same width \u2192 trivial to decode & pipeline",
                  LIGHT, 10, 500, italic=True))
    # variable
    b.append(text(600, 62, "Variable (CISC): 1\u201315 bytes", BLUE, 11, 700))
    varrows = [[("op", 60)],
               [("op", 60), ("modrm", 70), ("disp", 90)],
               [("prefix", 55), ("op", 60), ("modrm", 70), ("imm", 110)]]
    for row, segs in enumerate(varrows):
        x = 460
        y = 84 + row * 46
        for lab, w in segs:
            b.append(box(x, y, w, 34, lab, BLUE, size=9, rx=4))
            x += w + 2
    b.append(text(600, 232, "dense code, but decode is hard (find each "
                  "boundary)", LIGHT, 10, 500, italic=True))
    write("figures/instruction-encoding.svg",
          svg(W, H, "".join(b), "Instruction encoding"))


# ── 05 pipelining ───────────────────────────────────────────────────────────
def _pipe_row(x0, y, start, stages, cw=46, h=30):
    cols = [BLUE, BLUE_D, TEAL, GREY, GREY_D]
    out = []
    for i, st in enumerate(stages):
        x = x0 + (start + i) * cw
        out.append(box(x, y, cw - 3, h, st, cols[i % len(cols)], size=10,
                       rx=5))
    return out


def fig_pipeline_execution():
    W, H = 820, 360
    b = [text(W / 2, 26, "Pipelined execution: one instruction finishes each "
              "cycle", GREY, 15, 700)]
    stages = ["IF", "ID", "EX", "MEM", "WB"]
    x0, cw = 110, 46
    for c in range(9):
        b.append(text(x0 + c * cw + cw / 2 - 1, 56, str(c + 1), LIGHT, 10, 700))
    # steady-state highlight (cycle 5 = index 4)
    b.append(rrect(x0 + 4 * cw - 2, 66, cw, 205, AMBER, rx=6, opacity=0.16))
    b.append(text(x0 + 4 * cw + cw / 2, 286, "all 5 stages busy", AMBER, 9,
                  700))
    for i in range(5):
        y = 72 + i * 40
        b.append(text(x0 - 16, y + 15, f"I{i + 1}", GREY, 11, 700,
                      anchor="end"))
        for s in _pipe_row(x0, y, i, stages, cw):
            b.append(s)
    b.append(text(W / 2, H - 18, "sequential 5\u00d75 = 25 cycles  \u2192  "
                  "pipelined 5+(5\u22121) = 9 cycles  (2.78\u00d7); ideal CPI "
                  "\u2192 1", LIGHT, 11, 500, italic=True))
    write("figures/pipeline-execution.svg",
          svg(W, H, "".join(b), "Pipeline execution"))


def fig_data_hazard():
    W, H = 780, 340
    b = [text(W / 2, 26, "Data hazard solved by forwarding", GREY, 16, 700)]
    stages = ["IF", "ID", "EX", "MEM", "WB"]
    x0, cw = 150, 50
    b.append(text(x0 - 16, 92, "ADD", GREY, 11, 700, anchor="end"))
    for s in _pipe_row(x0, 78, 0, stages, cw):
        b.append(s)
    b.append(text(x0 - 16, 172, "SUB", GREY, 11, 700, anchor="end"))
    for s in _pipe_row(x0, 158, 1, stages, cw):
        b.append(s)
    # forward from ADD EX/MEM (end of cycle3) to SUB EX (cycle4)
    b.append(path(f"M{x0 + 2 * cw + 22} 108 C{x0 + 3 * cw} 135 {x0 + 3 * cw} "
                  f"135 {x0 + 3 * cw + 22} 158", TEAL, 2, arrow_end=True))
    b.append(text(x0 + 3 * cw + 40, 135, "forward EX/MEM \u2192 EX", TEAL, 10,
                  700, anchor="start"))
    b.append(text(W / 2, H - 40, "the result is routed straight to the next "
                  "ALU input \u2014 no stall, CPI stays 1", LIGHT, 11, 500,
                  italic=True))
    b.append(text(W / 2, H - 20, "(a load-use hazard still needs one bubble: "
                  "data isn't ready until MEM)", LIGHT, 10, 500, italic=True))
    write("figures/data-hazard.svg", svg(W, H, "".join(b), "Data hazard"))


def fig_control_hazard():
    W, H = 800, 340
    b = [text(W / 2, 26, "Control hazard: a taken branch flushes the pipeline",
              GREY, 15, 700)]
    stages = ["IF", "ID", "EX", "MEM", "WB"]
    x0, cw = 150, 50
    b.append(text(x0 - 16, 78, "BEQ", GREY, 11, 700, anchor="end"))
    for s in _pipe_row(x0, 64, 0, stages, cw):
        b.append(s)
    # wrong-path fetches (flushed)
    for k, yy in [(1, 108), (2, 138)]:
        b.append(text(x0 - 16, yy + 14, f"I{k + 1}?", RED, 10, 700,
                      anchor="end"))
        for i in range(2):
            x = x0 + (k + i) * cw
            b.append(box(x, yy, cw - 3, 26, "flush", "none", tcol=RED,
                         size=8, rx=5, stroke=RED, sw=1, dash="3 3"))
    b.append(text(x0 + 3.4 * cw, 178, "wrong-path instrs squashed (bubbles)",
                  RED, 10, 600, anchor="start"))
    # correct target
    b.append(text(x0 - 16, 214, "target", TEAL, 10, 700, anchor="end"))
    for s in _pipe_row(x0, 200, 3, stages, cw):
        b.append(s)
    b.append(text(W / 2, H - 16, "branch penalty = cycles lost until the branch "
                  "resolves; prediction hides most of it", LIGHT, 11, 500,
                  italic=True))
    write("figures/control-hazard.svg", svg(W, H, "".join(b), "Control hazard"))


def fig_superscalar():
    W, H = 780, 400
    b = [text(W / 2, 26, "Superscalar: issue several instructions per cycle",
              GREY, 15, 700)]
    b.append(box(240, 54, 300, 40, "Fetch Unit  (4\u20138 instr / cycle)", GREY,
                 size=11, rx=9))
    b.append(arrow(390, 94, 390, 112, GREY))
    b.append(box(240, 112, 300, 40, "Decode / Dispatch  (issue 2\u20134)",
                 BLUE, size=11, rx=9))
    units = [("ALU 0", BLUE_D), ("ALU 1", BLUE_D), ("Load/Store", TEAL),
             ("FP Unit", GREY_D)]
    for i, (name, col) in enumerate(units):
        x = 70 + i * 170
        b.append(arrow(390, 152, x + 70, 190, GREY))
        b.append(box(x, 190, 140, 46, name, col, size=11, rx=9))
        b.append(arrow(x + 70, 236, 390, 280, GREY))
    b.append(box(280, 280, 220, 40, "Write Back", GREY, size=12, rx=9))
    b.append(text(W / 2, H - 40, "multiple fetch/decode/execute units run "
                  "independent instructions together", LIGHT, 11, 500,
                  italic=True))
    b.append(text(W / 2, H - 20, "ideal CPI = 0.5 for dual-issue (IPC = 2); "
                  "4-wide \u2192 up to 4 per cycle", LIGHT, 11, 500,
                  italic=True))
    write("figures/superscalar.svg", svg(W, H, "".join(b), "Superscalar"))


# ── 06 I/O systems ──────────────────────────────────────────────────────────
def fig_dma():
    W, H = 780, 360
    b = [text(W / 2, 26, "Direct Memory Access (DMA)", GREY, 16, 700)]
    b.append(box(90, 62, 130, 56, "CPU", GREY, size=13, rx=10))
    b.append(box(560, 62, 130, 56, "Memory", BLUE_D, size=13, rx=10))
    b.append(rrect(90, 150, 600, 22, GREY, rx=6))
    b.append(text(390, 161, "System Bus", WHITE, 11, 700))
    b.append(arrow(155, 118, 155, 150, GREY))
    b.append(arrow(625, 118, 625, 150, GREY))
    b.append(box(150, 210, 160, 66, ["DMA", "Controller"], TEAL, size=12,
                 lh=16, rx=10))
    b.append(box(470, 210, 160, 66, ["Device Ctrl", "(disk / NIC)"], BLUE,
                 size=11, lh=15, rx=10))
    b.append(arrow(230, 172, 230, 210, GREY))
    b.append(arrow(550, 172, 550, 210, GREY))
    b.append(arrow(310, 243, 470, 243, GREY))
    # direct data path DMA <-> memory (highlighted)
    b.append(path("M230 210 C230 130 625 130 625 118", TEAL, 2.4, dash="6 4",
                  arrow_end=True))
    b.append(text(400, 118, "data moves directly \u2014 CPU not involved", TEAL,
                  10, 700))
    b.append(text(155, 300, "1. CPU programs the DMA controller", LIGHT, 10,
                  500, anchor="start"))
    b.append(text(155, 318, "2. DMA transfers device \u2194 memory on the bus",
                  LIGHT, 10, 500, anchor="start"))
    b.append(text(155, 336, "3. DMA raises an interrupt when done", LIGHT, 10,
                  500, anchor="start"))
    write("figures/dma.svg", svg(W, H, "".join(b), "DMA"))


def fig_io_methods():
    W, H = 800, 320
    b = [text(W / 2, 26, "Three ways to move I/O data", GREY, 16, 700)]
    cards = [
        ("Programmed I/O", BLUE, "CPU busy-waits,\npolling the\nstatus "
         "register", 0.95, "wastes CPU"),
        ("Interrupt-driven", TEAL, "CPU works elsewhere;\ndevice interrupts"
         "\nper byte / block", 0.45, "IRQ overhead"),
        ("DMA", GREY_D, "controller moves\nthe block; CPU only\nsets up + "
         "final IRQ", 0.12, "best for bulk"),
    ]
    tw = 240
    for i, (name, col, desc, load, tag) in enumerate(cards):
        x = 20 + i * 256
        b.append(obox(x, 54, tw, 210, "", col, col, rx=12))
        b.append(text(x + tw / 2, 78, name, col, 13, 700))
        b.append(box(x + 20, 96, tw - 40, 64, desc, "none", tcol=GREY, size=10,
                     lh=15, rx=8, stroke=col, sw=1))
        b.append(text(x + 20, 190, "CPU load", LIGHT, 9, 600, anchor="start"))
        b.append(rrect(x + 20, 198, tw - 40, 18, "#e5e7eb", rx=5))
        b.append(rrect(x + 20, 198, (tw - 40) * load, 18, col, rx=5))
        b.append(text(x + tw / 2, 240, tag, LIGHT, 10, 500, italic=True))
    write("figures/io-methods.svg", svg(W, H, "".join(b), "I/O methods"))


def fig_raid():
    W, H = 820, 380
    b = [text(W / 2, 26, "RAID levels: striping, mirroring, parity", GREY, 15,
              700)]

    def disks(x0, title, col, cols, rows, sub):
        out = [text(x0 + len(cols) * 33, 60, title, col, 12, 700)]
        for ci, cname in enumerate(cols):
            cx = x0 + ci * 66
            out.append(text(cx + 28, 82, cname, LIGHT, 9, 600))
            for ri, val in enumerate(rows):
                cell = val[ci]
                par = cell.endswith("p") or cell.endswith("q")
                out.append(box(cx, 92 + ri * 40, 56, 34, cell,
                               AMBER if par else col, size=10, rx=6))
        out.append(text(x0 + len(cols) * 33, 228, sub, LIGHT, 9, 500,
                        italic=True))
        return out
    b += disks(40, "RAID 0  (stripe)", BLUE,
               ["Disk0", "Disk1"],
               [["A0", "A1"], ["B0", "B1"], ["C0", "C1"]],
               "N\u00d7 speed \u00b7 no redundancy")
    b += disks(300, "RAID 1  (mirror)", TEAL,
               ["Disk0", "Disk1"],
               [["A", "A"], ["B", "B"], ["C", "C"]],
               "survives 1 loss \u00b7 half capacity")
    b += disks(560, "RAID 5  (parity)", GREY_D,
               ["Disk0", "Disk1", "Disk2"],
               [["A0", "A1", "Ap"], ["B0", "Bp", "B1"], ["Cp", "C0", "C1"]],
               "survives 1 loss \u00b7 rotating parity")
    b.append(text(W / 2, H - 14, "parity (amber) = XOR of the data blocks; "
                  "rebuild a lost disk by XOR-ing the rest", LIGHT, 11, 500,
                  italic=True))
    write("figures/raid.svg", svg(W, H, "".join(b), "RAID"))


# ── 07 advanced topics ──────────────────────────────────────────────────────
def fig_flynn():
    W, H = 680, 440
    b = [text(W / 2, 26, "Flynn's taxonomy", GREY, 16, 700)]
    b.append(text(W / 2, 58, "Instruction streams", LIGHT, 11, 700))
    b.append(text(300, 82, "Single", LIGHT, 11, 600))
    b.append(text(500, 82, "Multiple", LIGHT, 11, 600))
    b.append(text(110, 175, "Single", LIGHT, 11, 600))
    b.append(text(110, 320, "Multiple", LIGHT, 11, 600))
    b.append(text(110, 100, "Data", LIGHT, 11, 700))
    b.append(text(110, 116, "streams", LIGHT, 11, 700))
    cells = [(200, 100, "SISD", BLUE, "classic scalar CPU"),
             (400, 100, "MISD", GREY, "rare (fault-tolerant)"),
             (200, 245, "SIMD", TEAL, "GPU, vector, AVX"),
             (400, 245, "MIMD", BLUE_D, "multicore, clusters")]
    for x, y, name, col, ex in cells:
        b.append(box(x, y, 190, 130, "", col, rx=12))
        b.append(text(x + 95, y + 52, name, WHITE, 22, 800))
        b.append(text(x + 95, y + 86, ex, WHITE, 10, 500))
    b.append(text(W / 2, H - 14, "how many instruction and data streams run at "
                  "once", LIGHT, 11, 500, italic=True))
    write("figures/flynn.svg", svg(W, H, "".join(b), "Flynn's taxonomy"))


def fig_amdahl():
    import math
    W, H = 820, 440
    b = [text(W / 2, 26, "Amdahl's law: speedup is capped by the serial part",
              GREY, 15, 700)]
    ox, oy, pw, ph = 90, 360, 560, 280
    b.append(line(ox, oy, ox + pw, oy, GREY_D, 1.5))
    b.append(line(ox, oy, ox, oy - ph, GREY_D, 1.5))
    ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ymax = 20
    for gy in range(0, ymax + 1, 5):
        yy = oy - gy / ymax * ph
        b.append(line(ox, yy, ox + pw, yy, "#e5e7eb", 1))
        b.append(text(ox - 12, yy + 4, str(gy), LIGHT, 9, 600, anchor="end"))
    for i, n in enumerate(ns):
        xx = ox + i / (len(ns) - 1) * pw
        b.append(text(xx, oy + 16, str(n), LIGHT, 9, 600))
    b.append(text(ox + pw / 2, oy + 34, "processors (N)", LIGHT, 10, 700))
    b.append(text(ox - 40, oy - ph - 6, "speedup", LIGHT, 10, 700,
                  anchor="start"))
    curves = [(0.95, TEAL, "95% parallel"), (0.90, BLUE, "90% parallel"),
              (0.50, GREY, "50% parallel")]
    for P, col, lab in curves:
        pts = []
        for i, n in enumerate(ns):
            sp = 1.0 / ((1 - P) + P / n)
            xx = ox + i / (len(ns) - 1) * pw
            yy = oy - min(sp, ymax) / ymax * ph
            pts.append(f"{xx:.1f} {yy:.1f}")
        b.append(path("M" + " L".join(pts), col, 2.4))
        cap = 1.0 / (1 - P)
        yy = oy - min(cap, ymax) / ymax * ph
        b.append(line(ox, yy, ox + pw, yy, col, 1, dash="4 5"))
        b.append(text(ox + pw + 4, yy + 4, lab, col, 9, 700, anchor="start"))
    b.append(text(W / 2, H - 12, "Speedup = 1 / (S + P/N)  \u2014  more cores "
                  "never beat 1/S (e.g. 90% \u2192 max 10\u00d7)", LIGHT, 11,
                  500, italic=True))
    write("figures/amdahl.svg", svg(W, H, "".join(b), "Amdahl's law"))


def fig_mesi():
    W, H = 620, 440
    b = [text(W / 2, 26, "MESI cache-coherence states", GREY, 16, 700)]
    states = [("Invalid", "not valid", GREY, 60),
              ("Exclusive", "clean, only copy", TEAL, 150),
              ("Modified", "dirty, only copy", BLUE_D, 240),
              ("Shared", "clean, others too", BLUE, 330)]
    x, w = 210, 200
    ys = {}
    for name, sub, col, y in states:
        ys[name] = y
        b.append(box(x, y, w, 54, [name, sub], col, size=12, lh=15, rx=10))
    b.append(arrow(x + w / 2, ys["Invalid"] + 54, x + w / 2, ys["Exclusive"],
                   GREY))
    b.append(text(x + w / 2 + 8, ys["Invalid"] + 74, "read miss", LIGHT, 9,
                  600, anchor="start"))
    b.append(arrow(x + w / 2, ys["Exclusive"] + 54, x + w / 2, ys["Modified"],
                   GREY))
    b.append(text(x + w / 2 + 8, ys["Exclusive"] + 74, "CPU write", LIGHT, 9,
                  600, anchor="start"))
    b.append(arrow(x + w / 2, ys["Modified"] + 54, x + w / 2, ys["Shared"],
                   GREY))
    b.append(text(x + w / 2 + 8, ys["Modified"] + 74, "other core reads "
                  "(write back)", LIGHT, 9, 600, anchor="start"))
    b.append(path(f"M{x + w} {ys['Shared'] + 27} C{x + w + 70} "
                  f"{ys['Shared']} {x + w + 70} {ys['Exclusive'] + 27} "
                  f"{x + w} {ys['Exclusive'] + 27}", TEAL, RULE,
                  arrow_end=True))
    b.append(text(x + w + 74, (ys['Shared'] + ys['Exclusive']) / 2 + 27,
                  "re-load", TEAL, 9, 700, anchor="start"))
    b.append(path(f"M{x} {ys['Shared'] + 27} C{x - 80} {ys['Shared']} "
                  f"{x - 80} {ys['Invalid'] + 27} {x} {ys['Invalid'] + 27}",
                  RED, RULE, arrow_end=True))
    b.append(text(x - 84, (ys['Shared'] + ys['Invalid']) / 2 + 27,
                  "other core writes", RED, 9, 700, anchor="end"))
    b.append(text(W / 2, H - 12, "snooping caches watch the bus so every core "
                  "sees one consistent value", LIGHT, 11, 500, italic=True))
    write("figures/mesi.svg", svg(W, H, "".join(b), "MESI"))


def fig_cpu_vs_gpu():
    W, H = 800, 400
    b = [text(W / 2, 26, "CPU vs GPU: latency cores vs throughput cores", GREY,
              15, 700)]
    b.append(obox(40, 56, 340, 280, "", BLUE, BLUE, rx=12))
    b.append(text(210, 80, "CPU \u2014 a few powerful cores", BLUE, 12, 700))
    for i in range(4):
        r, c = divmod(i, 2)
        b.append(box(90 + c * 130, 100 + r * 100, 110, 84, ["Core", "OOO \u00b7 "
                     "big cache"], BLUE, size=11, lh=15, rx=10))
    b.append(text(210, 322, "optimized for single-thread latency", LIGHT, 10,
                  500, italic=True))
    b.append(obox(420, 56, 340, 280, "", TEAL, TEAL, rx=12))
    b.append(text(590, 80, "GPU \u2014 thousands of simple cores", TEAL, 12,
                  700))
    for r in range(6):
        for c in range(12):
            b.append(rrect(445 + c * 26, 100 + r * 26, 20, 20, TEAL, rx=4))
    b.append(text(590, 322, "SIMT, tiny caches \u2014 optimized for "
                  "throughput", LIGHT, 10, 500, italic=True))
    b.append(text(W / 2, H - 12, "SMs run warps of threads in lockstep over "
                  "high-bandwidth VRAM", LIGHT, 11, 500, italic=True))
    write("figures/cpu-vs-gpu.svg", svg(W, H, "".join(b), "CPU vs GPU"))


# ── 08 heterogeneous computing ──────────────────────────────────────────────
def fig_numa():
    W, H = 720, 420
    b = [text(W / 2, 26, "NUMA: local memory is fast, remote memory is slow",
              GREY, 15, 700)]

    def node(x, y, label, cores, mem, col):
        out = [obox(x, y, 300, 130, "", col, col, rx=12),
               text(x + 150, y + 22, label, col, 12, 700)]
        for i in range(4):
            out.append(box(x + 20 + i * 66, y + 34, 56, 34, cores[i], col,
                           size=10, rx=6))
        out.append(box(x + 70, y + 80, 160, 36, mem, GREY_D, size=11, rx=8))
        return out
    b += node(60, 60, "Node 0", ["C0", "C1", "C2", "C3"],
              "Local Mem 0", BLUE)
    b += node(60, 240, "Node 1", ["C4", "C5", "C6", "C7"],
              "Local Mem 1", TEAL)
    b.append(path("M210 176 L210 240", TEAL, 2, arrow_end=True))
    b.append(path("M210 240 L210 176", TEAL, 2, arrow_end=True))
    b.append(text(224, 210, "interconnect (UPI)", LIGHT, 10, 600,
                  anchor="start"))
    b.append(box(430, 120, 240, 40, "local access \u2248 80 ns", BLUE, size=11,
                 rx=9))
    b.append(box(430, 260, 240, 40, "remote access \u2248 140 ns  (1.75\u00d7)",
                 GREY_D, size=11, rx=9))
    b.append(text(W / 2, H - 12, "keep threads near their data; a remote hop "
                  "across the interconnect costs bandwidth + latency", LIGHT,
                  10, 500, italic=True))
    write("figures/numa.svg", svg(W, H, "".join(b), "NUMA"))


def fig_big_little():
    W, H = 720, 400
    b = [text(W / 2, 26, "ARM big.LITTLE heterogeneous cores", GREY, 16, 700)]
    b.append(obox(60, 52, 600, 288, "", GREY, GREY, rx=14))
    b.append(text(360, 74, "SoC", GREY, 12, 700))
    b.append(text(360, 100, "big cores \u2014 out-of-order, 3 GHz, high power",
                  TEAL, 11, 700))
    for i in range(3):
        b.append(box(140 + i * 160, 112, 140, 74, ["Big Core", "OOO \u00b7 "
                     "wide"], TEAL, size=11, lh=15, rx=10))
    b.append(text(360, 214, "LITTLE cores \u2014 in-order, 1.8 GHz, low power",
                  BLUE, 11, 700))
    for i in range(3):
        b.append(box(140 + i * 160, 226, 140, 62, ["LITTLE Core",
                     "in-order"], BLUE, size=11, lh=15, rx=10))
    b.append(box(200, 302, 320, 30, "Shared L3 + cache coherency (CCI)",
                 GREY_D, size=11, rx=8))
    b.append(text(W / 2, H - 12, "the scheduler runs light work on LITTLE "
                  "cores and bursts onto big cores \u2014 saving energy",
                  LIGHT, 11, 500, italic=True))
    write("figures/big-little.svg", svg(W, H, "".join(b), "big.LITTLE"))


# ── 09 performance analysis ─────────────────────────────────────────────────
def fig_cpu_perf_equation():
    W, H = 820, 320
    b = [text(W / 2, 30, "CPU Time = Instructions \u00d7 CPI \u00d7 Clock "
              "Cycle Time", GREY_D, 17, 800)]
    factors = [
        ("Instruction Count", BLUE, "how many instructions",
         "ISA \u00b7 compiler \u00b7 algorithm"),
        ("CPI", TEAL, "cycles per instruction",
         "microarch \u00b7 hazards \u00b7 cache misses"),
        ("Clock Cycle Time", GREY_D, "seconds per cycle",
         "process tech \u00b7 = 1 / frequency"),
    ]
    tw = 240
    for i, (name, col, what, det) in enumerate(factors):
        x = 30 + i * 256
        b.append(box(x, 78, tw, 58, name, col, size=13, rx=10))
        b.append(text(x + tw / 2, 156, what, GREY, 11, 600))
        b.append(text(x + tw / 2, 178, det, LIGHT, 10, 500))
        if i < 2:
            b.append(text(x + tw + 8, 108, "\u00d7", GREY_D, 22, 800))
    b.append(box(210, 216, 400, 56, ["10M \u00d7 2.0 \u00d7 0.333 ns = 6.66 ms"
                 "  \u00b7  \u2192 1500 MIPS"], AMBER, tcol=GREY_D, size=12,
                 rx=10))
    b.append(text(W / 2, H - 12, "lower any factor to go faster \u2014 but they "
                  "trade off (RISC cuts CPI, raises instruction count)", LIGHT,
                  11, 500, italic=True))
    write("figures/cpu-perf-equation.svg",
          svg(W, H, "".join(b), "CPU performance equation"))


# ── appendix ────────────────────────────────────────────────────────────────
def fig_latency_ladder():
    import math
    W, H = 840, 460
    b = [text(W / 2, 26, "Latency numbers every programmer should know", GREY,
              15, 700)]
    rows = [("Register", 0.3, TEAL), ("L1 cache", 1, TEAL),
            ("L2 cache", 3, BLUE), ("L3 cache", 10, BLUE),
            ("Main memory (DRAM)", 100, BLUE_D),
            ("NVMe SSD read", 1e5, GREY),
            ("Network (same DC)", 5e5, GREY),
            ("HDD seek+rotate", 1e7, GREY_D),
            ("Network (coast-to-coast)", 5e7, GREY_D)]
    labels = ["0.3 ns", "1 ns", "3 ns", "10 ns", "100 ns", "100 \u00b5s",
              "500 \u00b5s", "10 ms", "50 ms"]
    x0 = 210
    for i, ((name, ns, col), lab) in enumerate(zip(rows, labels)):
        y = 62 + i * 40
        b.append(text(x0 - 12, y + 18, name, GREY, 11, 600, anchor="end"))
        w = (math.log10(ns) + 1) * 62
        b.append(rrect(x0, y, w, 26, col, rx=6))
        b.append(text(x0 + w + 8, y + 18, lab, LIGHT, 10, 700, anchor="start"))
    b.append(text(W / 2, H - 14, "bar length \u221d log\u2081\u2080(latency): "
                  "each tier is orders of magnitude slower than the last",
                  LIGHT, 11, 500, italic=True))
    write("figures/latency-ladder.svg",
          svg(W, H, "".join(b), "Latency ladder"))


ALL = [
    # 01 basics
    fig_von_neumann, fig_harvard, fig_ieee754, fig_logic_gates,
    # 02 CPU
    fig_cpu_overview, fig_instruction_cycle, fig_branch_predictor,
    fig_datapath, fig_alu,
    # 03 memory hierarchy
    fig_memory_hierarchy, fig_locality, fig_cache_mapping, fig_paging,
    fig_tlb, fig_write_policies,
    # 04 instruction sets
    fig_cisc_vs_risc, fig_addressing_modes, fig_instruction_encoding,
    # 05 pipelining
    fig_pipeline_execution, fig_data_hazard, fig_control_hazard,
    fig_superscalar,
    # 06 I/O systems
    fig_dma, fig_io_methods, fig_raid,
    # 07 advanced topics
    fig_flynn, fig_amdahl, fig_mesi, fig_cpu_vs_gpu,
    # 08 heterogeneous computing
    fig_numa, fig_big_little,
    # 09 performance analysis
    fig_cpu_perf_equation,
    # appendix
    fig_latency_ladder,
]


def build_font_style(chars):
    """Subset Virgil to the glyphs actually used and return an SVG <style>
    block with the font embedded as a woff2 data URI."""
    if not os.path.exists(FONT_PATH):
        print("WARNING: Virgil.woff2 not found; figures will fall back to a "
              "system handwriting font.")
        return ""
    from fontTools.subset import Options, Subsetter
    from fontTools.ttLib import TTFont
    text = "".join(sorted(chars))
    opts = Options()
    opts.flavor = "woff2"
    opts.desubroutinize = True
    opts.notdef_outline = True
    opts.recalc_bounds = True
    font = TTFont(FONT_PATH)
    ss = Subsetter(options=opts)
    ss.populate(text=text)
    ss.subset(font)
    buf = io.BytesIO()
    font.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f"embedded font: {len(chars)} glyphs, "
          f"{len(buf.getvalue())} bytes woff2")
    return ('<style>@font-face{font-family:"Virgil";font-style:normal;'
            'font-weight:400 700;src:url("data:font/woff2;base64,'
            f'{b64}") format("woff2");}}</style>')


if __name__ == "__main__":
    # Pass 1: build every figure once to discover which glyphs are used.
    for fn in ALL:
        fn()
    # Subset + embed the font, then Pass 2: rewrite the figures with it.
    FONT_STYLE = build_font_style(USED_CHARS)
    for fn in ALL:
        fn()
    print(f"\nDone: {len(ALL)} figures generated.")
