"""Draw the README's main chart: puzzle accuracy against training examples seen.

    python results/plot_curve.py        # writes results/curve_light.svg, curve_dark.svg

Plain SVG, no plotting library: one chart does not justify a dependency, and GitHub
renders SVG in a README. Two files because GitHub picks one per theme through a
<picture> element; the dark one is its own set of colours, not an inverted copy.

Colours are the first three categorical slots of the dataviz reference palette,
which that palette documents as passing its colour-blindness and contrast checks for
all pairs in both modes. Every number below names the file it comes from.
"""
from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (examples seen, % of the 10,000 puzzles solved, source)
CURVE_15M = [
    (1.02e6, 19.9, "17_eval_on_kaggle.json, chess_1M.pt"),
    (6.1e6, 20.5, "17_eval_on_kaggle.json, chess_6.1M.pt"),
    (10.0e6, 20.9, "17_eval_on_kaggle.json, chess_10M.pt"),
    (20.5e6, 21.4, "17_eval_on_kaggle.json, chess_20.5M.pt"),
    (30.7e6, 23.7, "17_eval_on_kaggle.json, chess_30.7M.pt"),
    (50.0e6, 30.3, "17_eval_on_kaggle.json, chess_50M.pt"),
    (98.3e6, 49.1, "17_eval_on_kaggle.json, chess_100M.pt"),
    (225e6, 60.3, "17_eval_on_kaggle.json, chess_225M.pt"),
]
FINE_TUNE = (225e6, 74.6, "17_eval_on_kaggle.json, chess_225M_puzzles.pt")
MODEL_60M = (14.6e6, 21.4, "17_eval_on_kaggle.json, chess_60M_params.pt")
REFERENCES = [
    (88.9, "DeepMind's 9M model, ~1300x the data"),
    (27.4, "ten-line heuristic: take the biggest piece, or mate"),
    (0.6, "random legal move"),
]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926", s3="#199e70"),
}

W, H = 900, 460
LEFT, RIGHT, TOP, BOTTOM = 64, 596, 78, 392          # the legend lives right of the plot
X_MIN, X_MAX = 1e6, 3e8
FONT = "font-family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif'"


def x(examples: float) -> float:
    span = math.log10(X_MAX) - math.log10(X_MIN)
    return LEFT + (math.log10(examples) - math.log10(X_MIN)) / span * (RIGHT - LEFT)


def y(pct: float) -> float:
    return BOTTOM - pct / 100 * (BOTTOM - TOP)


def text(px, py, s, colour, size=12, anchor="start", weight="normal") -> str:
    return (f"<text x='{px:.1f}' y='{py:.1f}' fill='{colour}' font-size='{size}' "
            f"text-anchor='{anchor}' font-weight='{weight}' {FONT}>{s}</text>")


def dot(px, py, colour, surface) -> str:
    # an 8px marker with a 2px ring in the surface colour, so overlapping marks stay apart
    return (f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5' fill='{colour}' "
            f"stroke='{surface}' stroke-width='2'/>")


def draw(theme: str) -> str:
    c = THEMES[theme]
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' "
           f"height='{H}' role='img' aria-labelledby='t d'>",
           "<title id='t'>Puzzles solved against training examples seen</title>",
           "<desc id='d'>"
           + "; ".join(f"{e / 1e6:g}M examples: {p}%" for e, p, _ in CURVE_15M)
           + f"; after a puzzle fine-tune: {FINE_TUNE[1]}%; a 60M-parameter model at "
           f"{MODEL_60M[0] / 1e6:g}M examples: {MODEL_60M[1]}%.</desc>",
           f"<rect width='{W}' height='{H}' fill='{c['surface']}'/>",
           text(LEFT, 30, "Puzzles solved vs training examples seen", c["ink"], 16, weight="600"),
           text(LEFT, 50, "15M-parameter model on a free Kaggle T4, all 10,000 ChessBench "
                "test puzzles", c["ink2"], 12)]

    # grid and axes: recessive
    for pct in range(0, 101, 20):
        out.append(f"<line x1='{LEFT}' x2='{RIGHT}' y1='{y(pct):.1f}' y2='{y(pct):.1f}' "
                   f"stroke='{c['grid'] if pct else c['axis']}' stroke-width='1'/>")
        out.append(text(LEFT - 8, y(pct) + 4, f"{pct}%", c["muted"], 11, "end"))
    for tick, label in [(1e6, "1M"), (3e6, "3M"), (1e7, "10M"), (3e7, "30M"),
                        (1e8, "100M"), (3e8, "300M")]:
        out.append(text(x(tick), BOTTOM + 18, label, c["muted"], 11, "middle"))
    out.append(text((LEFT + RIGHT) / 2, BOTTOM + 40, "training examples seen (log scale)",
                    c["ink2"], 12, "middle"))

    # reference lines, labelled in place
    for pct, label in REFERENCES:
        out.append(f"<line x1='{LEFT}' x2='{RIGHT}' y1='{y(pct):.1f}' y2='{y(pct):.1f}' "
                   f"stroke='{c['muted']}' stroke-width='1' stroke-dasharray='4 4'/>")
        out.append(text(LEFT + 6, y(pct) - 6, f"{label}: {pct}%", c["ink2"], 11))

    # the 15M curve
    pts = " ".join(f"{x(e):.1f},{y(p):.1f}" for e, p, _ in CURVE_15M)
    out.append(f"<polyline points='{pts}' fill='none' stroke='{c['s1']}' stroke-width='2' "
               f"stroke-linejoin='round' stroke-linecap='round'/>")
    for e, p, _ in CURVE_15M:
        out.append(dot(x(e), y(p), c["s1"], c["surface"]))
    for e, p, _ in CURVE_15M[-3:]:                     # label only the points that move
        out.append(text(x(e) - 10, y(p) - 2, f"{p}%", c["ink"], 12, "end", "600"))

    # the fine-tune: same model, one more step, drawn as a jump from its starting point
    fe, fp, _ = FINE_TUNE
    out.append(f"<line x1='{x(fe):.1f}' x2='{x(fe):.1f}' y1='{y(CURVE_15M[-1][1]) - 7:.1f}' "
               f"y2='{y(fp) + 7:.1f}' stroke='{c['s2']}' stroke-width='2' "
               f"stroke-dasharray='3 3'/>")
    out.append(dot(x(fe), y(fp), c["s2"], c["surface"]))
    out.append(text(x(fe) - 10, y(fp) - 2, f"{fp}%", c["ink"], 12, "end", "600"))

    # the 60M-parameter model: more capacity, fewer examples per GPU-hour
    me, mp, _ = MODEL_60M
    out.append(dot(x(me), y(mp), c["s3"], c["surface"]))
    out.append(text(x(me), y(mp) + 22, "60M", c["ink2"], 11, "middle"))

    # legend: identity never rests on colour alone -- each entry is also named
    lx, ly = RIGHT + 24, TOP + 6
    entries = [(c["s1"], "15M model, Stockfish", "annotations only"),
               (c["s2"], "+ 4h fine-tune on", "other Lichess puzzles"),
               (c["s3"], "60M model, 9.5 T4 hours", "(15M model, 10 h: 30.3%)")]
    for i, (colour, line1, line2) in enumerate(entries):
        cy = ly + i * 48
        out.append(dot(lx + 5, cy, colour, c["surface"]))
        out.append(text(lx + 18, cy + 4, line1, c["ink"], 12))
        out.append(text(lx + 18, cy + 20, line2, c["ink2"], 12))

    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for theme in THEMES:
        path = HERE / f"curve_{theme}.svg"
        path.write_text(draw(theme))
        print("wrote", path)
