"""
eve_style.py -- shared "Hackers terminal" visual style for the EVE DEF CON deck.

ASCII only. No Unicode typographic characters anywhere in this file or its output.

All colors and fonts live here so every figure in the deck looks like it came off
the same green-phosphor CRT. Import COL and apply_style() at the top of each figure.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Hackers / RF Village palette (near-black CRT with neon phosphor) -----------
COL = {
    "bg":      "#0a0e0a",   # near-black terminal background
    "panel":   "#0f1610",   # slightly lifted panel
    "grid":    "#1f3a24",   # dim green grid
    "green":   "#39ff14",   # primary neon green (the phosphor)
    "green2":  "#1fbf0a",   # darker green
    "cyan":    "#00e5ff",   # secondary accent
    "amber":   "#ffb000",   # caution / thresholds
    "red":     "#ff3355",   # loss / not feasible
    "magenta": "#ff5cf4",   # highlight
    "text":    "#c8ffc8",   # pale green text
    "dim":     "#5f8f66",   # dim labels
}


def apply_style():
    """Set global rcParams for the terminal look. Call once per script."""
    mpl.rcParams.update({
        "font.family":        "DejaVu Sans Mono",
        "font.size":          12,
        "figure.facecolor":   COL["bg"],
        "axes.facecolor":     COL["panel"],
        "axes.edgecolor":     COL["green2"],
        "axes.labelcolor":    COL["text"],
        "axes.titlecolor":    COL["green"],
        "axes.linewidth":     1.2,
        "xtick.color":        COL["dim"],
        "ytick.color":        COL["dim"],
        "text.color":         COL["text"],
        "grid.color":         COL["grid"],
        "grid.linewidth":     0.8,
        "figure.dpi":         160,
        "savefig.dpi":        160,
        "savefig.facecolor":  COL["bg"],
        "legend.facecolor":   COL["panel"],
        "legend.edgecolor":   COL["green2"],
        "legend.labelcolor":  COL["text"],
    })


def terminal_frame(ax, tag=None):
    """Dim the top/right spines and optionally stamp a small corner tag."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=4)
    if tag:
        ax.text(0.995, 1.02, tag, transform=ax.transAxes, ha="right", va="bottom",
                color=COL["dim"], fontsize=9)


# Standard 16:9 figure size for slides (inches). 12.8 x 7.2 in at 160 dpi -> 2048x1152 px.
SLIDE = (12.8, 7.2)
