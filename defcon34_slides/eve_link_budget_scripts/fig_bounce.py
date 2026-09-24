"""
fig_bounce.py -- "the bounce": Venus returns only a small fraction of what hits it.

Per the accepted abstract, Venus returns "about 13%." The link budget brackets this:
    static radar albedo  0.152 = 15.2%  (Goldstein and Carpenter 1963, 13 cm)
    dynamic rho_eff      0.117 = 11.7%  (Magellan GREDR map, near Oct-2026 conjunction)
So the honest statement is "about 13% (roughly 12 to 15 percent)."

Frequency-dependent reflector gain (VERIFIED formula from the notebook), computed at
Dwingeloo's 1299.5 MHz by our validated model:
    venus reflection gain  +164.34 dB   (was 169.31 at 2304 MHz)
    albedo loss            -8.18 dB (0.152)  to  -9.32 dB (0.117)
This is a labelled schematic; the numbers are verified/computed, the geometry is not
to scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from eve_style import apply_style, terminal_frame, COL, SLIDE

apply_style()
fig, ax = plt.subplots(figsize=SLIDE)
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

earth = Circle((10, 30), 5, facecolor=COL["cyan"], edgecolor="white", alpha=0.85)
ax.add_patch(earth)
ax.text(10, 22, "EARTH\nDwingeloo", ha="center", va="top", color=COL["cyan"],
        fontsize=12, fontweight="bold")

venus = Circle((80, 30), 11, facecolor=COL["amber"], edgecolor="white", alpha=0.7)
ax.add_patch(venus)
rng = np.random.default_rng(1995)
for _ in range(220):
    a = rng.uniform(0, 2*np.pi); rr = 11*np.sqrt(rng.uniform(0, 1))
    ax.plot([80+rr*np.cos(a)], [30+rr*np.sin(a)], marker=".", ms=2,
            color=COL["red"], alpha=0.35)
ax.text(80, 16, "VENUS\nrough, spinning sphere\nR = 6051.8 km",
        ha="center", va="top", color=COL["amber"], fontsize=12, fontweight="bold")

ax.add_patch(FancyArrowPatch((16, 31), (69, 31), arrowstyle="-|>",
             mutation_scale=22, color=COL["green"], lw=3))
ax.text(42, 34, "we shout", ha="center", color=COL["green"], fontsize=12, fontweight="bold")
for dy in (-6, -3, 0, 3, 6):
    ax.add_patch(FancyArrowPatch((69, 29+dy*0.3), (16, 29+dy), arrowstyle="-|>",
                 mutation_scale=14, color=COL["red"], lw=1.4, alpha=0.7))
ax.text(42, 21, "about 13% comes back\n(and it comes back SMEARED)",
        ha="center", va="top", color=COL["red"], fontsize=12)

ledger = ("VENUS AS A REFLECTOR (at 1299.5 MHz)\n"
          "  reflector gain   +164.34 dB\n"
          "  returns about 13%  (albedo 0.117 to 0.152)\n"
          "  albedo loss      -9.32 to -8.18 dB")
ax.text(50, 56, ledger, ha="center", va="top", family="DejaVu Sans Mono",
        fontsize=12.5, color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.6", fc=COL["panel"], ec=COL["green2"], lw=1.2))
ax.set_title("THE BOUNCE:  VENUS IS A HUGE, LOUSY, SPINNING MIRROR",
             fontsize=16, fontweight="bold", color=COL["green"], pad=6)
terminal_frame(ax)
fig.tight_layout(); fig.savefig("../figs/fig_bounce.png")
print("saved fig_bounce.png")
