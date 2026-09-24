"""fig_noise.py -- the quiet we need. Dwingeloo T_sys is MEASURED at 65 K (Telkamp);
Effelsberg is MODELED at ~49 K. The colder Effelsberg receiver is ~1.2 dB of the
+13.27 dB rescue; the 100 m aperture is the other ~12 dB. (Optional/bonus slide.)"""
import numpy as np, matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE
apply_style()
fig, ax = plt.subplots(figsize=SLIDE); ax.grid(True, axis="y", alpha=0.4)
cats=["Dwingeloo 25 m\n(measured)","Effelsberg 100 m\n(modeled)"]
tsys=[65.0,49.0]; x=[0,1]
ax.bar(x, tsys, width=0.5, color=[COL["green"],COL["cyan"]], alpha=0.85, edgecolor="white")
for xi,v in zip(x,tsys):
    ax.text(xi, v+1.2, "T_sys %.0f K"%v, ha="center", color=COL["text"], fontsize=14, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=13); ax.set_ylim(0,80)
ax.set_ylabel("system noise temperature  (K)")
ax.set_title("THE QUIET WE NEED  (1299.5 MHz)", fontsize=17, fontweight="bold", pad=12)
ax.text(0.985,0.95,
        "Dwingeloo 65 K is MEASURED (Telkamp),\n"
        "not modeled -- it anchors the validated\n"
        "link budget. Effelsberg ~49 K modeled.\n"
        "Colder rx = ~1.2 dB of the rescue;\n"
        "the 100 m dish is the other ~12 dB.",
        transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=COL["panel"], ec=COL["green2"], lw=1.2))
terminal_frame(ax, tag="ORI EVE // 65K measured // 49K modeled")
fig.tight_layout(); fig.savefig("../figs/fig_noise.png"); print("saved fig_noise.png")
