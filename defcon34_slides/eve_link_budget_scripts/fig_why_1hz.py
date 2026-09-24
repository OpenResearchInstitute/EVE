"""fig_why_1hz.py -- 1 Hz is the ruler, not the receiver. And it is literally how the
CAMRAS echoes were measured: CNR in a 1 Hz band, motivated by the +/-0.75 Hz Doppler
spread. Anchors (Oct 2026): Dwingeloo alone -1.33 dB-Hz, Dwingeloo+Effelsberg +11.94."""
import numpy as np, matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE
apply_style()
B = np.logspace(-1,5,600)
for cn0,col,name in [(-1.33,COL["green"],"Dwingeloo alone"),(11.94,COL["cyan"],"Dwingeloo + Effelsberg")]:
    plt.plot(B, cn0-10*np.log10(B), color=col, lw=3, label="%s  (C/N0 %+.2f)"%(name,cn0))
    bx=10**(cn0/10.0); plt.scatter([bx],[0], s=110, color=col, edgecolor="white", zorder=6)
    plt.annotate("SNR=0 at %.2f Hz"%bx, xy=(bx,0), xytext=(bx, 6 if cn0>0 else -10),
                 color=col, fontsize=11, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.3))
ax=plt.gca(); ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.4)
ax.axhspan(0,25, color=COL["green"], alpha=0.05); ax.axhspan(-58,0, color=COL["red"], alpha=0.06)
ax.axhline(0, color=COL["amber"], lw=1.4, ls="--"); ax.axvline(1, color=COL["dim"], lw=1, ls=":")
ax.text(1.1,-52,"1 Hz\n(the ruler)", color=COL["dim"], fontsize=11)
ax.set_xlim(0.1,1e5); ax.set_ylim(-58,25)
ax.set_xlabel("effective noise bandwidth  B  (Hz)   --  same link, different ruler")
ax.set_ylabel("signal-to-noise  (dB)")
ax.set_title("WHY 1 Hz:  IT IS THE RULER, NOT THE RECEIVER", fontsize=17, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=11)
ax.text(0.015,0.05,
        "1 Hz states noise as a density, so the\n"
        "number describes the planet bounce.\n"
        "It is how we measured the CAMRAS echoes:\n"
        "CNR in a 1 Hz band, set by the +/-0.75 Hz\n"
        "Doppler spread. Effelsberg lifts it +13.3 dB.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=COL["panel"], ec=COL["green2"], lw=1.2))
terminal_frame(ax, tag="ORI EVE // measured in 1 Hz")
plt.tight_layout(); plt.savefig("../figs/fig_why_1hz.png"); print("saved fig_why_1hz.png")
