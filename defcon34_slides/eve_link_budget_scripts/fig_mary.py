"""fig_mary.py -- why M-ary orthogonal signaling (Pete's M=4096) is a good candidate:
it is power-efficient. Required Eb/N0 falls as M grows, toward the ultimate orthogonal
limit of -1.59 dB. Curve = non-coherent orthogonal union bound at symbol error 1e-3
(Ps <= (M-1)/2 * exp(-(k*Eb/N0)/2), k=log2 M). COMPUTED estimate, labeled as such."""
import numpy as np, matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE
apply_style()
Ps=1e-3
Ms=2**np.arange(1,14)              # 2 .. 8192
k=np.log2(Ms)
gamma=(2/k)*np.log((Ms-1)/(2*Ps))  # linear Eb/N0
EbN0=10*np.log10(gamma)
fig, ax = plt.subplots(figsize=SLIDE)
ax.grid(True, which="both", alpha=0.4)
ax.plot(np.log2(Ms), EbN0, "-o", color=COL["green"], lw=3, ms=6, label="non-coherent M-ary orthogonal (union bound, Ps=1e-3)")
ax.axhline(-1.59, color=COL["amber"], lw=1.8, ls="--")
ax.text(1.2,-1.1,"ultimate orthogonal limit  -1.59 dB", color=COL["amber"], fontsize=12, fontweight="bold")
# mark M=4096
i=np.where(Ms==4096)[0][0]
ax.scatter([12],[EbN0[i]], s=150, color=COL["cyan"], edgecolor="white", zorder=6)
ax.annotate("Pete: M=4096\n~%.1f dB Eb/N0\n(~%.1f dB above the limit)"%(EbN0[i],EbN0[i]+1.59),
            xy=(12,EbN0[i]), xytext=(6.5,EbN0[i]+3.5), color=COL["cyan"], fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL["cyan"], lw=1.5))
ax.scatter([1],[EbN0[0]], s=110, color=COL["red"], edgecolor="white", zorder=6)
ax.annotate("binary FSK\n~%.1f dB"%EbN0[0], xy=(1,EbN0[0]), xytext=(2.2,EbN0[0]+1.0),
            color=COL["red"], fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL["red"], lw=1.3))
ax.set_xlabel("bits per symbol  log2(M)"); ax.set_ylabel("required Eb/N0  (dB)")
ax.set_xlim(0,14); ax.set_ylim(-3,12)
ax.set_title("WHY M-ARY ORTHOGONAL: BIG M BUYS POWER EFFICIENCY",
             fontsize=16, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=10.5)
ax.text(0.015,0.06,
        "Power-limited link -> spend bandwidth to save power.\n"
        "M=4096 needs ~7 dB less Eb/N0 than binary, and sits\n"
        "within a few dB of the -1.59 dB ceiling. Bandwidth is\n"
        "cheap here (22 kHz); sensitivity is everything.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=COL["panel"], ec=COL["green2"], lw=1.2))
terminal_frame(ax, tag="ORI EVE // union-bound estimate")
fig.tight_layout(); fig.savefig("../figs/fig_mary.png"); print("saved fig_mary.png M=4096 -> %.2f dB"%EbN0[i])
