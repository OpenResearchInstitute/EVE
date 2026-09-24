"""
fig_handoff.py -- the spin damages information -> over to Pete.
Venus spins, so the echo of a clean tone comes back SMEARED (Doppler spread,
CAMRAS measured +/-0.75 Hz, Estevez March 2025) and drifting. That breaks the
coherent-integration assumption every stock weak-signal amateur mode relies on.
So no existing amateur protocol closes this link: a custom waveform is required.
"""
import numpy as np
import matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE

apply_style()
f = np.linspace(-6, 6, 1000)
tx = np.exp(-(f+3)**2/(2*0.05**2)); tx/=tx.max()
rx = np.exp(-(f-1.5)**2/(2*1.0**2)); rx/=rx.max(); rx*=0.55

fig, ax = plt.subplots(figsize=SLIDE)
ax.grid(True, alpha=0.35)
ax.fill_between(f,0,tx, color=COL["green"], alpha=0.85); ax.plot(f,tx,color=COL["green"],lw=2)
ax.text(-3,1.06,"WE SEND\nclean tone", ha="center", color=COL["green"], fontsize=13, fontweight="bold")
ax.fill_between(f,0,rx, color=COL["red"], alpha=0.5); ax.plot(f,rx,color=COL["red"],lw=2)
ax.text(1.5,0.63,"WE GET BACK\nsmeared + drifting", ha="center", color=COL["red"], fontsize=13, fontweight="bold")
ax.annotate("", xy=(1.5-0.9,0.12), xytext=(1.5+0.9,0.12),
            arrowprops=dict(arrowstyle="<|-|>", color=COL["amber"], lw=1.8))
ax.text(1.5,0.045,"Doppler SPREAD  +/-0.75 Hz  (Venus spins)", ha="center",
        color=COL["amber"], fontsize=12, fontweight="bold")
ax.set_xlim(-6,6); ax.set_ylim(0,1.25); ax.set_yticks([])
ax.set_xlabel("frequency offset (conceptual; spread value verified)")
ax.set_title("THE SPIN DAMAGES INFORMATION   ->   OVER TO PETE",
             fontsize=16, fontweight="bold", pad=12)
ax.text(0.985, 0.93,
        "No stock mode (WSPR, FST4W, Q65) survives this:\n"
        "they assume a stable tone for hundreds of seconds.\n"
        "Venus will not give them one.\n"
        "WHAT NEEDED DEVELOPING: a spread-tolerant waveform.\n"
        "WHAT IS NEXT: Pete's signal design closes the link.",
        transform=ax.transAxes, ha="right", va="top", fontsize=11, color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=COL["panel"], ec=COL["green2"], lw=1.2))
terminal_frame(ax, tag="ORI EVE // spread verified // CAMRAS 2025")
fig.tight_layout(); fig.savefig("../figs/fig_handoff.png")
print("saved fig_handoff.png")
