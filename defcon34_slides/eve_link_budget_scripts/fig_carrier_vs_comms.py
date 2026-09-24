"""
fig_carrier_vs_comms.py -- the thesis of the talk. A dead carrier only has to be
DETECTED. A communications signal has to be DECODED, which needs more signal-to-
noise per bit AND has to survive a spinning reflector that smears information.

Qualitative/conceptual figure. The one hard number shown, the CAMRAS-measured
Doppler spread of about +/-0.75 Hz (Estevez, March 2025), is verified from the
notebook. The rest is framing.
"""
import numpy as np
import matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE

apply_style()

fig, axes = plt.subplots(1, 2, figsize=SLIDE)

# LEFT: dead carrier -- just detect a line
ax = axes[0]
f = np.linspace(-10, 10, 800)
tone = np.exp(-(f**2) / (2*0.15**2)); tone /= tone.max()
ax.fill_between(f, 0, tone, color=COL["green"], alpha=0.85)
ax.plot(f, tone, color=COL["green"], lw=2)
ax.set_title("2009  DEAD CARRIER", color=COL["green"], fontsize=15, fontweight="bold")
ax.text(0, 1.12, "just DETECT a line", ha="center", color=COL["text"], fontsize=13)
ax.text(0.5, 0.02,
        "one tone\nintegrate long enough\nand it appears\nLOW bar",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=12,
        color=COL["dim"])
ax.set_xlim(-10, 10); ax.set_ylim(0, 1.3); ax.set_yticks([])
ax.set_xlabel("frequency")
terminal_frame(ax)

# RIGHT: comms signal -- decode bits through a smearing channel
ax = axes[1]
# a little modulated spectrum, then smeared
base = (np.exp(-(f-2)**2/(2*0.6**2)) + np.exp(-(f+2)**2/(2*0.6**2))
        + 0.7*np.exp(-(f)**2/(2*0.6**2)))
base /= base.max()
smear = np.convolve(base, np.exp(-(np.linspace(-3,3,60))**2/2), mode="same")
smear /= smear.max()
ax.fill_between(f, 0, smear, color=COL["red"], alpha=0.5)
ax.plot(f, smear, color=COL["red"], lw=2)
ax.set_title("2026  COMMUNICATIONS SIGNAL", color=COL["red"], fontsize=15, fontweight="bold")
ax.text(0, 1.12, "must DECODE bits", ha="center", color=COL["text"], fontsize=13)
ax.text(0.5, 0.02,
        "needs Eb/N0 per bit (more than a carrier)\n"
        "AND Venus SPINS: +/-0.75 Hz spread\n"
        "smears the information\nHIGH bar",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5,
        color=COL["dim"])
ax.set_xlim(-10, 10); ax.set_ylim(0, 1.3); ax.set_yticks([])
ax.set_xlabel("frequency")
terminal_frame(ax)

fig.suptitle("THE REAL HACK: FROM DETECTING A TONE TO DECODING A MESSAGE",
             fontsize=16, fontweight="bold", color=COL["green"], y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("../figs/fig_carrier_vs_comms.png")
print("saved fig_carrier_vs_comms.png")
