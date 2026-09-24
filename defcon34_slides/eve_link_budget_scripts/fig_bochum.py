"""
fig_bochum.py -- the 2009 origin: a dead carrier echoed off Venus.

VERIFIED (web sources: AMSAT-DL, ARRL, CAMRAS): 2009-03-25, 10:38 UTC, AMSAT-DL,
Bochum 20 m dish (IUZ Sternwarte), ~5 kW magnetron-based transmitter near 2.4 GHz,
round trip about 5 minutes, first time amateurs bounced a signal off another planet.
The signal was a pure carrier -- no information. This figure is a stylized
spectrogram: one bright tone line dug out of noise by long FFT integration.
"""
import numpy as np
import matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE

apply_style()
rng = np.random.default_rng(2009)

# fake spectrogram: noise + one bright vertical carrier line
nt, nf = 400, 300
spec = rng.normal(0.0, 1.0, (nf, nt))
carrier_bin = nf // 2
spec[carrier_bin-1:carrier_bin+2, :] += np.linspace(3, 9, nt)  # tone builds with integration

fig, ax = plt.subplots(figsize=SLIDE)
ax.imshow(spec, aspect="auto", cmap="Greens", origin="lower",
          extent=[0, 300, -150, 150], vmin=-1, vmax=8)
ax.axhline(0, color=COL["amber"], lw=0.8, ls=":", alpha=0.6)
ax.annotate("THE ECHO\na single dead carrier\noff the surface of Venus",
            xy=(210, 3), xytext=(150, 90), color=COL["amber"],
            fontsize=14, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL["amber"], lw=1.8))

ax.set_xlabel("integration time  ->  (longer FFT digs the tone out of the noise)")
ax.set_ylabel("frequency offset  (Hz)")
ax.set_title("2009: AMSAT-DL BOUNCES A CARRIER OFF VENUS",
             fontsize=17, fontweight="bold", pad=12)
ax.text(0.985, 0.05,
        "2009-03-25  Bochum 20 m dish\n"
        "~5 kW magnetron near 2.4 GHz\n"
        "round trip about 5 minutes\n"
        "FIRST amateur planetary bounce\n"
        "...but a carrier carries no information.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5,
        color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.55", fc=COL["panel"], ec=COL["green2"], lw=1.2))
terminal_frame(ax, tag="ORI EVE // history // verified")
fig.tight_layout()
fig.savefig("../figs/fig_bochum.png")
print("saved fig_bochum.png")
