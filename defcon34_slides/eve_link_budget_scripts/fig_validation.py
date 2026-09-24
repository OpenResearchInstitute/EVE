"""
fig_validation.py -- the credibility centerpiece: ORI's predicted CNR in 1 Hz vs the
MEASURED CNR in 1 Hz from the four CAMRAS Venus echoes, 2025-03-22.

All numbers VERIFIED from "Validation of the ORI Earth-Venus-Earth Link Budget"
(ORI, April 2026) and eve_cnr_measurement.ipynb:
    measured echoes: +1.455, -0.087, +0.285, +0.927 dB  (mean +0.645, sd 0.684)
    predicted (link budget): +0.560 dB
    residual (measured mean - predicted): +0.085 dB
    95% CI (t_3=3.182): +/-1.088 dB
This is only the second amateur EVE detection in history (after Bochum 2009).
"""
import numpy as np
import matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE

apply_style()
echoes = [("12:05:39", 1.455), ("12:15:39", -0.087),
          ("12:25:39", 0.285), ("12:35:39", 0.927)]
labels=[e[0] for e in echoes]; vals=[e[1] for e in echoes]
mean=0.645; pred=0.560; ci=1.088

fig, ax = plt.subplots(figsize=SLIDE)
ax.grid(True, axis="y", alpha=0.4)
x=np.arange(len(echoes))
ax.bar(x, vals, width=0.5, color=COL["green"], alpha=0.85, edgecolor="white",
       label="measured echo")
for xi,v in zip(x,vals):
    ax.text(xi, v+(0.05 if v>=0 else -0.13), "%+.3f"%v, ha="center",
            color=COL["text"], fontsize=12, fontweight="bold")

# predicted line + mean line + CI band
ax.axhline(pred, color=COL["cyan"], lw=2, ls="--", label="link budget PREDICTED  +0.560")
ax.axhline(mean, color=COL["amber"], lw=2, label="measured MEAN  +0.645")
ax.axhspan(mean-ci, mean+ci, color=COL["amber"], alpha=0.07)
ax.text(3.35, mean+ci-0.1, "95%% CI +/-%.3f"%ci, color=COL["amber"], fontsize=10,
        ha="right", va="top")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
ax.set_xlabel("CAMRAS Venus echo,  2025-03-22  (four 278 s carrier bursts)")
ax.set_ylabel("CNR in 1 Hz   (dB)")
ax.set_xlim(-0.6, 3.6); ax.set_ylim(-1.6, 2.9)
ax.set_title("WE DID NOT MODEL THIS. WE MEASURED IT OFF VENUS.",
             fontsize=16, fontweight="bold", pad=12)
ax.legend(loc="lower left", fontsize=11)
ax.text(0.985, 0.985,
        "Predicted   +0.560 dB\n"
        "Measured    +0.645 dB\n"
        "Residual    +0.085 dB\n"
        "\n"
        "The ORI link budget is validated\n"
        "against real Venus echoes to < 1 dB.\n"
        "Only the 2nd amateur EVE ever.",
        transform=ax.transAxes, ha="right", va="top", family="DejaVu Sans Mono",
        fontsize=12, color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.6", fc=COL["panel"], ec=COL["green2"], lw=1.2))
terminal_frame(ax, tag="ORI EVE // T_sys 65 K measured // verified")
fig.tight_layout(); fig.savefig("../figs/fig_validation.png")
print("saved fig_validation.png  mean=%.3f pred=%.3f residual=%.3f"%(mean,pred,mean-pred))
