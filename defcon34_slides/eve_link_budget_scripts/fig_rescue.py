"""
fig_rescue.py -- HERO. October 2026 conjunction: Dwingeloo alone vs Dwingeloo +
Effelsberg, at 1299.5 MHz, with MEASURED Dwingeloo T_sys = 65 K (Telkamp) and
MODELED Effelsberg T_sys = 49 K (labeled). Dynamic albedo per date.

Computed with eve_model (reproduces the notebook DSES scan AND the CAMRAS
validation white paper to <0.01 dB). Best day 2026-10-25:
    Dwingeloo alone         -1.33 dB-Hz   (below zero)
    Dwingeloo + Effelsberg  +11.94 dB-Hz
    Effelsberg advantage    +13.27 dB
"""
import numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates
from datetime import datetime
from eve_style import apply_style, terminal_frame, COL, SLIDE
from eve_model import link_budget, lna_k
from eve_conjunction_data import SCAN, BEST_DAY

apply_style()
dates, dwin, effel = [], [], []
for (date, rho, elev, dist_mkm, _, _) in SCAN:
    dates.append(datetime.strptime(date, "%Y-%m-%d"))
    dwin.append(link_budget(dist_mkm*1e6, rho, 1299.5, elev, 1000.0,25.0,0.69,25.0,0.69, lna_k(0.629), tsys_override=65.0)[0])
    effel.append(link_budget(dist_mkm*1e6, rho, 1299.5, elev, 1000.0,25.0,0.69,100.0,0.69, lna_k(0.3), tsys_override=49.0)[0])
bi = [r[0] for r in SCAN].index(BEST_DAY)

fig, ax = plt.subplots(figsize=SLIDE)
ax.grid(True, alpha=0.5)
ax.axhspan(0,16, color=COL["green"], alpha=0.05); ax.axhspan(-6,0, color=COL["red"], alpha=0.06)
ax.plot(dates, effel, color=COL["cyan"], lw=3, zorder=5, label="Dwingeloo + Effelsberg (49 K modeled)")
ax.plot(dates, dwin, color=COL["green"], lw=3, zorder=5, label="Dwingeloo alone (65 K measured)")
ax.axhline(0, color=COL["amber"], lw=1.5, ls="--")
ax.annotate("", xy=(dates[bi], effel[bi]), xytext=(dates[bi], dwin[bi]),
            arrowprops=dict(arrowstyle="<|-|>", color=COL["magenta"], lw=2.2))
ax.text(dates[19], 4.8, "EFFELSBERG\n+13.27 dB\nphone a friend",
        color=COL["magenta"], fontsize=13, fontweight="bold", va="center", ha="left")
ax.scatter([dates[bi]],[dwin[bi]], s=150, color=COL["green"], edgecolor="white", zorder=6)
ax.scatter([dates[bi]],[effel[bi]], s=150, color=COL["cyan"], edgecolor="white", zorder=6)
ax.annotate("alone: -1.33 dB-Hz", xy=(dates[bi],dwin[bi]), xytext=(dates[13],-5.0),
            color=COL["green"], fontsize=12, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=COL["green"], lw=1.4))
ax.annotate("rescued: +11.94 dB-Hz", xy=(dates[bi],effel[bi]), xytext=(dates[4],14.3),
            color=COL["cyan"], fontsize=12, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=COL["cyan"], lw=1.4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
ax.set_ylim(-6,16)
ax.set_xlabel("date  (2026 inferior conjunction -- the 18-month window)")
ax.set_ylabel("C/N0 in 1 Hz   (dB-Hz)   at 1299.5 MHz")
ax.set_title("OCTOBER 2026: ALONE WE SINK, TOGETHER WE GET IN",
             fontsize=16, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=11)
ax.text(0.015,0.02,
        "Model validated to <0.1 dB vs March 2025\n"
        "CAMRAS echoes; October uses the harder\n"
        "dynamic albedo.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=COL["dim"])
terminal_frame(ax, tag="ORI EVE // 1299.5 MHz // T_sys 65K meas")
fig.tight_layout(); fig.savefig("../figs/fig_rescue.png")
print("saved fig_rescue.png  dwin=%.2f effel=%.2f gap=%.2f"%(dwin[bi],effel[bi],effel[bi]-dwin[bi]))
