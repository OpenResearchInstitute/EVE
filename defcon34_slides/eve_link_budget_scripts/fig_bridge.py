"""
fig_bridge.py -- link budget bridge, Dwingeloo October 2026 attempt, 1299.5 MHz,
best day, dynamic albedo 0.117, MEASURED T_sys = 65 K (Telkamp). Lands at -1.33 dB-Hz.
Effelsberg (100 m, modeled 49 K) adds +13.27 dB -> +11.94 dB-Hz.
Computed with the CAMRAS-validated model.
"""
import numpy as np, matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE
from eve_model import link_budget, lna_k
apply_style()
c1, d = link_budget(40.82e6, 0.117, 1299.5, 34.4, 1000.0,25.0,0.69,25.0,0.69, lna_k(0.629), tsys_override=65.0)
steps = [
    ("TX power\n1000 W",           d["tx_power_dbw"], "start"),
    ("TX gain\n25 m",              d["tx_gain"],      "gain"),
    ("Free-space\nloss (rt)",      -d["fsl"],         "loss"),
    ("Venus\nreflector",           d["venus_gain"],   "gain"),
    ("Venus albedo\n~13% back",    d["venus_loss"],   "loss"),
    ("RX gain\n25 m",              d["rx_gain"],      "gain"),
    ("Feedline",                   -1.00,             "loss"),
]
labels=[s[0] for s in steps]; deltas=[s[1] for s in steps]; kinds=[s[2] for s in steps]
totals=[]; run=0.0
for lab,dv,kind in steps:
    start=0.0 if kind=="start" else run; end=dv if kind=="start" else run+dv
    totals.append((start,end)); run=end
rx_power=run
fig, ax = plt.subplots(figsize=SLIDE); ax.grid(True, axis="y", alpha=0.4)
for i,((s,e),kind) in enumerate(zip(totals,kinds)):
    color=COL["green"] if kind in ("start","gain") else COL["red"]
    ax.bar(i, abs(e-s), bottom=min(s,e), width=0.62, color=color, alpha=0.85, edgecolor="white", linewidth=0.6)
    if i<len(steps)-1: ax.plot([i+0.31,i+1-0.31],[e,e],color=COL["dim"],lw=1,ls=":")
    dv=deltas[i]; sign="+" if dv>=0 and kind!="start" else ""
    ax.text(i, e+(7 if e>=s else -16), "%s%.2f"%(sign,dv), ha="center",
            color=(COL["green"] if kind in ("start","gain") else COL["red"]), fontsize=11, fontweight="bold")
ax.set_xticks(range(len(steps))); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("running total  (dB / dBW)"); ax.set_ylim(-450,120)
ax.axhline(rx_power, color=COL["cyan"], lw=1.4, ls="--")
ax.text(0.02, rx_power+8, "Dwingeloo RX power = %.2f dBW"%rx_power,
        transform=ax.get_yaxis_transform(), color=COL["cyan"], fontsize=12, fontweight="bold")
result=("DWINGELOO ALONE (T_sys 65 K measured)\n"
        "  C/N0 = -1.33 dB-Hz   below zero\n\n"
        "PHONE A FRIEND: Effelsberg 100 m\n"
        "  +13.27 dB  (aperture + cold rx)\n"
        "  C/N0 = +11.94 dB-Hz   we are in")
ax.text(0.985,0.97, result, transform=ax.transAxes, ha="right", va="top",
        family="DejaVu Sans Mono", fontsize=12, color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.6", fc=COL["panel"], ec=COL["green2"], lw=1.2))
ax.set_title("WHICH PART MATTERS MOST?  THE PATH LOSS CLIFF",
             fontsize=16, fontweight="bold", pad=12)
terminal_frame(ax, tag="ORI EVE // 1299.5 MHz // 65K measured")
fig.tight_layout(); fig.savefig("../figs/fig_bridge.png")
print("saved fig_bridge.png  rx_power=%.2f  cnr=%.2f"%(rx_power,c1))
