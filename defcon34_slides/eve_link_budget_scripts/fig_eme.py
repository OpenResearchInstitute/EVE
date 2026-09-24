"""fig_eme.py [Q4] -- why the Moon is the ideal EVE testbed. Harsher Doppler spread on a
far easier link, available most nights. Moon numbers VERIFIED from EME literature
(K1JT 2010 w50 at 1296 MHz; ~271 dB path loss, ~7% albedo). Venus numbers computed."""
import numpy as np, matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE
apply_style()
fig, ax = plt.subplots(figsize=SLIDE)
ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)

rows=[
 ("",                    "MOON (EME)",            "VENUS (EVE)"),
 ("path loss, round trip","271 dB",               "494 dB  (223 dB harder)"),
 ("echo strength",        "~223 dB STRONGER",      "baseline"),
 ("Doppler spread @1296", "w50 = 6.2 Hz (K1JT)",   "~2.87 Hz forecast"),
 ("coherence time",       "~0.16 s (harsher)",     "~0.35 s"),
 ("albedo",               "~7%",                   "~13%"),
 ("availability",         "most nights",           "every ~18 months"),
 ("dishes / band / chain","Dwingeloo, Effelsberg,", "SAME"),
 ("",                     "Stockert @ 23 cm",      "SAME"),
]
y0=88; dy=8.7
for i,(a,b,c) in enumerate(rows):
    y=y0-i*dy
    head = (i==0)
    col = COL["green"] if head else COL["text"]
    ax.text(3,y,a,ha="left",va="center",color=COL["dim"] if not head else COL["green"],
            fontsize=12, fontweight="bold" if head else "normal")
    ax.text(42,y,b,ha="left",va="center",color=COL["cyan"] if head else COL["text"],
            fontsize=12, fontweight="bold" if head else "normal")
    ax.text(72,y,c,ha="left",va="center",color=COL["amber"] if head else COL["text"],
            fontsize=12, fontweight="bold" if head else "normal")
    if head:
        ax.plot([2,98],[y-dy*0.5,y-dy*0.5],color=COL["green2"],lw=1)

ax.text(50,6,
  "The Moon is a HARDER Doppler-spread channel on a FAR easier link, available tonight. "
  "It stress-tests\nthe coherence-time strategy and the whole pipeline before the once-in-18-months Venus shot.",
  ha="center", va="center", color=COL["green"], fontsize=12, fontweight="bold")
ax.set_title("PROVE IT ON THE MOON FIRST: EME AS THE EVE TESTBED",
             fontsize=16, fontweight="bold", color=COL["green"], pad=8)
fig.tight_layout(); fig.savefig("../figs/fig_eme.png"); print("saved fig_eme.png")
