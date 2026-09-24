"""
fig_crew.py -- the crew and the timing. Named sites (per user direction):
  Dwingeloo (NL) 25 m  -- THE CREW, transmits + receives at 1299.5 MHz
  Effelsberg (DE) 100 m -- PHONE A FRIEND, the big cold backup receiver
  Bochum (DE) 20 m      -- 2009 ORIGIN (dead-carrier bounce), shown faded as history
Diameters verified from site dataclasses. Apertures drawn to scale.
Timing: Earth and Venus line up only about every 18 months (inferior conjunction);
unlike the Moon, Venus is barely close enough for even the largest amateur dishes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from eve_style import apply_style, terminal_frame, COL, SLIDE

apply_style()
crew = [
    ("Bochum",     "DE - 2009 origin", 20.0,  COL["dim"],  "carrier bounce\n(history)"),
    ("Dwingeloo",  "NL - the crew",    25.0,  COL["green"],"tx + rx\n1299.5 MHz"),
    ("Effelsberg", "DE - phone a friend",100.0,COL["cyan"], "big cold rx\nsaves the day"),
]
fig, ax = plt.subplots(figsize=SLIDE)
ax.set_xlim(-6, 106); ax.set_ylim(0, 74); ax.axis("off")
maxd = 100.0
xs = [18, 45, 82]
for x,(name,loc,d,color,note) in zip(xs, crew):
    r = (d/maxd)*20
    base=20
    ax.add_patch(Wedge((x,base), r, 20, 160, width=r*0.32, facecolor=color,
                       edgecolor="white", alpha=0.85, lw=0.8))
    ax.plot([x],[base+r*0.9], marker="o", ms=5, color=COL["amber"])
    ax.plot([x,x],[base,base+r*0.9], color=COL["dim"], lw=1)
    ax.text(x, base-4, "%s\n%s"%(name,loc), ha="center", va="top", color=color,
            fontsize=12.5, fontweight="bold")
    ax.text(x, base-13, "%.0f m"%d, ha="center", va="top", color=COL["dim"], fontsize=11)
    ax.text(x, base+r+3, note, ha="center", va="bottom", color=COL["dim"], fontsize=10)
ax.set_title("THE CREW AND THE TIMING",
             fontsize=17, fontweight="bold", color=COL["green"], pad=6)
ax.text(0.5, 0.055,
        "Venus lines up only about every 18 months (inferior conjunction). "
        "Unlike the Moon, it is\nbarely close enough for even the largest amateur dishes. "
        "Timing and coordination matter\nas much as the radio.",
        transform=ax.transAxes, ha="center", va="bottom", color=COL["text"], fontsize=12)
fig.tight_layout(); fig.savefig("../figs/fig_crew.png")
print("saved fig_crew.png")
