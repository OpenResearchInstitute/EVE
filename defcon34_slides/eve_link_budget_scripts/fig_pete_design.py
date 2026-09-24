"""fig_pete_design.py [Q3] -- Pete Wyckoff's "Venus Bounce Transmitter Spiral #2"
(June 2026), redrawn, with a verdict. All parameters VERIFIED from Pete's video slide.
Shows why each choice fits the Venus channel."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from eve_style import apply_style, terminal_frame, COL, SLIDE
apply_style()
fig, ax = plt.subplots(figsize=SLIDE)
ax.set_xlim(0,100); ax.set_ylim(0,60); ax.axis("off")

blocks=[("106 bits\n+CRC",8),("BCH\n(127,106)",24),("Map to\nM-ary=4096",42),("NCO\n(spiral)",60),("SigMF",76)]
for label,x in blocks:
    ax.add_patch(FancyBboxPatch((x-7,42),14,9, boxstyle="round,pad=0.3",
                 fc=COL["panel"], ec=COL["green"], lw=1.6))
    ax.text(x,46.5,label,ha="center",va="center",color=COL["text"],fontsize=11,fontweight="bold")
for i in range(len(blocks)-1):
    ax.add_patch(FancyArrowPatch((blocks[i][1]+7,46.5),(blocks[i+1][1]-7,46.5),
                 arrowstyle="-|>", mutation_scale=18, color=COL["green"], lw=2))
ax.text(84,46.5,"->  TX",ha="left",va="center",color=COL["green"],fontsize=12,fontweight="bold")

# verified parameters
params=("VERIFIED FROM PETE'S DESIGN (KA3WCA, June 2026)\n"
        "  R_bw = 2.87 Hz        = Doppler-spread forecast\n"
        "  T_sym = 164.794 s     (RX Monte Carlo, C/N0 = 0 dB)\n"
        "  TX BW = 22 kHz        = 4096 x 2 x R_bw\n"
        "  11 symbols            TX duration ~30 min 12.7 s\n"
        "  RX: White Rabbit + H-maser shared reference\n"
        "  RX: astropy removes Doppler shift + rate\n"
        "  RX: FFT at R_bw, combine 440 NON-coherently / symbol")
ax.text(3,36,params,ha="left",va="top",family="DejaVu Sans Mono",fontsize=11,color=COL["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=COL["panel"], ec=COL["green2"], lw=1.2))

why=("WHY IT FITS THE CHANNEL\n"
     "  frame = 1/R_bw ~ 0.35 s  = the coherence time\n"
     "  tone spacing 2*R_bw = 5.74 Hz > the spread\n"
     "     -> tones never smear together\n"
     "  440 non-coherent combines -> sensitivity\n"
     "     without exceeding coherence time\n"
     "  M=4096 orthogonal -> within a few dB of the\n"
     "     -1.59 dB power limit\n"
     "  VERDICT: well matched to a spinning Venus.")
ax.text(53,36,why,ha="left",va="top",family="DejaVu Sans Mono",fontsize=11,color=COL["cyan"],
        bbox=dict(boxstyle="round,pad=0.5", fc=COL["panel"], ec=COL["cyan"], lw=1.2))

ax.set_title("PETE'S SPIRAL: A DESIGN BUILT AROUND THE COHERENCE TIME",
             fontsize=16, fontweight="bold", color=COL["green"], pad=8)
fig.tight_layout(); fig.savefig("../figs/fig_pete_design.png"); print("saved fig_pete_design.png")
