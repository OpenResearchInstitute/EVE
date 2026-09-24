"""fig_zadoff.py [Q2] -- is Zadoff-Chu a good candidate? ZC = constant-amplitude
zero-autocorrelation (CAZAC) chirp -- the 'spiral'. Left: the ZC chirp's instantaneous
frequency ramp (a swept tone). Right: its periodic autocorrelation -- a single sharp
spike, which is what makes it superb for finding the echo and its delay/Doppler.
Illustrative sequence (length 353, root 1)."""
import numpy as np, matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE
apply_style()
N=353; u=1
n=np.arange(N)
zc=np.exp(-1j*np.pi*u*n*(n+1)/N)          # Zadoff-Chu, odd N
# periodic autocorrelation
ac=np.array([np.abs(np.sum(zc*np.conj(np.roll(zc,-s)))) for s in range(N)])/N
inst_f=np.diff(np.unwrap(np.angle(zc)))/(2*np.pi)  # instantaneous freq (cycles/sample)

fig, axes = plt.subplots(1,2,figsize=SLIDE)
ax=axes[0]
ax.plot(n[:-1], inst_f, color=COL["green"], lw=2.5)
ax.set_title("THE SPIRAL: A SWEPT TONE (constant amplitude)", color=COL["green"], fontsize=13, fontweight="bold")
ax.set_xlabel("sample n"); ax.set_ylabel("instantaneous frequency")
ax.grid(True, alpha=0.4); terminal_frame(ax)
ax.text(0.5,0.06,"|x[n]| = 1 for all n  ->  loves high-power TX", transform=ax.transAxes,
        ha="center", color=COL["dim"], fontsize=11)

ax=axes[1]
ax.plot(n, ac, color=COL["cyan"], lw=2)
ax.set_title("ONE SHARP SPIKE (CAZAC autocorrelation)", color=COL["cyan"], fontsize=13, fontweight="bold")
ax.set_xlabel("delay (samples)"); ax.set_ylabel("|autocorrelation|")
ax.grid(True, alpha=0.4); ax.set_ylim(0,1.1); terminal_frame(ax)
ax.text(0.5,0.55,"perfect for finding the echo\nand its delay + Doppler", transform=ax.transAxes,
        ha="center", color=COL["dim"], fontsize=11)

fig.suptitle("ZADOFF-CHU: GREAT FOR ACQUISITION; STILL BOUND BY COHERENCE TIME",
             fontsize=15, fontweight="bold", color=COL["green"], y=0.99)
fig.text(0.5,0.005,
  "Verdict: strong candidate for SYNC/ACQUISITION (constant envelope, sharp autocorrelation, "
  "chirp tolerates the big Doppler SHIFT) and as an M-ary alphabet -- but coherent correlation "
  "still cannot exceed T_c, so it combines non-coherently just like Pete's M-FSK.",
  ha="center", va="bottom", color=COL["text"], fontsize=10.5)
fig.tight_layout(rect=[0,0.03,1,0.95]); fig.savefig("../figs/fig_zadoff.png")
print("saved fig_zadoff.png  peak-to-sidelobe=%.0f"%(ac[0]/max(ac[1:])))
