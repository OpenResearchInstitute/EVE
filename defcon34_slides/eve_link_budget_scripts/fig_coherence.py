"""fig_coherence.py [Q1] -- Doppler spread -> coherence time, and the repercussion.
Venus rotation smears the echo of a pure tone. Measured spread (CAMRAS Mar 2025)
= +/-0.75 Hz; Pete's Oct forecast ~2.87 Hz. Coherence time T_c ~ 1/spread.
The repercussion: you cannot integrate COHERENTLY longer than T_c. Beyond it you must
combine NON-coherently, which gains only ~5*log10(N) instead of 10*log10(N) -- the
'coherence-time tax'. This is exactly why Pete chops each symbol into ~440 frames."""
import numpy as np, matplotlib.pyplot as plt
from eve_style import apply_style, terminal_frame, COL, SLIDE
apply_style()
Tc = 1/2.87                 # coherence time from Oct forecast spread (s)
T0 = 0.01                   # reference frame (s)
T = np.logspace(np.log10(T0), 3, 400)
ideal = 10*np.log10(T/T0)                          # if channel stayed coherent forever
ach = np.where(T<=Tc, 10*np.log10(T/T0),
               10*np.log10(Tc/T0)+5*np.log10(np.maximum(T/Tc,1)))  # coherent then non-coherent
fig, ax = plt.subplots(figsize=SLIDE)
ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.4)
ax.fill_between(T, ach, ideal, where=ideal>=ach, color=COL["red"], alpha=0.12)
ax.plot(T, ideal, color=COL["dim"], lw=2, ls="--", label="ideal coherent (10 log N) -- NOT allowed")
ax.plot(T, ach, color=COL["green"], lw=3, label="achievable: coherent to T_c, then non-coherent")
ax.axvline(Tc, color=COL["amber"], lw=1.8, ls="--")
ax.text(Tc*1.15, 3, "coherence time\nT_c ~ 1/spread ~ 0.35 s", color=COL["amber"], fontsize=12, fontweight="bold")
# Pete's symbol point
Tsym=164.794
gi=10*np.log10(Tsym/T0); ga=10*np.log10(Tc/T0)+5*np.log10(Tsym/Tc)
ax.scatter([Tsym],[ga], s=130, color=COL["cyan"], edgecolor="white", zorder=6)
ax.annotate("Pete's symbol: 165 s\n= 440 frames combined\nnon-coherently",
            xy=(Tsym,ga), xytext=(9,ga-9), color=COL["cyan"], fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL["cyan"], lw=1.4))
ax.annotate("", xy=(Tsym,ideal[-1]-0.5), xytext=(Tsym,ga+0.5),
            arrowprops=dict(arrowstyle="<|-|>", color=COL["red"], lw=1.8))
ax.text(Tsym*0.62, (gi+ga)/2, "coherence-time\ntax ~%.0f dB"%(gi-ga), color=COL["red"],
        fontsize=11, fontweight="bold", ha="right", va="center")
ax.set_xlim(T0,1000); ax.set_ylim(-5,48)
ax.set_xlabel("integration time  (s)   [log]")
ax.set_ylabel("processing gain  (dB)")
ax.set_title("THE REAL DAMAGE: THE SPIN CAPS COHERENT INTEGRATION",
             fontsize=16, fontweight="bold", pad=12)
ax.legend(loc="upper left", fontsize=11)
terminal_frame(ax, tag="ORI EVE // T_c from forecast 2.87 Hz")
fig.tight_layout(); fig.savefig("../figs/fig_coherence.png"); print("saved fig_coherence.png tax=%.1f dB"%(gi-ga))
