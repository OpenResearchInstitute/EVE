"""
compute_sites.py -- run the VALIDATED eve_model over the conjunction window for the
three configurations that matter to the DEF CON narrative.

DSES numbers are reproductions of verified notebook output. Dwingeloo and Effelsberg
numbers are COMPUTED with the validated model and are NOT printed in the notebook.

Frequency: Dwingeloo is constrained to 1299.5 MHz (feed focus issue), so the whole
Dwingeloo bounce -- transmit and any receiver of its echo -- is at 1299.5 MHz.

Per-date distance and rho_eff are verified and site-independent (Venus geometry).
Per-date elevation is DSES-derived and used as an approximation for the European
sites; at 1299.5 MHz the C/N0 sensitivity to elevation over 20-40 deg is < 0.3 dB.
The albedo rho_eff was derived from a 2.385 GHz map; its use at 1299.5 MHz is an
approximation the notebook does not verify (flagged in the report).
"""
import numpy as np
from eve_model import link_budget, lna_k
from eve_conjunction_data import SCAN, BEST_DAY

DWIN_LNA = lna_k(0.629)   # 45.20 K
EFFEL_LNA = lna_k(0.3)    # 20.74 K
DSES_LNA = lna_k(0.4)     # 27.98 K

configs = {
    # name: dict of params for link_budget (freq, tx_power_w, tx_D, tx_eff, rx_D, rx_eff, rx_temp, pointing)
    "DSES mono 2304 (verified)": dict(freq=2304.0, txw=1500.0, txD=18.29, rxD=18.29,
                                      rxtemp=DSES_LNA, point=0.01),
    "Dwingeloo alone 1299.5":    dict(freq=1299.5, txw=1000.0, txD=25.0, rxD=25.0,
                                      rxtemp=DWIN_LNA, point=0.01),
    "Dwingeloo tx + Effelsberg rx 1299.5": dict(freq=1299.5, txw=1000.0, txD=25.0, rxD=100.0,
                                                rxtemp=EFFEL_LNA, point=0.01),
}


def run(cfg):
    out = []
    for (date, rho, elev, dist_mkm, _, _) in SCAN:
        c1, d = link_budget(dist_mkm * 1e6, rho, cfg["freq"], elev,
                            cfg["txw"], cfg["txD"], 0.69, cfg["rxD"], 0.69,
                            cfg["rxtemp"], pointing_error_deg=cfg["point"])
        out.append((date, c1, d))
    return out


print("%-38s %10s %8s" % ("config", "best C/N0", "best day"))
print("-" * 60)
results = {}
for name, cfg in configs.items():
    r = run(cfg)
    best = max(r, key=lambda x: x[1])
    results[name] = r
    print("%-38s %+8.2f  %s" % (name, best[1], best[0]))

# Effelsberg advantage (albedo/elevation independent part is the gain+noise delta)
print("\n=== Effelsberg advantage over Dwingeloo-alone (same date) ===")
dwin = {d: c for d, c, _ in results["Dwingeloo alone 1299.5"]}
effel = {d: c for d, c, _ in results["Dwingeloo tx + Effelsberg rx 1299.5"]}
adv = [effel[d] - dwin[d] for d in dwin]
print("delta C/N0 = %.2f dB (constant across window)" % np.mean(adv))
print("  (std across window: %.3f dB)" % np.std(adv))

# Best-day detail for the two 1299.5 configs
print("\n=== best-day detail (2026-10-25) ===")
for name in ("Dwingeloo alone 1299.5", "Dwingeloo tx + Effelsberg rx 1299.5"):
    for date, c1, d in results[name]:
        if date == BEST_DAY:
            print("\n%s:" % name)
            print("  C/N0 1Hz  %+.2f dB-Hz" % c1)
            print("  tx_gain   %.2f  rx_gain %.2f" % (d["tx_gain"], d["rx_gain"]))
            print("  fsl       %.2f  venus_gain %.2f  venus_loss %.2f"
                  % (d["fsl"], d["venus_gain"], d["venus_loss"]))
            print("  T_sys     %.2f K  rx_power %.2f dBW" % (d["t_sys"], d["rx_power"]))
