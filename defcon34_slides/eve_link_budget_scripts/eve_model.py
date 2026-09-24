"""
eve_model.py -- faithful reconstruction of the ORI EVE link budget model, taken
verbatim (formula by formula) from Link_Budget_Modeling.pdf.

Purpose: the notebook only computes a C/N0 scan for DSES monostatic at 2304 MHz.
To say anything quantitative about Dwingeloo (1299.5 MHz) or Effelsberg, we must
run the model ourselves. This file reproduces the notebook's math EXACTLY and is
validated against the notebook's printed DSES numbers before we trust it on any
other site. If the validation below does not match, we do NOT use derived numbers.

IMPORTANT LABELING RULE: DSES outputs are reproductions of verified notebook values.
Dwingeloo / Effelsberg outputs are COMPUTED with this model and are NOT printed in
the notebook. The Venus albedo (rho_eff) was derived from a 2.385 GHz Magellan map;
using it at Dwingeloo's 1299.5 MHz is an approximation the notebook does not verify.
"""
import numpy as np

C = 299792458.0          # m/s          (notebook)
K = 1.380649e-23         # J/K          (notebook)
R_V_KM = 6051.8          # Venus radius (notebook)
BW_HZ = 100e3            # receiver_noise_bandwidth (notebook)

# noise-model defaults (notebook get_noise_temperature_summary defaults)
MAIN_BEAM_EFF = 0.69
SPILLOVER_EFF = 0.95
SURFACE_RMS_MM = 3.0
GROUND_TEMP = 290.0
WEATHER = {"clear": 1.0, "cloudy": 1.5, "rain": 3.0}


def wavelength(freq_mhz):
    return C / (freq_mhz * 1e6)


def antenna_gain_db(diameter_m, efficiency, freq_mhz):
    lam = wavelength(freq_mhz)
    return 10 * np.log10(efficiency * (np.pi * diameter_m / lam) ** 2)


def fsl_roundtrip_db(distance_km, freq_mhz):
    lam = wavelength(freq_mhz)
    one_way = 20 * np.log10(4 * np.pi * distance_km * 1000.0 / lam)
    return 2 * one_way


def venus_gain_db(freq_mhz):
    lam = wavelength(freq_mhz)
    r_m = R_V_KM * 1000.0
    sigma_geom = np.pi * r_m ** 2
    return 10 * np.log10(4 * np.pi * sigma_geom / lam ** 2)


def venus_loss_db(rho):
    return 10 * np.log10(rho)


def beamwidth_rad(freq_mhz, tx_diameter_m):
    return 1.22 * wavelength(freq_mhz) / tx_diameter_m


def pointing_loss_db(pointing_error_deg, freq_mhz, tx_diameter_m):
    err = np.radians(pointing_error_deg)
    return -12 * (err / beamwidth_rad(freq_mhz, tx_diameter_m)) ** 2


def sky_noise_k(freq_mhz, elevation_deg, weather="clear"):
    elev = np.radians(elevation_deg)
    air_mass = 1.0 / np.sin(elev)
    freq_ghz = freq_mhz / 1000.0
    freq_factor = 0.1 * freq_ghz / 10.0
    mult = WEATHER.get(weather, 1.0)
    return 2.7 + (270 * (1 - np.exp(-freq_factor * air_mass))) * mult


def scatter_noise_k(freq_mhz, surface_rms_mm=SURFACE_RMS_MM):
    wl_mm = 300000.0 / freq_mhz
    surf_eff = np.exp(-((4 * np.pi * surface_rms_mm / wl_mm) ** 2))
    return 290 * (1 - surf_eff)


def t_sys_k(freq_mhz, elevation_deg, receiver_temp_k, weather="clear"):
    sky = sky_noise_k(freq_mhz, elevation_deg, weather)
    spill = GROUND_TEMP * (1 - SPILLOVER_EFF)          # 14.5 K
    scat = scatter_noise_k(freq_mhz)
    t_ant = MAIN_BEAM_EFF * sky + spill + scat
    return t_ant + receiver_temp_k


def lna_k(nf_db):
    return 290 * (10 ** (nf_db / 10.0) - 1)


def link_budget(distance_km, rho, freq_mhz, elevation_deg,
                tx_power_w, tx_diameter_m, tx_eff,
                rx_diameter_m, rx_eff, rx_receiver_temp_k,
                tx_line_db=0.5, rx_line_db=0.5,
                pointing_error_deg=0.01, weather="clear", tsys_override=None):
    """Return (cnr_db_1hz, detail dict). Physically single-frequency (radar bounce)."""
    tx_power_dbw = 10 * np.log10(tx_power_w)
    tx_g = antenna_gain_db(tx_diameter_m, tx_eff, freq_mhz)
    rx_g = antenna_gain_db(rx_diameter_m, rx_eff, freq_mhz)
    fsl = fsl_roundtrip_db(distance_km, freq_mhz)
    vg = venus_gain_db(freq_mhz)
    vl = venus_loss_db(rho)
    pl = pointing_loss_db(pointing_error_deg, freq_mhz, tx_diameter_m)
    rx_power = (tx_power_dbw + tx_g + rx_g - fsl - tx_line_db - rx_line_db
                + pl + vl + vg)
    tsys = tsys_override if tsys_override is not None else t_sys_k(freq_mhz, elevation_deg, rx_receiver_temp_k, weather)
    noise_dbw = 10 * np.log10(K * tsys * BW_HZ)
    cnr = rx_power - noise_dbw
    cnr_1hz = cnr + 10 * np.log10(BW_HZ)
    return cnr_1hz, dict(tx_power_dbw=tx_power_dbw, tx_gain=tx_g, rx_gain=rx_g,
                         fsl=fsl, venus_gain=vg, venus_loss=vl, pointing=pl,
                         rx_power=rx_power, t_sys=tsys, noise_dbw=noise_dbw,
                         cnr_db=cnr, cnr_db_1hz=cnr_1hz)


# ---------------------------------------------------------------------------
# VALIDATION against verified notebook DSES numbers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== VALIDATION vs verified notebook DSES values ===\n")

    # Static min-distance case: 38 Mkm, 30 deg, albedo 0.152.
    # Notebook: tx_gain 51.29, rx_gain 51.29, fsl 502.59, venus_gain 169.31,
    #           venus_loss -8.18, T_sys 76.05, rx_power -208.12, noise -159.79,
    #           cnr_db -48.33, cnr_db_1hz +1.67
    dses_lna = lna_k(0.4)
    c1, d1 = link_budget(38e6, 0.152, 2304.0, 30.0,
                         1500.0, 18.29, 0.69, 18.29, 0.69, dses_lna)
    print("STATIC 38 Mkm / 0.152 / 30 deg  (notebook -> reproduce):")
    print("  tx_gain   %.2f  (nb 51.29)" % d1["tx_gain"])
    print("  fsl       %.2f  (nb 502.59)" % d1["fsl"])
    print("  venus_gain%.2f  (nb 169.31)" % d1["venus_gain"])
    print("  venus_loss%.2f  (nb -8.18)" % d1["venus_loss"])
    print("  T_sys     %.2f  (nb 76.05)" % d1["t_sys"])
    print("  rx_power  %.2f  (nb -208.12)" % d1["rx_power"])
    print("  noise_dbw %.2f  (nb -159.79)" % d1["noise_dbw"])
    print("  cnr_db    %.2f  (nb -48.33)" % d1["cnr_db"])
    print("  cnr_1hz   %+.2f  (nb +1.67)\n" % d1["cnr_db_1hz"])

    # Dynamic best day: 40.82 Mkm, rho 0.1170, 34.4 deg -> notebook C/N0 -0.66, T_sys 75.1
    c2, d2 = link_budget(40.82e6, 0.1170, 2304.0, 34.4,
                         1500.0, 18.29, 0.69, 18.29, 0.69, dses_lna)
    print("DYNAMIC best day 40.82 Mkm / 0.117 / 34.4 deg (notebook -> reproduce):")
    print("  T_sys     %.2f  (nb 75.1)" % d2["t_sys"])
    print("  cnr_1hz   %+.2f  (nb -0.66)" % d2["cnr_db_1hz"])
