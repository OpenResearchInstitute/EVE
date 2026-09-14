#!/usr/bin/env python3
"""
eve_twt.py -- TWT (traveling-wave tube) amplifier model for the EVE waveform, and the
proof that Pete's constant-envelope M-ary FSK sails through a SATURATED tube undamaged
-- so DSES can replace 8x250 W PAs + phasing harness + combiner with one TWT run flat out.

Model: Saleh (1981) AM/AM + AM/PM, the standard TWT baseband nonlinearity.
    r = |x|
    A(r)   = aa * r / (1 + ba * r^2)          (gain compression -> saturation)
    Phi(r) = ap * r^2 / (1 + bp * r^2)          (amplitude-dependent phase)
    y = A(r) * exp( j*(angle(x) + Phi(r)) )
Classic Saleh coefficients (normalized, saturation near r=1):
    aa=2.1587 ba=1.1517 ap=4.0033 bp=9.1040
Saturation input r_sat = 1/sqrt(ba) = 0.932; drive the constant-envelope signal there
for maximum power with ZERO back-off.

Why it works: our signal is one pure tone at a time, so |x| is CONSTANT. Every sample
sees the SAME A(r) and SAME Phi(r): the tube applies one gain and one fixed phase
offset. A fixed phase is invisible to non-coherent per-symbol detection. No AM to
compress, no back-off, and one tone at a time means NO in-band intermodulation --
only out-of-band harmonics you filter after the tube.
"""
import numpy as np

SALEH = dict(aa=2.1587, ba=1.1517, ap=4.0033, bp=9.1040)
R_SAT = 1.0 / np.sqrt(SALEH["ba"])       # ~0.932 input amplitude at saturation

def saleh(x, p=SALEH):
    r = np.abs(x)
    A = p["aa"] * r / (1 + p["ba"] * r * r)
    Phi = p["ap"] * r * r / (1 + p["bp"] * r * r)
    ph = np.angle(x)
    return A * np.exp(1j * (ph + Phi))

def phase_noise(n, fs, ripple_hz=100.0, ripple_rad=0.02, floor_rad_rms=0.01, rng=None):
    """TWT phase noise: HV-supply ripple (a PM tone) + a broadband floor. Returns phase
    samples (radians). ripple_rad is the PM index of the ripple line; floor_rad_rms is
    the broadband rms over the record."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(n) / fs
    return ripple_rad * np.sin(2 * np.pi * ripple_hz * t) + floor_rad_rms * rng.standard_normal(n)

def through_twt(x, fs=None, pn=False, **pn_kw):
    y = saleh(x)
    if pn:
        y = y * np.exp(1j * phase_noise(y.size, fs, **pn_kw))
    return y

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import eve_tx_sigmf as tx
    fs = 48000.0; R = tx.R_BW
    # one constant-envelope symbol tone at saturation drive
    t_sym = 5.0; nsps = int(t_sym * fs); n = np.arange(nsps)
    d = 1234
    s = R_SAT * np.exp(2j * np.pi * (d * tx.SPACING) * n / fs)   # |s| = R_SAT (saturation)
    y = saleh(s)
    print("=== constant-envelope tone through SATURATED TWT ===")
    print("input  |x| : min %.4f max %.4f  (dead constant at saturation)" % (np.abs(s).min(), np.abs(s).max()))
    print("output |y| : min %.4f max %.4f  (still constant -> undistorted)" % (np.abs(y).min(), np.abs(y).max()))
    # is the tone still clean? FFT bin purity
    S = np.abs(np.fft.fft(y)); k = np.argmax(S)
    inband = S[k]; rest = np.sqrt(np.mean(np.delete(S, k) ** 2))
    print("tone/floor : %.0f dB  (single tone -> no in-band IMD)" % (20 * np.log10(inband / rest)))
    print("phase off  : constant %.3f rad (invisible to non-coherent detection)"
          % (np.angle(y[0]) - np.angle(s[0])))
    print()
    # contrast: TWO simultaneous tones (amplitude-varying) through the SAME tube -> IMD
    s2 = 0.5 * R_SAT * (np.exp(2j*np.pi*(1000*tx.SPACING)*n/fs) + np.exp(2j*np.pi*(2000*tx.SPACING)*n/fs))
    y2 = saleh(s2)
    print("=== two simultaneous tones (envelope VARIES) through the same tube ===")
    print("input  |x| : min %.4f max %.4f  (100%% AM -> the tube must distort it)"
          % (np.abs(s2).min(), np.abs(s2).max()))
    F = np.abs(np.fft.fft(y2)); binhz = fs / nsps
    f1, f2 = 1000*tx.SPACING, 2000*tx.SPACING
    imd = 2*f1 - f2                      # third-order product falls IN BAND
    b = lambda f: int(round(f/binhz))
    print("IMD3 spur  : at %.0f Hz is %.0f dB below carriers  (in-band garbage)"
          % (imd, 20*np.log10(F[b(f1)] / (F[b(imd)] + 1e-9))))
    print("\nConclusion: one tone at a time -> run the TWT saturated, full power, clean.")
