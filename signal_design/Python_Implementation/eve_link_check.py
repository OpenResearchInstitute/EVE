#!/usr/bin/env python3
"""
eve_link_check.py -- does Pete's waveform close the link at the C/N0 we have?

Reproduces the "RX Monte Carlo for C/N0 = 0 dB" that sized T_sym. Models the actual
receiver: per symbol, take N frames of length 1/R_bw (coherent within a frame, which
is ~ the Venus coherence time), FFT each, sum the frame powers NON-coherently, pick
the peak among the M candidate tone bins. In AWGN with a tone exactly on an FFT bin,
each candidate bin power after combining is:
    true tone bin : 0.5 * noncentral_chi2(2N, lambda = 2*N*gamma_f)
    other bins    : 0.5 * central_chi2(2N)
with per-frame bin SNR gamma_f = (C/N0 linear) / R_bw. Symbol error if any of the
M-1 other bins beats the true bin. This is exact for AWGN; real Doppler spread adds a
few dB of intra-frame loss (frame length is set = coherence time to bound it), so treat
these as optimistic-by-a-few-dB AWGN references, same as any WSJT-mode sensitivity curve.

FER (frame error rate): a wrong M-ary symbol flips ~6 of its 12 bits, far more than
BCH(127,106) t=3 can fix, so a frame needs all 11 symbols right:
    FER ~= 1 - (1 - SER)**11   (BCH only rescues rare single-bit slips; ignored here).
"""
import numpy as np
import eve_params                       # R_bw / spacing from (freq, target)

# No hardcoded R_bw. It comes from eve_params for the (freq, target) under test.
# The defaults below are a Venus @ 1299.5 MHz REFERENCE so the module runs standalone;
# __main__ recomputes them for whatever --rf/--target you pass.
M = 4096
N_SYM = 11
_P0   = eve_params.compute(1299.5e6, "venus", M)
R_BW  = _P0.R_bw_hz                     # reference R_bw (parameter default only)
T_SYM = 164.794                         # Venus C/N0=0 reference symbol length
N_FRAMES = int(round(T_SYM * R_BW))

def ser_montecarlo(cn0_db, n_frames=N_FRAMES, m=M, r_bw=R_BW, trials=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    cn0 = 10 ** (cn0_db / 10.0)
    gamma_f = cn0 / r_bw
    df = 2 * n_frames
    z_true = 0.5 * rng.noncentral_chisquare(df, 2 * n_frames * gamma_f, size=trials)
    errs = 0
    # max over m-1 competing bins, chunked to bound memory
    for i in range(trials):
        z_false_max = 0.5 * rng.chisquare(df, size=m - 1).max()
        if z_false_max >= z_true[i]:
            errs += 1
    return errs / trials

def end_to_end_check(cn0_db, n_frames, m=M, r_bw=R_BW, trials=200, fs=48000.0, rng=None):
    """Genuine synth+detect cross-check (small n_frames to stay cheap)."""
    rng = rng or np.random.default_rng(1)
    cn0 = 10 ** (cn0_db / 10.0)
    nfft = int(round(fs / r_bw))            # samples per frame
    A = 1.0
    N0 = A * A / cn0                        # noise density for carrier power A^2
    sigma2 = N0 * fs                        # complex noise variance per sample
    cand_bins = (np.arange(m) * 2) % nfft   # tone d -> bin 2d (spacing = 2*R_bw)
    errs = 0
    n = np.arange(nfft * n_frames)
    for _ in range(trials):
        d = rng.integers(0, m)
        f = d * 2 * r_bw
        s = A * np.exp(2j * np.pi * f * n / fs)
        noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size))
        x = (s + noise).reshape(n_frames, nfft)
        P = (np.abs(np.fft.fft(x, axis=1)) ** 2).sum(axis=0)   # non-coherent combine
        dhat = np.argmax(P[cand_bins])
        errs += (dhat != d)
    return errs / trials

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EVE link-compliance Monte Carlo (FER vs C/N0).")
    ap.add_argument("--rf", type=float, default=1299.5e6, help="RF center freq (Hz)")
    ap.add_argument("--target", default="venus", help="target body (venus|moon)")
    ap.add_argument("--tsym", type=float, default=T_SYM, help="symbol seconds to size N_frames")
    args = ap.parse_args()

    P = eve_params.compute(args.rf, args.target, M)
    r_bw = P.R_bw_hz
    n_frames = int(round(args.tsym * r_bw))
    print("%s @ %.4f MHz: R_bw=%.3f Hz, coherence=%.3f s, t_sym=%.1f s -> %d frames/symbol"
          % (args.target, args.rf/1e6, r_bw, P.coherence_time_s, args.tsym, n_frames))
    # cross-validate abstract model vs real synth+detect at small N
    for cn0 in (0.0,):
        for N in (30,):
            a = ser_montecarlo(cn0, n_frames=N, r_bw=r_bw, trials=2000)
            e = end_to_end_check(cn0, n_frames=N, r_bw=r_bw, trials=300)
            print("cross-check  C/N0=%+.1f N=%d : abstract SER=%.3f  end-to-end SER=%.3f"
                  % (cn0, N, a, e))
    print()
    print("%-10s %-12s %-12s %-10s" % ("C/N0(dBHz)", "SER", "FER(11 sym)", "verdict"))
    for cn0 in [-3, -2, -1.33, -1, 0, 0.645, 1, 2, 3, 11.94]:
        ser = ser_montecarlo(cn0, n_frames=n_frames, r_bw=r_bw, trials=3000)
        fer = 1 - (1 - ser) ** N_SYM
        verdict = "closes" if fer < 0.1 else ("marginal" if fer < 0.5 else "fails")
        print("%-10.2f %-12.4f %-12.4f %-10s" % (cn0, ser, fer, verdict))

def ser_fast(cn0_db, n_frames=N_FRAMES, m=M, r_bw=R_BW, trials=2500, coh_loss=1.0, rng=None):
    """Vectorized SER; coh_loss<1 models intra-frame Doppler-spread coherence loss."""
    rng = rng or np.random.default_rng(0)
    cn0 = 10 ** (cn0_db / 10.0)
    gamma_f = coh_loss * cn0 / r_bw
    df = 2 * n_frames
    z_true = 0.5 * rng.noncentral_chisquare(df, 2 * n_frames * gamma_f, size=trials)
    z_false_max = 0.5 * rng.chisquare(df, size=(trials, m - 1)).max(axis=1)
    return float(np.mean(z_false_max >= z_true))
