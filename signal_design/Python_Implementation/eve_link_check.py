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

R_BW = 2.87
M = 4096
T_SYM = 164.794
N_SYM = 11
N_FRAMES = int(round(T_SYM * R_BW))     # 473

def ser_montecarlo(cn0_db, n_frames=N_FRAMES, m=M, trials=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    cn0 = 10 ** (cn0_db / 10.0)
    gamma_f = cn0 / R_BW
    df = 2 * n_frames
    z_true = 0.5 * rng.noncentral_chisquare(df, 2 * n_frames * gamma_f, size=trials)
    errs = 0
    # max over m-1 competing bins, chunked to bound memory
    for i in range(trials):
        z_false_max = 0.5 * rng.chisquare(df, size=m - 1).max()
        if z_false_max >= z_true[i]:
            errs += 1
    return errs / trials

def end_to_end_check(cn0_db, n_frames, m=M, trials=200, fs=48000.0, rng=None):
    """Genuine synth+detect cross-check (small n_frames to stay cheap)."""
    rng = rng or np.random.default_rng(1)
    cn0 = 10 ** (cn0_db / 10.0)
    nfft = int(round(fs / R_BW))            # samples per frame
    A = 1.0
    N0 = A * A / cn0                        # noise density for carrier power A^2
    sigma2 = N0 * fs                        # complex noise variance per sample
    cand_bins = (np.arange(m) * 2) % nfft   # tone d -> bin 2d
    errs = 0
    n = np.arange(nfft * n_frames)
    for _ in range(trials):
        d = rng.integers(0, m)
        f = d * 2 * R_BW
        s = A * np.exp(2j * np.pi * f * n / fs)
        noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size))
        x = (s + noise).reshape(n_frames, nfft)
        P = (np.abs(np.fft.fft(x, axis=1)) ** 2).sum(axis=0)   # non-coherent combine
        dhat = np.argmax(P[cand_bins])
        errs += (dhat != d)
    return errs / trials

if __name__ == "__main__":
    print("N_frames per symbol:", N_FRAMES)
    # cross-validate abstract model vs real synth+detect at small N
    for cn0 in (0.0,):
        for N in (30,):
            a = ser_montecarlo(cn0, n_frames=N, trials=2000)
            e = end_to_end_check(cn0, n_frames=N, trials=300)
            print("cross-check  C/N0=%+.1f N=%d : abstract SER=%.3f  end-to-end SER=%.3f"
                  % (cn0, N, a, e))
    print()
    print("%-10s %-12s %-12s %-10s" % ("C/N0(dBHz)", "SER", "FER(11 sym)", "verdict"))
    for cn0 in [-3, -2, -1.33, -1, 0, 0.645, 1, 2, 3, 11.94]:
        ser = ser_montecarlo(cn0, trials=3000)
        fer = 1 - (1 - ser) ** N_SYM
        verdict = "closes" if fer < 0.1 else ("marginal" if fer < 0.5 else "fails")
        print("%-10.2f %-12.4f %-12.4f %-10s" % (cn0, ser, fer, verdict))

def ser_fast(cn0_db, n_frames=N_FRAMES, m=M, trials=2500, coh_loss=1.0, rng=None):
    """Vectorized SER; coh_loss<1 models intra-frame Doppler-spread coherence loss."""
    rng = rng or np.random.default_rng(0)
    cn0 = 10 ** (cn0_db / 10.0)
    gamma_f = coh_loss * cn0 / R_BW
    df = 2 * n_frames
    z_true = 0.5 * rng.noncentral_chisquare(df, 2 * n_frames * gamma_f, size=trials)
    z_false_max = 0.5 * rng.chisquare(df, size=(trials, m - 1)).max(axis=1)
    return float(np.mean(z_false_max >= z_true))
