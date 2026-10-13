#!/usr/bin/env python3
"""
eve_rx.py -- EVE receiver. Turns a SigMF recording of the echo into a decoded message.
This is the "what do I do with the recording" answer for DSES.

Chain (matches Pete's RX notes):
  recording -> remove bulk Doppler shift + rate -> for each symbol: frame at 1/R_bw,
  FFT, sum frame powers NON-coherently, pick the peak among the M tone bins -> d_m ->
  de-map 11 symbols -> 127 coded bits -> BCH(127,106) decode -> 106 payload ->
  split 90 message + 16 CRC -> check CRC-16 -> recovered text.

Run the self-test (TX -> simulated Venus echo: delay + Doppler + AWGN -> decode):
    python3 eve_rx.py --selftest
Decode a real recording:
    python3 eve_rx.py rx.sigmf-meta --f-dopp <Hz> --f-rate <Hz/s> [--t0 <s>]
"""
import argparse, json, sys, numpy as np
import eve_tx_sigmf as tx   # reuse the coder + PROTOCOL-structure constants
import eve_params           # waveform params (R_bw, spacing) from (freq, target)
import galois

# Protocol-STRUCTURE constants are safe to import (fixed): M, symbols, coder.
# R_bw and tone spacing are NOT constants -- the RX computes them itself from the
# agreed (freq, target) via eve_params, exactly as the TX did. It never depends on
# the transmitter's file, which keeps the receive path causal.
M = tx.M; N_SYM = tx.N_SYMBOLS; BITS_SYM = tx.BITS_SYM
_bch = tx._bch

def doppler_correct(iq, fs, f_dopp=0.0, f_rate=0.0, t0=0.0):
    """De-rotate a constant Doppler shift and linear Doppler rate."""
    t = t0 + np.arange(iq.size) / fs
    return iq * np.exp(-1j * 2 * np.pi * (f_dopp * t + 0.5 * f_rate * t * t))

def detect_symbol(seg, fs, spacing, r_bw, freq_offset=0.0):
    """Non-coherent M-ary detection over one symbol. Returns (d_hat, metric).
    spacing, r_bw come from eve_params.compute(rf, target) -- same numbers the TX used."""
    nfft = int(round(fs / r_bw))                     # samples per coherent frame
    nfr = seg.size // nfft
    if nfr < 1:
        return 0, 0.0
    frames = seg[:nfr * nfft].reshape(nfr, nfft)
    P = (np.abs(np.fft.fft(frames, axis=1)) ** 2).sum(axis=0)   # non-coherent combine
    # candidate tone bins: f_d = d*spacing + offset  -> bin = round(f/ (fs/nfft))
    binhz = fs / nfft
    d = np.arange(M)
    bins = np.round((d * spacing + freq_offset) / binhz).astype(int) % nfft
    cand = P[bins]
    d_hat = int(np.argmax(cand))
    # metric: peak / mean of the rest (detection confidence)
    metric = cand[d_hat] / (np.median(cand) + 1e-12)
    return d_hat, float(metric)

def symbols_to_message(symbols):
    bits = np.zeros(N_SYM * BITS_SYM, dtype=np.uint8)
    for m, d in enumerate(symbols):
        for i in range(BITS_SYM):                    # MSB first
            bits[m * BITS_SYM + i] = (d >> (BITS_SYM - 1 - i)) & 1
    cw = bits[:tx.BCH_N]
    payload = np.array(_bch.decode(galois.GF2(cw)), dtype=np.uint8)   # 106 bits
    if payload.size != tx.PAYLOAD_BITS:              # decode failure returns -1 array
        return None, False, payload
    msg, crc_rx = payload[:tx.MSG_BITS], payload[tx.MSG_BITS:]
    crc_calc = np.array(tx.crc16_ccitt(msg), dtype=np.uint8)
    crc_ok = bool(np.array_equal(crc_rx, crc_calc))
    text = np.packbits(msg).tobytes().decode("ascii", "replace")
    return text, crc_ok, payload

def decode(iq, fs, t_sym, spacing, r_bw, freq_offset=0.0, f_dopp=0.0, f_rate=0.0, t0=0.0, verbose=True):
    iq = doppler_correct(iq, fs, f_dopp, f_rate, t0)
    nsps = int(round(t_sym * fs))
    symbols, metrics = [], []
    for m in range(N_SYM):
        seg = iq[m * nsps:(m + 1) * nsps]
        d, met = detect_symbol(seg, fs, spacing, r_bw, freq_offset)
        symbols.append(d); metrics.append(met)
    text, crc_ok, payload = symbols_to_message(symbols)
    if verbose:
        print("detected d_m :", symbols)
        print("confidence   :", [round(x, 1) for x in metrics])
        print("message      :", repr(text))
        print("CRC          :", "OK" if crc_ok else "FAIL")
    return dict(symbols=symbols, metrics=metrics, text=text, crc_ok=crc_ok)

# ---- self-test: TX -> simulated echo -> RX ----------------------------------
# the self-tests exercise the DSP at a concrete (freq, target): Venus @ 1299.5 MHz.
_P = eve_params.compute(1299.5e6, "venus", M)     # spacing/R_bw for the tests

def _clean_test(fs=48000.0, t_sym=20.0, f_dopp=3.1, f_rate=0.02, message="ORI EVE TST"):
    """No noise: validates the decode CHAIN (framing, FFT, bins, de-map, BCH, CRC),
    including Doppler shift+rate correction. Must be perfect. offset=0 keeps tones
    below Nyquist at fs=48000."""
    payload = tx.build_payload(message); cw, symbols = tx.encode_symbols(payload)
    iq, _ = tx.synthesize(symbols, fs, t_sym, _P.tone_spacing_hz, 0.0, amplitude=1.0)
    t = np.arange(iq.size)/fs
    echo = iq * np.exp(1j*2*np.pi*(f_dopp*t + 0.5*f_rate*t*t))
    r = decode(echo, fs, t_sym, _P.tone_spacing_hz, _P.R_bw_hz, 0.0, f_dopp, f_rate, verbose=False)
    ok = (r["symbols"] == symbols) and r["crc_ok"] and (r["text"].rstrip(chr(0)) == message)
    print("CLEAN CHAIN  : recovered=%s  CRC=%s  msg=%r" % (ok, r["crc_ok"], r["text"]))
    return ok

def _noise_test(cn0_db, fs=48000.0, t_sym=164.794, f_dopp=3.1, f_rate=0.02,
                message="ORI EVE TST", seed=3):
    """Full symbols at the real channel, processed one symbol at a time.
    Doppler applied and corrected per symbol. Uses Venus @ 1299.5 MHz spacing/R_bw."""
    rng = np.random.default_rng(seed)
    payload = tx.build_payload(message); cw, symbols = tx.encode_symbols(payload)
    nsps = int(round(t_sym*fs)); cn0 = 10**(cn0_db/10.0); sigma2 = (1.0/cn0)*fs
    n = np.arange(nsps); t = n/fs
    dop = np.exp(1j*2*np.pi*(f_dopp*t + 0.5*f_rate*t*t))
    dhats = []
    for d in symbols:
        s = np.exp(1j*2*np.pi*(d*_P.tone_spacing_hz)*n/fs)
        echo = s*dop + np.sqrt(sigma2/2)*(rng.standard_normal(nsps)+1j*rng.standard_normal(nsps))
        seg = doppler_correct(echo.astype(np.complex64), fs, f_dopp, f_rate)
        dh, _ = detect_symbol(seg, fs, _P.tone_spacing_hz, _P.R_bw_hz, 0.0); dhats.append(dh)
    text, crc_ok, _ = symbols_to_message(dhats)
    nerr = sum(a != b for a, b in zip(dhats, symbols))
    ok = (dhats == symbols) and crc_ok
    print("C/N0=%+5.2f  : %d/%d symbols correct  CRC=%s  msg=%r  RECOVERED=%s"
          % (cn0_db, N_SYM-nerr, N_SYM, crc_ok, text, ok))
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("meta", nargs="?"); ap.add_argument("--selftest", action="store_true")
    # CAUSAL RX: the waveform params come from the AGREED (freq, target), computed here --
    # not from the transmitter's file. The recording supplies only capture facts (fs).
    ap.add_argument("--rf", type=float, default=1299.5e6, help="agreed RF center freq (Hz)")
    ap.add_argument("--target", default="venus", help="agreed target body (venus|moon)")
    ap.add_argument("--M", type=int, default=M)
    ap.add_argument("--tsym", type=float, default=None, help="agreed symbol seconds (from the schedule)")
    ap.add_argument("--f-dopp", type=float, default=0.0); ap.add_argument("--f-rate", type=float, default=0.0)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--freq-offset", type=float, default=None, help="agreed comb offset (Hz); default from eve_params")
    args = ap.parse_args()
    if args.selftest or not args.meta:
        print("=== decode chain (no noise, with Doppler) ===")
        _clean_test()
        print("\n=== full-length symbols at the channel we have (Doppler + AWGN) ===")
        _noise_test(0.645); _noise_test(0.0)
    else:
        params = eve_params.compute(args.rf, args.target, args.M)     # RX computes its own params
        meta = json.load(open(args.meta)); fs = meta["global"]["core:sample_rate"]
        # t_sym and the comb offset are agreed operational settings the operator supplies.
        # Fall back to a design block ONLY if this recording carries one (e.g. a loopback
        # capture we wrote ourselves); a real echo recording will not, and --tsym is required.
        dd = meta["global"].get("ori:design", {})
        t_sym = args.tsym if args.tsym is not None else dd.get("t_sym_s")
        if t_sym is None:
            sys.exit("supply --tsym (symbol seconds from the agreed schedule)")
        foff = (args.freq_offset if args.freq_offset is not None
                else dd.get("freq_offset_hz", params.freq_offset_hz))
        raw = np.fromfile(args.meta.replace(".sigmf-meta", ".sigmf-data"), dtype=np.float32)
        iq = raw[0::2] + 1j*raw[1::2]
        print("decoding as %s @ %.4f MHz: R_bw %.3f Hz, spacing %.3f Hz, t_sym %.3f s, offset %.0f Hz"
              % (args.target, args.rf/1e6, params.R_bw_hz, params.tone_spacing_hz, t_sym, foff))
        decode(iq, fs, t_sym, params.tone_spacing_hz, params.R_bw_hz, foff,
               args.f_dopp, args.f_rate, args.t0)
