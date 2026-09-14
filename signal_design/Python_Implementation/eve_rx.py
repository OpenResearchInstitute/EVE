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
import argparse, json, numpy as np
import eve_tx_sigmf as tx   # reuse the exact TX constants + coder for consistency
import galois

R_BW = tx.R_BW; M = tx.M; SPACING = tx.SPACING
N_SYM = tx.N_SYMBOLS; BITS_SYM = tx.BITS_SYM
_bch = tx._bch

def doppler_correct(iq, fs, f_dopp=0.0, f_rate=0.0, t0=0.0):
    """De-rotate a constant Doppler shift and linear Doppler rate."""
    t = t0 + np.arange(iq.size) / fs
    return iq * np.exp(-1j * 2 * np.pi * (f_dopp * t + 0.5 * f_rate * t * t))

def detect_symbol(seg, fs, freq_offset=0.0):
    """Non-coherent M-ary detection over one symbol. Returns (d_hat, metric)."""
    nfft = int(round(fs / R_BW))                     # samples per coherent frame
    nfr = seg.size // nfft
    if nfr < 1:
        return 0, 0.0
    frames = seg[:nfr * nfft].reshape(nfr, nfft)
    P = (np.abs(np.fft.fft(frames, axis=1)) ** 2).sum(axis=0)   # non-coherent combine
    # candidate tone bins: f_d = d*SPACING + offset  -> bin = round(f/ (fs/nfft))
    binhz = fs / nfft
    d = np.arange(M)
    bins = np.round((d * SPACING + freq_offset) / binhz).astype(int) % nfft
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

def decode(iq, fs, t_sym, freq_offset=0.0, f_dopp=0.0, f_rate=0.0, t0=0.0, verbose=True):
    iq = doppler_correct(iq, fs, f_dopp, f_rate, t0)
    nsps = int(round(t_sym * fs))
    symbols, metrics = [], []
    for m in range(N_SYM):
        seg = iq[m * nsps:(m + 1) * nsps]
        d, met = detect_symbol(seg, fs, freq_offset)
        symbols.append(d); metrics.append(met)
    text, crc_ok, payload = symbols_to_message(symbols)
    if verbose:
        print("detected d_m :", symbols)
        print("confidence   :", [round(x, 1) for x in metrics])
        print("message      :", repr(text))
        print("CRC          :", "OK" if crc_ok else "FAIL")
    return dict(symbols=symbols, metrics=metrics, text=text, crc_ok=crc_ok)

# ---- self-test: TX -> simulated echo -> RX ----------------------------------
def _clean_test(fs=48000.0, t_sym=20.0, f_dopp=3.1, f_rate=0.02, message="ORI EVE TST"):
    """No noise: validates the decode CHAIN (framing, FFT, bins, de-map, BCH, CRC),
    including Doppler shift+rate correction. Must be perfect. offset=0 keeps tones
    below Nyquist at fs=48000."""
    payload = tx.build_payload(message); cw, symbols = tx.encode_symbols(payload)
    iq, _ = tx.synthesize(symbols, fs, t_sym, 0.0, amplitude=1.0)
    t = np.arange(iq.size)/fs
    echo = iq * np.exp(1j*2*np.pi*(f_dopp*t + 0.5*f_rate*t*t))
    r = decode(echo, fs, t_sym, 0.0, f_dopp, f_rate, verbose=False)
    ok = (r["symbols"] == symbols) and r["crc_ok"] and (r["text"].rstrip(chr(0)) == message)
    print("CLEAN CHAIN  : recovered=%s  CRC=%s  msg=%r" % (ok, r["crc_ok"], r["text"]))
    return ok

def _noise_test(cn0_db, fs=48000.0, t_sym=164.794, f_dopp=3.1, f_rate=0.02,
                message="ORI EVE TST", seed=3):
    """Full 473-frame symbols at the real channel, processed one symbol at a time.
    offset=0 (tones 0..23.5 kHz < fs/2). Doppler applied and corrected per symbol."""
    rng = np.random.default_rng(seed)
    payload = tx.build_payload(message); cw, symbols = tx.encode_symbols(payload)
    nsps = int(round(t_sym*fs)); cn0 = 10**(cn0_db/10.0); sigma2 = (1.0/cn0)*fs
    n = np.arange(nsps); t = n/fs
    dop = np.exp(1j*2*np.pi*(f_dopp*t + 0.5*f_rate*t*t))
    dhats = []
    for d in symbols:
        s = np.exp(1j*2*np.pi*(d*SPACING)*n/fs)
        echo = s*dop + np.sqrt(sigma2/2)*(rng.standard_normal(nsps)+1j*rng.standard_normal(nsps))
        seg = doppler_correct(echo.astype(np.complex64), fs, f_dopp, f_rate)
        dh, _ = detect_symbol(seg, fs, 0.0); dhats.append(dh)
    text, crc_ok, _ = symbols_to_message(dhats)
    nerr = sum(a != b for a, b in zip(dhats, symbols))
    ok = (dhats == symbols) and crc_ok
    print("C/N0=%+5.2f  : %d/%d symbols correct  CRC=%s  msg=%r  RECOVERED=%s"
          % (cn0_db, N_SYM-nerr, N_SYM, crc_ok, text, ok))
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("meta", nargs="?"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tsym", type=float, default=None)
    ap.add_argument("--f-dopp", type=float, default=0.0); ap.add_argument("--f-rate", type=float, default=0.0)
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--freq-offset", type=float, default=25000.0)
    args = ap.parse_args()
    if args.selftest or not args.meta:
        print("=== decode chain (no noise, with Doppler) ===")
        _clean_test()
        print("\n=== full-length symbols at the channel we have (Doppler + AWGN) ===")
        _noise_test(0.645); _noise_test(0.0)
    else:
        meta = json.load(open(args.meta)); fs = meta["global"]["core:sample_rate"]
        d = meta["global"].get("ori:design", {})
        raw = np.fromfile(args.meta.replace(".sigmf-meta", ".sigmf-data"), dtype=np.float32)
        iq = raw[0::2] + 1j*raw[1::2]
        decode(iq, fs, d.get("t_sym_s", args.tsym), d.get("freq_offset_hz", args.freq_offset),
               args.f_dopp, args.f_rate, args.t0)
