#!/usr/bin/env python3
"""
eve_tx_sigmf.py -- generate a SigMF recording implementing Pete Wyckoff's (KA3WCA)
"Venus Bounce Transmitter Spiral #2" waveform, for transmit through a USRP B210
(via GNU Radio or uhd_siggen-style playback). ASCII only.

DESIGN (verified from Pete's June 2026 block diagram):
  106 payload bits (message + CRC) -> BCH(127,106) -> map to 11 x M-ary(4096)
  symbols -> NCO -> complex baseband.

  NCO (Pete's equation):
      s_n = exp( i * 2*pi * d_{floor(n/Tsym)} * (2*R_bw) * n / Fs )
  i.e. symbol value d in [0,4095] is an orthogonal tone at
      f_d = d * (2 * R_bw)                       [one-sided baseband]

  R_bw   = 2.87 Hz     (FFT bin width = Doppler-spread forecast)
  spacing= 2*R_bw = 5.74 Hz   (one guard bin between tones)
  M      = 4096        (12 bits / symbol)
  N_sym  = 11 symbols
  T_sym  = 164.794 s   (RX Monte Carlo result for C/N0 = 0 dB)
  BW     = M * spacing = 4096 * 5.74 = 23511 Hz  (Pete's "~22 kHz")
  T_tx   = 11 * T_sym  = 1812.734 s  (~30 min 12.7 s)

RX side (documented, not implemented here): shared H-maser via White Rabbit;
astropy removes bulk Doppler shift + rate; FFT at R_bw, combine ~440 non-coherently
per symbol.

NOTE ON DC: Pete's mapping puts symbol value d=0 at DC (0 Hz baseband). On a real
B210, LO leakage sits at DC. For on-air transmit use --freq-offset to move the whole
comb up off DC (the receiver just subtracts the same offset). Default 0.0 is faithful
to Pete's equation; a few kHz offset is recommended for real hardware.
"""
import argparse, json, struct, datetime, hashlib
import numpy as np
import galois

# ---- fixed design constants (Pete) ------------------------------------------
M          = 4096
BITS_SYM   = 12                 # log2(M)
N_SYMBOLS  = 11
R_BW       = 2.87               # Hz
SPACING    = 2 * R_BW           # Hz, tone spacing
T_SYM_FULL = 164.794            # s
BCH_N, BCH_K = 127, 106
PAYLOAD_BITS = BCH_K            # 106 = message + CRC
CRC_BITS   = 16
MSG_BITS   = PAYLOAD_BITS - CRC_BITS   # 90

_bch = galois.BCH(BCH_N, BCH_K)

# ---- payload: message (90b) + CRC-16-CCITT (16b) = 106 bits -----------------
def crc16_ccitt(bits):
    """CRC-16-CCITT (poly 0x1021, init 0xFFFF, no reflect) over bits packed MSB-first
    into bytes (zero-padded to a byte boundary). Returns 16 bits (MSB first)."""
    b = np.packbits(np.asarray(bits, dtype=np.uint8))
    crc = 0xFFFF
    for byte in b:
        crc ^= (int(byte) << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return [(crc >> (15 - i)) & 1 for i in range(16)]

def text_to_msg_bits(text):
    raw = np.unpackbits(np.frombuffer(text.encode("ascii", "replace"), dtype=np.uint8))
    bits = np.zeros(MSG_BITS, dtype=np.uint8)
    n = min(len(raw), MSG_BITS)
    bits[:n] = raw[:n]
    return bits

def build_payload(text):
    msg = text_to_msg_bits(text)             # 90 bits
    crc = np.array(crc16_ccitt(msg), dtype=np.uint8)   # 16 bits
    return np.concatenate([msg, crc])        # 106 bits

# ---- BCH + M-ary mapping -----------------------------------------------------
def encode_symbols(payload106):
    cw = np.array(_bch.encode(galois.GF2(payload106)), dtype=np.uint8)   # 127 bits
    padded = np.zeros(N_SYMBOLS * BITS_SYM, dtype=np.uint8)              # 132 bits
    padded[:BCH_N] = cw
    syms = []
    for m in range(N_SYMBOLS):
        chunk = padded[m*BITS_SYM:(m+1)*BITS_SYM]
        d = 0
        for b in chunk:                      # MSB first
            d = (d << 1) | int(b)
        syms.append(d)
    return cw, syms

# ---- NCO synthesis (Pete's equation) ----------------------------------------
def synthesize(symbols, fs, t_sym, freq_offset=0.0, amplitude=0.8):
    nsps = int(round(t_sym * fs))            # samples per symbol
    total = nsps * len(symbols)
    n = np.arange(total, dtype=np.float64)
    d = np.repeat(np.array(symbols, dtype=np.float64), nsps)[:total]
    f = d * SPACING + freq_offset            # tone frequency per sample
    phase = 2.0 * np.pi * f * n / fs         # Pete: 2*pi*d*(2*R_bw)*n/Fs
    s = amplitude * np.exp(1j * phase)
    return s.astype(np.complex64), nsps

# ---- SigMF output ------------------------------------------------------------
def write_sigmf(iq, fs, base, symbols, nsps, t_sym, rf_freq, freq_offset, text):
    data_path = base + ".sigmf-data"
    meta_path = base + ".sigmf-meta"
    inter = np.empty(iq.size * 2, dtype=np.float32)
    inter[0::2] = iq.real; inter[1::2] = iq.imag
    inter.tofile(data_path)
    sha = hashlib.sha512(open(data_path, "rb").read()).hexdigest()

    ann = []
    for m, d in enumerate(symbols):
        f0 = d * SPACING + freq_offset
        ann.append({
            "core:sample_start": m * nsps,
            "core:sample_count": nsps,
            "core:freq_lower_edge": rf_freq + f0 - R_BW,
            "core:freq_upper_edge": rf_freq + f0 + R_BW,
            "core:label": "sym %d  d=%d  f=%.2f Hz" % (m, d, f0),
        })
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(fs),
            "core:version": "1.0.0",
            "core:sha512": sha,
            "core:author": "Design: Pete Wyckoff KA3WCA. Generated by ORI EVE tools.",
            "core:description": "EVE M-ary(4096) orthogonal FSK, BCH(127,106), "
                                "11 symbols. Venus Bounce Transmitter Spiral #2.",
            "core:recorder": "eve_tx_sigmf.py",
            "ori:design": {
                "M": M, "bits_per_symbol": BITS_SYM, "n_symbols": N_SYMBOLS,
                "R_bw_hz": R_BW, "tone_spacing_hz": SPACING,
                "t_sym_s": t_sym, "t_sym_design_s": T_SYM_FULL,
                "bch": "BCH(127,106) t=3 narrow-sense systematic; genpoly "
                       "x^21+x^18+x^17+x^15+x^14+x^12+x^11+x^8+x^7+x^6+x^5+x+1",
                "crc": "CRC-16-CCITT poly 0x1021 init 0xFFFF over 90 msg bits",
                "payload_text": text,
                "symbols_d": symbols,
                "freq_offset_hz": freq_offset,
                "nco": "s_n = exp(i*2*pi*d*(2*R_bw)*n/Fs)  (Pete's equation)",
            },
        },
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": float(rf_freq),
            "core:datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }],
        "annotations": ann,
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    return data_path, meta_path, sha

def main():
    ap = argparse.ArgumentParser(description="Generate Pete's EVE SigMF waveform.")
    ap.add_argument("-o", "--out", default="eve_spiral", help="output basename")
    ap.add_argument("--fs", type=float, default=250000.0, help="sample rate (Hz)")
    ap.add_argument("--tsym", type=float, default=None,
                    help="symbol seconds (default: full design %.3f)" % T_SYM_FULL)
    ap.add_argument("--smoke", action="store_true",
                    help="short smoke-test file (tsym=2.0 s) to validate the chain")
    ap.add_argument("--rf", type=float, default=1296e6, help="RF center freq for metadata (Hz)")
    ap.add_argument("--freq-offset", type=float, default=0.0,
                    help="shift comb up off DC for real TX (Hz); RX subtracts it")
    ap.add_argument("--amplitude", type=float, default=0.8)
    ap.add_argument("--message", default="ORI EVE DE PI9CAM",
                    help="payload text (packed to 90 bits)")
    args = ap.parse_args()

    t_sym = 2.0 if args.smoke else (args.tsym if args.tsym else T_SYM_FULL)

    payload = build_payload(args.message)
    cw, symbols = encode_symbols(payload)
    # self-check: BCH round-trips
    dec = np.array(_bch.decode(galois.GF2(cw)), dtype=np.uint8)
    assert np.array_equal(dec, payload), "BCH self-check failed"

    iq, nsps = synthesize(symbols, args.fs, t_sym, args.freq_offset, args.amplitude)
    dpath, mpath, sha = write_sigmf(iq, args.fs, args.out, symbols, nsps, t_sym,
                                    args.rf, args.freq_offset, args.message)

    dur = iq.size / args.fs
    mb = iq.size * 8 / 1e6
    print("message      :", args.message)
    print("payload(106) :", "".join(map(str, payload)))
    print("symbols d_m  :", symbols)
    print("tone freqs Hz:", [round(d*SPACING + args.freq_offset, 2) for d in symbols])
    print("t_sym        : %.3f s   samples/sym: %d" % (t_sym, nsps))
    print("duration     : %.2f s   samples: %d   size: %.1f MB" % (dur, iq.size, mb))
    print("sample_rate  : %.0f Hz   comb: %.0f..%.0f Hz (offset %.0f)"
          % (args.fs, args.freq_offset, (M-1)*SPACING + args.freq_offset, args.freq_offset))
    print("wrote        :", dpath, mpath)

if __name__ == "__main__":
    main()
