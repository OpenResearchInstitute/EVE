#!/usr/bin/env python3
"""
eve_loopback.py -- bench loopback harness: TX the EVE waveform, capture it, decode it,
and draw a spectrogram (a software 'spectrum analyzer') so you can see it.

THREE MODES
  --sim         pure software (no radio): TX -> channel(attenuation, Doppler, AWGN) -> RX.
                Validates the whole harness + decoder here in the lab with no hardware.
  --hardware    real B210: USRP Sink (TX) --cable+ATTENUATOR--> USRP Source (RX), captured
                and decoded. (UHD template -- verify on your unit; see BENCH WIRING below.)
  (default is --sim)

Always writes:  <out>.sigmf-data/.sigmf-meta  (the capture)  and  <out>_spec.png  (view).
Then decodes with eve_rx and prints the recovered message + CRC.

BENCH WIRING for --hardware (never wire TX to RX bare):
    B210 TX/A  --> directional coupler --(-30 dB coupled)--> SPECTRUM ANALYZER
                        |
                     (through)
                        v
                 fixed attenuator (40-60 dB, power-rated)
                        v
    B210 RX/A  <--------+
  Set attenuation so RX input stays well under the ADC clip (aim ~ -30 to -20 dBFS) and
  the SA sees a safe level. Inject a known Doppler in software (--f-dopp/--f-rate) and let
  eve_rx remove it. Scope the PTT/GPIO line here before it ever drives a real sequencer.
"""
import argparse, json, numpy as np
import eve_tx_sigmf as tx
import eve_rx as rx
try:
    import eve_twt as twt
except Exception:
    twt = None

def build_tx(message, fs, t_sym, freq_offset, n_sym=None):
    payload = tx.build_payload(message)
    cw, symbols = tx.encode_symbols(payload)
    if n_sym:
        symbols = symbols[:n_sym]
    if len(symbols)*t_sym*fs > 60e6:
        raise SystemExit("Too many samples for the bench harness (%.0fM). Use short --tsym "
                         "for loopback; test sensitivity with: python3 eve_rx.py --selftest"
                         % (len(symbols)*t_sym*fs/1e6))
    iq, nsps = tx.synthesize(symbols, fs, t_sym, freq_offset, amplitude=0.8)
    return symbols, iq.astype(np.complex64), nsps

def sim_channel(iq, fs, cn0_db, f_dopp, f_rate, use_twt=False, rng=None):
    rng = rng or np.random.default_rng(0)
    x = iq.copy()
    if use_twt and twt is not None:
        x = twt.saleh(x / (np.abs(x).max() + 1e-9) * twt.R_SAT)   # drive to saturation
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))                  # normalize power
    t = np.arange(x.size) / fs
    x = x * np.exp(1j * 2 * np.pi * (f_dopp * t + 0.5 * f_rate * t * t))   # Doppler
    p = np.mean(np.abs(x) ** 2)
    cn0 = 10 ** (cn0_db / 10.0)
    sigma2 = (p / cn0) * fs                                       # noise for target C/N0
    x = x + np.sqrt(sigma2 / 2) * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return x.astype(np.complex64)

def hardware_loopback(iq, fs, rf_hz, tx_gain, rx_gain, freq_offset, f_dopp, f_rate):
    """Concurrent TX+RX on one B210 over a cable+attenuator. UHD template -- verify."""
    import uhd, threading
    # inject Doppler in software on the TX baseband (repeatable); RX removes it later
    t = np.arange(iq.size) / fs
    txbb = (iq * np.exp(1j * 2 * np.pi * (f_dopp * t + 0.5 * f_rate * t * t))).astype(np.complex64)
    usrp = uhd.usrp.MultiUSRP("")
    usrp.set_tx_rate(fs); usrp.set_rx_rate(fs)
    usrp.set_tx_freq(uhd.types.TuneRequest(rf_hz - freq_offset))
    usrp.set_rx_freq(uhd.types.TuneRequest(rf_hz - freq_offset))
    usrp.set_tx_gain(tx_gain); usrp.set_rx_gain(rx_gain)
    n = txbb.size + int(0.2 * fs)
    rxbuf = np.zeros(n, dtype=np.complex64)
    rxs = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "fc32"))
    txs = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "fc32"))
    t0 = usrp.get_time_now().get_real_secs() + 2.0
    def do_rx():
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        cmd.num_samps = n; cmd.stream_now = False; cmd.time_spec = uhd.types.TimeSpec(t0)
        rxs.issue_stream_cmd(cmd); md = uhd.types.RXMetadata(); got = 0
        while got < n:
            got += rxs.recv(rxbuf[got:], md)
    def do_tx():
        md = uhd.types.TXMetadata(); md.has_time_spec = True
        md.time_spec = uhd.types.TimeSpec(t0 + 0.05)
        md.start_of_burst = True; md.end_of_burst = True
        txs.send(txbb, md)
    r = threading.Thread(target=do_rx); r.start(); do_tx(); r.join()
    return rxbuf

def spectrogram(iq, fs, path, title, cmap="magma", dyn_range=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "figure.facecolor": "#0a0e0a",
        "axes.facecolor": "#0f1610", "axes.edgecolor": "#1fbf0a", "axes.labelcolor": "#c8ffc8",
        "axes.titlecolor": "#39ff14", "xtick.color": "#5f8f66", "ytick.color": "#5f8f66",
        "savefig.facecolor": "#0a0e0a", "figure.dpi": 130})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    nfft = 1 << int(np.log2(max(256, fs / tx.R_BW)))
    Pxx, f, t, im = ax.specgram(iq, NFFT=nfft, Fs=fs, noverlap=nfft // 2,
                                cmap=cmap, scale="dB")
    floor = 10*np.log10(np.median(Pxx) + 1e-30)   # noise floor (dB)
    peak  = 10*np.log10(Pxx.max() + 1e-30)         # brightest tone (dB)
    lo = floor + 3.0 if dyn_range is None else peak - dyn_range
    im.set_clim(lo, peak)                          # noise -> dark, tones -> bright
    ax.set_xlabel("time (s)"); ax.set_ylabel("baseband freq (Hz)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(path); print("wrote", path)

def write_sigmf(iq, fs, rf_hz, freq_offset, t_sym, out):
    inter = np.empty(iq.size * 2, dtype=np.float32); inter[0::2] = iq.real; inter[1::2] = iq.imag
    inter.tofile(out + ".sigmf-data")
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": float(fs),
                       "core:version": "1.0.0", "core:description": "EVE loopback capture",
                       "ori:design": {"t_sym_s": t_sym, "freq_offset_hz": freq_offset}},
            "captures": [{"core:sample_start": 0, "core:frequency": float(rf_hz)}], "annotations": []}
    json.dump(meta, open(out + ".sigmf-meta", "w"), indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", action="store_true", help="use a real B210 (else --sim)")
    ap.add_argument("--sim", action="store_true", help="explicit software mode (default)")
    ap.add_argument("-o", "--out", default="eve_loop")
    ap.add_argument("--message", default="ORI EVE TST")
    ap.add_argument("--nsym", type=int, default=4, help="symbols to send (bench: a few)")
    ap.add_argument("--tsym", type=float, default=8.0, help="symbol seconds (bench: short)")
    ap.add_argument("--cn0", type=float, default=20.0, help="sim C/N0 dB-Hz (cable is strong)")
    ap.add_argument("--twt", action="store_true", help="sim: pass through the saturated TWT model")
    ap.add_argument("--f-dopp", type=float, default=3.1)
    ap.add_argument("--f-rate", type=float, default=0.02)
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--freq-offset", type=float, default=None)
    ap.add_argument("--rf", type=float, default=1296e6)
    ap.add_argument("--tx-gain", type=float, default=50); ap.add_argument("--rx-gain", type=float, default=30)
    ap.add_argument("--colormap", default="magma", help="spectrogram colormap (magma/inferno/viridis/turbo)")
    ap.add_argument("--dyn-range", type=float, default=None, help="dB below peak to display (default: auto from noise floor)")
    a = ap.parse_args()

    # sim wants offset=0 and fs that fits the comb; hardware wants 250k + 25k off DC
    fs = a.fs or (250000.0 if a.hardware else 48000.0)
    foff = a.freq_offset if a.freq_offset is not None else (25000.0 if a.hardware else 0.0)

    symbols, iq, nsps = build_tx(a.message, fs, a.tsym, foff, a.nsym)
    print("sent d_m     :", symbols)

    if a.hardware:
        print("HARDWARE loopback via B210 (verify wiring + attenuator!)")
        rxiq = hardware_loopback(iq, fs, a.rf, a.tx_gain, a.rx_gain, foff, a.f_dopp, a.f_rate)
    else:
        print("SIM loopback: C/N0=%.1f dB-Hz, Doppler %.2f Hz + %.3f Hz/s%s"
              % (a.cn0, a.f_dopp, a.f_rate, ", saturated TWT" if a.twt else ""))
        rxiq = sim_channel(iq, fs, a.cn0, a.f_dopp, a.f_rate, use_twt=a.twt)

    write_sigmf(rxiq, fs, a.rf, foff, a.tsym, a.out)
    spectrogram(rxiq, fs, a.out + "_spec.png",
                "EVE loopback capture (%s)" % ("hardware" if a.hardware else "sim"),
                cmap=a.colormap, dyn_range=a.dyn_range)
    print("\n--- decode ---")
    r = rx.decode(rxiq, fs, a.tsym, foff, a.f_dopp, a.f_rate)
    nerr = sum(x != y for x, y in zip(r["symbols"][:len(symbols)], symbols))
    print("PASS" if (nerr == 0 and r["crc_ok"]) else "symbols wrong: %d" % nerr)

if __name__ == "__main__":
    main()
