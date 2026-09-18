#!/usr/bin/env python3
"""
eve_b210.py -- drive an Ettus B210 for EVE from the command line, using the UHD Python
API (tested target: UHD 4.11 via conda-forge on macOS/Apple Silicon).

Run inside the conda env that has uhd:   conda activate eve

SUBCOMMANDS
  probe                 confirm the B210 (serial, clock) -- no RF
  txonly                synthesize the comb and TRANSMIT ONLY (watch it on a spectrum
                        analyzer; RX is never touched -- zero risk). START HERE.
  loopback              bench cable loopback: TX and RX concurrently (round trip ~ 0),
                        capture, then decode. TX -> ATTENUATOR(40-60 dB) -> RX. Never bare.
  run --schedule S.json burst / wait(RTT) / capture per an eve_gated_*_schedule.json
                        (EME ~2.56 s, EVE ~4.5 min). Coarse timing; captures a generous
                        window and the decoder locks onto the burst by energy.

SAFETY: for loopback/run, TX out must reach RX only through a fixed, power-rated
40-60 dB attenuator (plus your coupler tee to the SA). Start with low --tx-gain and
confirm the level on the SA / rx rssi before nudging up. Aim RX ~ -25 dBFS.
"""
import argparse, json, sys, time, threading
import numpy as np
import eve_tx_sigmf as tx
import eve_rx as rx

def find_burst_start(iq, rate, guard_ms=20.0):
    """Coarse sync: return the sample index where the burst begins, by energy.
    Robust to TX/RX start latency in a hardware capture (burst is not at sample 0)."""
    if iq.size == 0:
        return 0
    win = max(1, int(rate * 0.005))                 # 5 ms smoothing
    p = np.abs(iq) ** 2
    csum = np.cumsum(np.insert(p, 0, 0.0))
    sm = (csum[win:] - csum[:-win]) / win           # moving-average power
    ref = max(1, int(rate * 0.010))                 # first 10 ms ~ lead-in noise floor
    floor = np.median(sm[:ref]); peak = np.percentile(sm, 90)
    if peak < 4 * floor:                            # burst already at start (no lead-in)
        return 0
    thr = floor + 0.5 * (peak - floor)
    idx = int(np.argmax(sm > thr))
    return max(0, idx - int(rate * guard_ms / 1000.0))

def _uhd():
    try:
        import uhd
        return uhd
    except Exception as e:
        sys.exit("Could not import uhd. Are you in the conda env? (conda activate eve)\n  %s" % e)

# ----- PTT via B210 FP0 GPIO_0 (drives a MOSFET/opto that pulls the SEQ 4 / 23 G4 PTT to
# GND). 3.3V CMOS, <=5 mA -- never drive the sequencer directly. Manual GPIO (not ATR);
# host-timed, which is fine because the SEQ 4 guard is 100..150 ms and ours is >=200 ms.
PTT_BANK = "FP0"
PTT_MASK = 0x01          # GPIO_0

def ptt_setup(usrp):
    usrp.set_gpio_attr(PTT_BANK, "CTRL", 0x00, PTT_MASK)     # 0 = manual GPIO (not ATR)
    usrp.set_gpio_attr(PTT_BANK, "DDR",  PTT_MASK, PTT_MASK) # 1 = output
    usrp.set_gpio_attr(PTT_BANK, "OUT",  0x00, PTT_MASK)     # start LOW = RX (safe)

def ptt_set(usrp, on):
    usrp.set_gpio_attr(PTT_BANK, "OUT", PTT_MASK if on else 0x00, PTT_MASK)

# ----- shared helpers --------------------------------------------------------
def make_usrp(uhd, rate, rf, tx_gain=None, rx_gain=None, tx_ant="TX/RX", rx_ant="RX2",
              freq_offset=0.0):
    u = uhd.usrp.MultiUSRP("")
    u.set_tx_rate(rate); u.set_rx_rate(rate)
    # tune the LO 'freq_offset' low so the one-sided baseband comb lands on rf
    u.set_tx_freq(uhd.types.TuneRequest(rf - freq_offset), 0)
    u.set_rx_freq(uhd.types.TuneRequest(rf - freq_offset), 0)
    if tx_gain is not None: u.set_tx_gain(tx_gain, 0)
    if rx_gain is not None: u.set_rx_gain(rx_gain, 0)
    u.set_tx_antenna(tx_ant, 0); u.set_rx_antenna(rx_ant, 0)
    return u

def apply_doppler(iq, rate, f_dopp, f_rate):
    if f_dopp == 0.0 and f_rate == 0.0:
        return iq
    t = np.arange(iq.size) / rate
    return (iq * np.exp(1j * 2 * np.pi * (f_dopp * t + 0.5 * f_rate * t * t))).astype(np.complex64)

def synth(message, rate, tsym, nsym, freq_offset, amp=0.7):
    payload = tx.build_payload(message)
    _, symbols = tx.encode_symbols(payload)
    symbols = symbols[:nsym]
    iq, nsps = tx.synthesize(symbols, rate, tsym, freq_offset, amplitude=amp)
    return symbols, iq.astype(np.complex64), nsps

def stream_tx(uhd, u, samples, start_time=None):
    st = u.get_tx_stream(uhd.usrp.StreamArgs("fc32", "fc32"))
    md = uhd.types.TXMetadata()
    md.start_of_burst = True; md.end_of_burst = False
    if start_time is not None:
        md.has_time_spec = True; md.time_spec = uhd.types.TimeSpec(start_time)
    spp = st.get_max_num_samps()
    i = 0
    while i < samples.size:
        n = min(spp, samples.size - i)
        md.end_of_burst = (i + n >= samples.size)
        st.send(samples[i:i+n], md)
        md.start_of_burst = False; md.has_time_spec = False
        i += n
    # NB: the last chunk above carries end_of_burst=True, which closes the burst.
    # Do NOT send a zero-length array to flush -- UHD rejects its shape.

def capture_rx(uhd, u, n, start_time=None):
    st = u.get_rx_stream(uhd.usrp.StreamArgs("fc32", "fc32"))
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = n
    cmd.stream_now = (start_time is None)
    if start_time is not None:
        cmd.time_spec = uhd.types.TimeSpec(start_time)
    st.issue_stream_cmd(cmd)
    buf = np.zeros(n, dtype=np.complex64); md = uhd.types.RXMetadata(); got = 0
    recv = np.zeros(st.get_max_num_samps(), dtype=np.complex64)
    while got < n:
        k = st.recv(recv, md, 5.0)
        if k == 0:
            if str(md.error_code) != "ERROR_CODE_NONE": print("  rx:", md.error_code)
            break
        m = min(k, n - got); buf[got:got+m] = recv[:m]; got += m
    return buf[:got]

def write_sigmf(iq, rate, rf, freq_offset, tsym, out):
    inter = np.empty(iq.size*2, dtype=np.float32); inter[0::2]=iq.real; inter[1::2]=iq.imag
    inter.tofile(out + ".sigmf-data")
    meta = {"global": {"core:datatype":"cf32_le","core:sample_rate":float(rate),
                       "core:version":"1.0.0","core:description":"EVE B210 capture",
                       "ori:design":{"t_sym_s":tsym,"freq_offset_hz":freq_offset}},
            "captures":[{"core:sample_start":0,"core:frequency":float(rf)}],"annotations":[]}
    json.dump(meta, open(out + ".sigmf-meta","w"), indent=2)

# ----- subcommands -----------------------------------------------------------
def cmd_probe(a):
    uhd = _uhd(); u = uhd.usrp.MultiUSRP("")
    print("B210 OK  |  master clock %.3f MHz" % (u.get_master_clock_rate()/1e6))
    print("TX antennas:", u.get_tx_antennas(0), " RX antennas:", u.get_rx_antennas(0))
    print("TX gain range:", u.get_tx_gain_range(0).start(), "..", u.get_tx_gain_range(0).stop())

def cmd_txonly(a):
    uhd = _uhd()
    syms, iq, _ = synth(a.message, a.rate, a.tsym, a.nsym, a.freq_offset)
    print("TX-ONLY (RX untouched). symbols:", syms)
    print("watch the SA at %.6f MHz; comb spans +%.0f..%.0f Hz above tune"
          % (a.rf/1e6, a.freq_offset, a.freq_offset + (tx.M-1)*tx.SPACING))
    #u = make_usrp(uhd, a.rate, a.rf, tx_gain=a.tx_gain, tx_ant=a.tx_ant, freq_offset=a.freq_offset)
    u = make_usrp(uhd, a.rate, a.rf, tx_gain=a.tx_gain, tx_ant=a.tx_ant)
    reps = max(1, a.repeat)
    for r in range(reps):
        print("  transmitting pass %d/%d (%.1f s)" % (r+1, reps, iq.size/a.rate))
        stream_tx(uhd, u, iq)
    print("done.")

def cmd_loopback(a):
    uhd = _uhd()
    syms, iq, nsps = synth(a.message, a.rate, a.tsym, a.nsym, a.freq_offset)
    print("LOOPBACK (TX->attenuator->RX). symbols:", syms)
    iq = apply_doppler(iq, a.rate, a.f_dopp, a.f_rate)   # inject on TX; decoder removes it
    u = make_usrp(uhd, a.rate, a.rf, tx_gain=a.tx_gain, rx_gain=a.rx_gain,
                  tx_ant=a.tx_ant, rx_ant=a.rx_ant, freq_offset=a.freq_offset)
    t0 = u.get_time_now().get_real_secs() + 1.0
    n_rx = iq.size + int(0.3*a.rate)
    cap = {}
    th = threading.Thread(target=lambda: cap.__setitem__("rx", capture_rx(uhd, u, n_rx, t0)))
    th.start(); stream_tx(uhd, u, iq, t0 + 0.02); th.join()
    rxiq = cap.get("rx", np.zeros(0, np.complex64))
    pk = np.abs(rxiq).max() if rxiq.size else 0
    print("captured %d samples (peak |rx| = %.3f%s)"
          % (rxiq.size, pk, "  CLIPPING -- lower --rx-gain!" if pk > 0.98 else ""))
    start = find_burst_start(rxiq, a.rate)
    print("burst starts at sample %d (%.3f s into capture)" % (start, start/a.rate))
    _finish(rxiq[start:], a, syms)

def cmd_run(a):
    uhd = _uhd()
    sch = json.load(open(a.schedule))
    rate = a.rate; foff = sch.get("freq_offset_hz", a.freq_offset); W = sch["symbol_s"]
    rtt = sch["rtt_s"]; syms = sch["symbols_d"]
    print("SCHEDULE run: RTT %.3f s, %d symbols of %.3f s (%s)"
          % (rtt, len(syms), W, a.schedule))
    u = make_usrp(uhd, rate, a.rf, tx_gain=a.tx_gain, rx_gain=a.rx_gain,
                  tx_ant=a.tx_ant, rx_ant=a.rx_ant, freq_offset=foff)
    GUARD = 0.25            # >= SEQ 4 delay (100..150 ms) + relay travel + margin
    if a.ptt:
        ptt_setup(u)
        print("PTT ENABLED (FP0 GPIO_0). *** Verify with 'ptt-test' + scope FIRST. ***")
    caps = []
    try:
        for k, d in enumerate(syms):
            burst, _ = tx.synthesize([d], rate, W, foff, amplitude=0.7)
            t0 = u.get_time_now().get_real_secs() + 1.0
            if a.ptt:                                   # key TX; SEQ 4: relay->TX, then PA
                ptt_set(u, True); time.sleep(GUARD)     # let relay settle + PA come up
            stream_tx(uhd, u, burst.astype(np.complex64), t0 + 0.05)
            if a.ptt:                                   # RF done -> unkey; SEQ 4: PA off, relay->RX
                time.sleep(W + 0.1); ptt_set(u, False); time.sleep(GUARD)
            # RX the echo: from just before it returns to just after it ends
            n_rx = int((W + 1.0) * rate)
            rxseg = capture_rx(uhd, u, n_rx, t0 + rtt - 0.5)
            caps.append(rxseg)
            print("  sym %2d d=%4d: TX %.1f s, waited RTT %.1f s, captured %d samp"
                  % (k, d, W, rtt, rxseg.size))
    finally:
        if a.ptt:
            ptt_set(u, False)                           # ALWAYS leave PTT LOW (RX/safe)
            print("PTT forced LOW (RX) on exit.")
    # each segment is one echo in a (W+1)s window -> sync + detect per segment
    dhat = []
    for seg in caps:
        st = find_burst_start(seg, rate)
        d, _ = rx.detect_symbol(seg[st:st+int(W*rate)], rate, foff)
        dhat.append(d)
    print("detected d_m:", dhat)
    text, crc, _ = rx.symbols_to_message(dhat)
    nerr = sum(x != y for x, y in zip(dhat, syms))
    print("message: %r  CRC: %s  %s" % (text, "OK" if crc else "FAIL",
          "PASS" if (nerr==0 and crc) else "symbols wrong: %d"%nerr))
    if caps:
        try:
            from eve_loopback import spectrogram
            spectrogram(np.concatenate(caps), rate, a.out+"_spec.png", "EVE B210 schedule capture", cmap="magma")
        except Exception as e: print("  (spectrogram skipped: %s)"%e)

def _finish(rxiq, a, syms):
    if rxiq.size == 0:
        print("no samples captured -- check attenuation/wiring; try the SA path."); return
    write_sigmf(rxiq, a.rate, a.rf, a.freq_offset, a.tsym, a.out)
    try:
        from eve_loopback import spectrogram
        spectrogram(rxiq, a.rate, a.out + "_spec.png",
                    "EVE B210 capture", cmap="magma")
    except Exception as e:
        print("  (spectrogram skipped: %s)" % e)
    print("--- decode ---")
    r = rx.decode(rxiq, a.rate, a.tsym, a.freq_offset, a.f_dopp, a.f_rate)
    nerr = sum(x != y for x, y in zip(r["symbols"][:len(syms)], syms))
    print("PASS" if (nerr == 0 and r["crc_ok"]) else "symbols wrong: %d (see spectrogram)" % nerr)

def cmd_ptt_test(a):
    """Toggle FP0 GPIO_0 with NO RF so you can scope the pin, your MOSFET/opto, and the
    SEQ 4 outputs, and confirm logic sense + the LNA-protect ordering at ZERO RF risk.
    This IS the Stage-2 dry run, driven by the real code. Leaves PTT LOW (RX) on exit."""
    uhd = _uhd(); u = uhd.usrp.MultiUSRP("")
    ptt_setup(u)
    print("PTT TEST on %s GPIO_0 (mask 0x01). NO RF is transmitted." % PTT_BANK)
    print("Scope the GPIO pin -> your MOSFET/opto -> PTT line -> SEQ 4 relay/PA outputs.")
    print("HIGH = TX (PTT pulled to GND by your interface), LOW = RX. Ctrl-C to stop.")
    try:
        while True:
            print("  PTT HIGH  (TX)  -- expect: relay->TX/load, then (after SEQ delay) PA-supply line")
            ptt_set(u, True);  time.sleep(a.ptt_period)
            print("  PTT LOW   (RX)  -- expect: PA-supply off, then relay->RX")
            ptt_set(u, False); time.sleep(a.ptt_period)
    except KeyboardInterrupt:
        ptt_set(u, False)
        print("\nPTT forced LOW (RX). done.")

def main():
    p = argparse.ArgumentParser(description="EVE B210 station (UHD 4.11 Python API).")
    p.add_argument("--rate", type=float, default=250000.0)
    p.add_argument("--rf", type=float, default=1296e6)
    p.add_argument("--freq-offset", type=float, default=25000.0, help="comb offset off DC")
    p.add_argument("--tx-gain", type=float, default=30.0)
    p.add_argument("--rx-gain", type=float, default=30.0)
    p.add_argument("--tx-ant", default="TX/RX")
    p.add_argument("--rx-ant", default="RX2")
    p.add_argument("--message", default="ORI EVE TST")
    p.add_argument("--nsym", type=int, default=11)
    p.add_argument("--tsym", type=float, default=8.0)
    p.add_argument("--repeat", type=int, default=1, help="txonly: repeat passes")
    p.add_argument("--ptt", action="store_true", help="run: key SEQ 4 via FP0 GPIO_0 per burst")
    p.add_argument("--ptt-period", type=float, default=2.0, help="ptt-test: seconds per half-cycle")
    p.add_argument("--f-dopp", type=float, default=0.0)
    p.add_argument("--f-rate", type=float, default=0.0)
    p.add_argument("--schedule", help="run: *_schedule.json")
    p.add_argument("-o", "--out", default="eve_b210_cap")
    p.add_argument("cmd", choices=["probe", "txonly", "loopback", "run", "ptt-test"])
    a = p.parse_args()
    {"probe": cmd_probe, "txonly": cmd_txonly, "loopback": cmd_loopback,
     "run": cmd_run, "ptt-test": cmd_ptt_test}[a.cmd](a)

if __name__ == "__main__":
    main()
