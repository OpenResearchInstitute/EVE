#!/usr/bin/env python3
"""
eve_tx_gated.py -- monostatic (one-dish) gating for DSES: transmit a symbol, stop,
switch the T/R relay to RX, hear the echo, repeat. Produces the timing schedule that
protects the LNA and lets a single station hear its own echo.

WHY: a dish cannot receive while its PA is keyed (the LNA would be destroyed). So each
symbol is a separate TX burst; between bursts the PA is OFF, the sequencer switches the
dish from PA to LNA, and the station records its own echo.

MONOSTATIC TIMING (per symbol k, cycle C = RTT + W + 2*guard):
    PTT on   = k*C - ptt_lead        (sequencer: relay->TX, LNA protected, PA on)
    TX burst = [k*C , k*C + W]        (RF on; raised-cosine ramps so no clicks)
    PTT off  = k*C + W + ptt_lag      (RF already 0; sequencer relay->RX, LNA in)
    echo in  = [k*C + RTT , k*C + RTT + W]
    RX rec   = [k*C + RTT - m , k*C + RTT + W + m]
Guards guarantee RF is zero whenever the relay moves -> the LNA never sees the PA.

For EME (RTT ~2.56 s, short W) this also writes a full-timeline gated SigMF you can
play directly. For Venus (W=164.794 s, ~80 min total) it writes only the schedule +
symbol values; the B210 control script synthesizes each burst just-in-time (a full
zero-filled file would be gigabytes).
"""
import argparse, json, csv, numpy as np
import eve_tx_sigmf as tx
import eve_params            # R_bw / tone spacing from (freq, target)
C_LIGHT = 299792458.0

def build_schedule(rtt, W, nsym, guard=1.0, ptt_lead=0.3, ptt_lag=0.3,
                   rx_margin=0.5, relay_settle=0.05):
    C = rtt + W + 2 * guard
    rows = []; clip = False; unsafe = False
    for k in range(nsym):
        ts = k * C
        tx_stop = ts + W
        ptt_off = tx_stop + ptt_lag            # relay -> RX here; RF already 0
        echo_in = ts + rtt
        if ptt_off <= tx_stop:
            unsafe = True                      # RF still on when relay moves (never happens if ptt_lag>0)
        rx_start = max(echo_in - rx_margin, ptt_off + relay_settle)
        if rx_start > echo_in:
            clip = True                        # gap too small; echo start clipped
        rows.append({"sym": k, "ptt_on": ts - ptt_lead, "tx_start": ts, "tx_stop": tx_stop,
                     "ptt_off": ptt_off, "echo_in": echo_in, "rx_start": rx_start,
                     "rx_stop": echo_in + W + rx_margin})
    return C, rows, clip, unsafe

def ramp(sig, fs, ramp_ms):
    n = int(fs * ramp_ms / 1000.0)
    if n < 1 or 2 * n >= sig.size:
        return sig
    w = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))   # raised cosine
    sig = sig.copy(); sig[:n] *= w; sig[-n:] *= w[::-1]
    return sig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="eve_spiral_gated")
    ap.add_argument("--distance-mkm", type=float, help="one-way distance (million km)")
    ap.add_argument("--rtt", type=float, help="round-trip time (s); overrides distance")
    ap.add_argument("--eme", action="store_true", help="preset Moon RTT=2.563 s")
    ap.add_argument("--tsym", type=float, default=2.0)
    ap.add_argument("--fs", type=float, default=250000.0)
    ap.add_argument("--rf", type=float, default=1296e6, help="RF center freq (Hz)")
    ap.add_argument("--target", default="venus", help="target body (venus|moon) -> R_bw/spacing")
    ap.add_argument("--guard", type=float, default=1.0)
    ap.add_argument("--ramp-ms", type=float, default=50.0)
    ap.add_argument("--freq-offset", type=float, default=25000.0)
    ap.add_argument("--message", default="CQ DE W6ORI")
    ap.add_argument("--write-iq", action="store_true",
                    help="also write a full-timeline gated SigMF (only sane for short/EME runs)")
    args = ap.parse_args()

    P = eve_params.compute(args.rf, args.target, tx.M)   # R_bw / tone spacing for this band/target

    if args.eme:
        rtt = 2.563
    elif args.rtt:
        rtt = args.rtt
    elif args.distance_mkm:
        rtt = 2 * args.distance_mkm * 1e9 / C_LIGHT
    else:
        rtt = 2 * 40.82e9 / C_LIGHT          # Venus best day default
    W = args.tsym
    payload = tx.build_payload(args.message)
    cw, symbols = tx.encode_symbols(payload)
    C, rows, clip, unsafe = build_schedule(rtt, W, tx.N_SYMBOLS, guard=args.guard)
    total = tx.N_SYMBOLS * C

    print("band/target   : %s @ %.4f MHz  ->  R_bw %.3f Hz, spacing %.3f Hz, coh %.3f s"
          % (args.target, args.rf/1e6, P.R_bw_hz, P.tone_spacing_hz, P.coherence_time_s))
    print("RTT           : %.3f s   (%s)" % (rtt, "EME" if args.eme else "one-way %.2f Mkm" % (rtt*C_LIGHT/2/1e9)))
    print("symbol W      : %.3f s   guard: %.2f s   cycle C: %.3f s" % (W, args.guard, C))
    print("symbols d_m   :", symbols)
    print("total time    : %.1f s  (%.1f min)" % (total, total/60))
    print("monostatic OK : W < RTT ? %s  (must be True)" % (W < rtt))
    print("LNA safe      : RF off before relay switch ? " + str(not unsafe))
    if clip:
        print("WARNING       : gap RTT-W too small; echo start clipped by %.3f s. "
              "Use shorter --tsym or go bistatic." % (rows[0]["rx_start"]-rows[0]["echo_in"]))
    else:
        print("TX->echo gap  : %.3f s clean (RF off %.1f s before echo)" % (rtt-W, rows[0]["echo_in"]-rows[0]["tx_stop"]))
    print()
    print("%-4s %9s %9s %9s %9s %9s %9s" % ("sym","ptt_on","tx_start","tx_stop","ptt_off","rx_start","rx_stop"))
    for r in rows:
        print("%-4d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f" %
              (r["sym"],r["ptt_on"],r["tx_start"],r["tx_stop"],r["ptt_off"],r["rx_start"],r["rx_stop"]))

    # schedule.csv + schedule embedded in a small SigMF-meta-style json
    with open(args.out + "_schedule.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    sched = {"rtt_s": rtt, "symbol_s": W, "cycle_s": C, "guard_s": args.guard,
             "fs": args.fs, "freq_offset_hz": args.freq_offset, "symbols_d": symbols,
             "message": args.message, "rows": rows,
             "rf_hz": args.rf, "target": P.target,
             "R_bw_hz": P.R_bw_hz, "tone_spacing_hz": P.tone_spacing_hz,
             "note": "monostatic gating; PA off + relay settled before LNA connects"}
    json.dump(sched, open(args.out + "_schedule.json", "w"), indent=2)
    print("\nwrote", args.out + "_schedule.csv", "and", args.out + "_schedule.json")

    if args.write_iq:
        n_total = int(round(total * args.fs))
        if n_total * 8 > 500e6:
            print("REFUSING --write-iq: full timeline is %.1f GB. Use JIT synthesis "
                  "(the control script) for long runs." % (n_total*8/1e9)); return
        buf = np.zeros(n_total, dtype=np.complex64)
        for r, d in zip(rows, symbols):
            s, _ = tx.synthesize([d], args.fs, W, P.tone_spacing_hz, args.freq_offset, amplitude=0.8)
            s = ramp(s, args.fs, args.ramp_ms).astype(np.complex64)
            i0 = int(round(r["tx_start"] * args.fs))
            buf[i0:i0 + s.size] = s[:max(0, n_total - i0)]
        inter = np.empty(buf.size * 2, dtype=np.float32)
        inter[0::2] = buf.real; inter[1::2] = buf.imag
        inter.tofile(args.out + ".sigmf-data")
        meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": float(args.fs),
                            "core:version": "1.0.0",
                            "core:description": "EVE monostatic GATED TX (Pete's waveform). "
                                                "RF only during tx bursts; gaps are RF-off for T/R + RX.",
                            "ori:schedule": sched},
                "captures": [{"core:sample_start": 0, "core:frequency": float(args.rf)}],
                "annotations": [{"core:sample_start": int(r["tx_start"]*args.fs),
                                 "core:sample_count": int(W*args.fs),
                                 "core:label": "TX burst sym %d d=%d" % (r["sym"], d)}
                                for r, d in zip(rows, symbols)]}
        json.dump(meta, open(args.out + ".sigmf-meta", "w"), indent=2)
        print("wrote full-timeline gated %s.sigmf-data (%.1f MB)" % (args.out, n_total*8/1e6))

if __name__ == "__main__":
    main()
