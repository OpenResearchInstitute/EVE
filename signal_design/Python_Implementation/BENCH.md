# EVE bench loopback -- test TX + RX + decode in the remote lab

`ON A MAC (Apple Silicon)? See **MAC_STATION.md** first for UHD install, the B210
USB-morph quirk, and why we drive the radio via UHD's CLI binaries (no Python bindings).`

This is an attempt to write down bench test results. MAC_STATION.md has the steps that
worked on that particular computer, for the Hello Giggy EME station build. 

One command exercises the whole chain and draws a spectrogram (a software spectrum
analyzer). Works with NO hardware (software channel) or with a real B210 over a cable.
ASCII only. This goal has been achieved and the implementation can be improved from here.

## VERIFIED BASELINE -- B210 cable loopback (known-good reference)

First full hardware pass. Keep these numbers and compare against them when something
later misbehaves.

This is the "it works on my computer" section. 

Setup: MacBook Air (Apple Silicon), UHD 4.11.0.0 (conda env 'eve'), B210 serial
309AF9C over USB 3. Wiring: TX/RX --> 50 dB fixed attenuator --> RX2. No SA (teed
port available). Command:

    python3 eve_b210.py loopback --nsym 11 --tsym 8 --rf 1296e6 --tx-gain 30 --rx-gain 50

Result (GOOD):
    captured 22075000 samples
    peak |rx| = 0.005            (plenty of headroom; could raise gains toward ~0.1-0.3)
    burst starts at sample 0     (cable latency negligible -- expected)
    detected d_m == sent [1269,585,516,1366,1106,84,1333,1065,1789,3717,736]
    confidence ~ 4.6 million, uniform across all 11 symbols
    message 'ORI EVE TST'  CRC OK  PASS
    spectrogram: 11 tones stepping cleanly in the 25-48 kHz comb (eve_b210_cap_spec.png)
At tx-gain 30 / rx-gain 30 the same test also PASSED with peak |rx| = 0.001 and
confidence ~235k -- lower level, still decoded (cable SNR to spare).

Doppler check (correction path):
    add  --f-dopp 3.1 --f-rate 0.02   -> should still PASS. eve_b210.py injects the
    Doppler on the TX baseband and the decoder removes it (mirrors the sim).
    KNOWN TRAP: an older eve_b210.py removed Doppler in the decoder but did NOT inject
    it on hardware TX (the cable adds none), so it de-rotated a signal that was never
    rotated -> every symbol came out exactly -1 (a ~half-tone-spacing offset), CRC FAIL,
    with confidences climbing over the record (the rate term accumulating). If you ever
    see a uniform -1 (or +1) across all symbols, suspect a frequency offset of one tone
    spacing, not noise. Fixed in current eve_b210.py.

What a healthy run shows vs a sick one:
    healthy -> detected == sent, confidence high AND uniform, CRC OK.
    level   -> peak |rx| well under 1.0 (raise/lower --rx-gain to place it).
    freq    -> all symbols off by the same +/-N -> frequency offset (Doppler/ref), not noise.
    noise   -> symbols scattered randomly, confidence low/uneven -> real SNR problem.

## 0. Restore + deps (if the box is fresh)
    pip install numpy galois sigmf matplotlib
    # needs eve_tx_sigmf.py, eve_rx.py, eve_twt.py in the same folder.

## 1. Software loopback (no radio) -- run this first
Validates waveform + decoder + Doppler removal + (optionally) the TWT, right here.
    python3 eve_loopback.py --sim --nsym 11 --tsym 8 --cn0 45 -o eve_loop
    # -> sent d_m, decoded d_m, message, CRC, PASS/FAIL
    # -> eve_loop.sigmf-data/.sigmf-meta  (the capture)
    # -> eve_loop_spec.png  (11 tones stepping through the band = your SA view)
Useful switches:
    --twt              pass through the saturated TWT model (should still PASS)
    --cn0 0            lower C/N0 toward the real channel (use SHORT symbols on the bench;
                       for true C/N0=0 sensitivity use full symbols via eve_rx --selftest,
                       which streams symbol-by-symbol and won't blow up memory)
    --f-dopp / --f-rate   inject a known Doppler; eve_rx removes it
    --nsym 11          full frame is needed for a CRC-OK PASS (a partial frame fails CRC)

Sensitivity (the "does it close at 0 dB-Hz" test) lives in the other tools, which are
memory-safe for full 165 s symbols:
    python3 eve_rx.py --selftest        # TX -> echo(Doppler+AWGN@0 dB-Hz) -> decode
    python3 eve_link_check.py           # the FER-vs-C/N0 table

## 2. Real B210 loopback (one radio, a cable, an attenuator)
NEVER wire TX to RX bare. Use a fixed, power-rated attenuator, and tee to the SA with a
directional coupler:

    B210 TX/A  --> directional coupler --(coupled, ~-30 dB)--> SPECTRUM ANALYZER
                        | (through)
                        v
                 fixed attenuator  (40-60 dB, rated for the power)
                        v
    B210 RX/A  <--------+

  - Attenuation: aim for RX at roughly -30 to -20 dBFS (watch the ADC; back off gain if
    it clips). Start with more attenuation and reduce.
  - SA: the coupled port shows the live TX comb (the same tones as eve_loop_spec.png).
    Confirm you see one clean tone at a time stepping in frequency, and that harmonics
    are where you expect.
  - Run it:
        python3 eve_loopback.py --hardware --nsym 11 --tsym 8 \
            --rf 1296e6 --tx-gain 50 --rx-gain 30 --f-dopp 3.1 -o eve_loop_hw
    Doppler is injected in software on TX and removed in the decoder, so you get a
    realistic, repeatable offset without moving an LO.
  - Output: eve_loop_hw.sigmf-* + eve_loop_hw_spec.png + a decode with CRC.

  *** The --hardware path is a UHD TEMPLATE not yet run on a B210. *** Verify sample rate,
  gains, and (if you gate) the GPIO/PTT line on a scope at low level before trusting it.
  Start with lots of attenuation and low tx-gain.

## 3. What "good" looks like (alt text style)
  - Spectrogram: distinct horizontal tone segments, one per symbol, stepping in frequency
    (d * 5.74 Hz + offset). No smearing within a symbol.
  - Decode: detected d_m == sent d_m, confidence >> 2 (clean loopback gives thousands),
    message correct, CRC OK, PASS.
  - With --twt: identical result (constant envelope; see TWT_GUIDE.md / fig_twt.png).

## 4. Turning the bench test into an on-air test
Same decoder, same SigMF. Replace the software/cable channel with the antenna + the
monostatic gating (eve_tx_gated.py + eve_tx_rx_b210.py) for DSES, and feed eve_rx the
astropy Doppler for the real path. The loopback is the honest dress rehearsal.
