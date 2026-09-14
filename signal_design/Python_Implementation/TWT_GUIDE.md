# Using a TWT for EVE -- the complete guide (replaces DSES's 8 x 250 W harness)

Purpose: run Pete's EVE waveform through ONE traveling-wave tube instead of eight
250 W solid-state PAs, an 8-way combiner, and a phase-matched harness. This document
explains how, why it is safe for the signal, and how to not get hurt or kill the
tube. Read all of it before keying. ASCII only.

================================================================================
0. TL;DR
================================================================================
- Our signal is CONSTANT ENVELOPE (one pure tone at a time). A saturated TWT amplifies
  it with ZERO distortion that matters: one gain, one fixed phase. Proven in simulation
  (eve_twt.py, eve_link_check.py): at C/N0 = 0 dB-Hz the message decodes 11/11 with CRC
  good through a saturated tube, even with phase noise 10x worse than a real tube. A
  realistic tube costs 0.001 dB. See fig_twt.png and fig_twt_compliance.png.
- Therefore: run the TWT SATURATED, full power, NO back-off, NO predistortion, NO
  combiner, NO phasing. One tube, one coax, done.
- The three things you must still do right: (1) gate the DRIVE not the beam, (2) protect
  the LNA with the T/R sequencer exactly as before, (3) protect the tube with an
  isolator + good load, and respect the high voltage.

================================================================================
1. WHY IT WORKS (do not skip -- this is the whole justification)
================================================================================
A TWT does not modulate. Your B210/exciter makes the modulated tone at low level; the
TWT is only the final power amplifier. The waveform is M-ary orthogonal FSK: within a
164.794 s symbol it is a single CW tone; between symbols it hops. So |signal| is flat.

A nonlinear amplifier hurts a signal two ways:
  AM/AM (gain compression): distorts AMPLITUDE variations. We have none.
    We sit at ONE point on the curve (see fig_twt.png, top-left). Run at saturation.
  AM/PM (amplitude-to-phase): turns AMPLITUDE variation into PHASE variation. Constant
    amplitude means ONE constant phase offset (top-right). A fixed phase is invisible to
    non-coherent per-symbol detection. Does not apply.
  Intermodulation: needs two or more tones present at once. We transmit ONE tone at a
    time, so the tube makes only HARMONICS (2f, 3f at 4.6, 6.9 GHz) which a filter after
    the tube removes. There is NO in-band IMD (fig_twt.png, bottom row: our one tone is
    clean; two simultaneous tones would splatter IMD across the band).
The only residual effect is the tube's own PHASE NOISE, and it is negligible: over one
0.348 s coherent frame a realistic tube loses 0.001 dB; you would need absurd phase
noise (>1 rad rms per frame) to lose even 1 dB.

This is exactly why deep-space and satcom links use constant-envelope modulation through
saturated TWTs: maximum power, maximum efficiency, no back-off. Pete's waveform was, in
effect, built for a tube.

Contrast with what breaks a tube's signal: QAM/OFDM/multi-carrier or any scheme that
combines tones (high peak-to-average power). Those force big back-off or predistortion.
Do NOT try to transmit multiple EVE tones simultaneously to "go faster" -- that
reintroduces AM and IMD and throws away the entire advantage.

================================================================================
2. CHOOSE THE TUBE  (strongly prefer a TWTA, not a bare tube)
================================================================================
- Get a TWTA (TWT Amplifier): a tube WITH its integrated high-voltage power supply,
  protection, interlocks, and cooling in one chassis. A bare TWT + home-built HV supply
  is a lethal science project according to people way smarter than me. 
  A TWTA is an appliance with an RF connector. For an all-volunteer group this is not 
  optional. Use a TWTA.
- Band: S-band, must cover 2304 MHz (or 2320/2400 if you move). Surplus S-band satcom
  and radar TWTAs are common. Confirm the passband includes your frequency with margin.
- Power: size for your EIRP target. A single 200-700 W S-band TWTA typically replaces
  the 8 x 250 W array once you remove combiner + harness losses (which were eating
  1-2 dB anyway). More saturated power straight into the antenna moves you UP the
  compliance curve.
- Gain: TWTAs run 30-50 dB gain, so a few mW to tens of mW of drive saturates them.
  Your B210 output (~ -10 dBm) usually needs a small driver amp/attenuator to hit the
  correct drive; see section 4.
- Duty and pulse: confirm the tube is rated for your duty. Our monostatic Venus schedule
  is ~50% duty in ~165 s keydowns; make sure the TWTA is CW-rated or high-duty rated,
  not a short-pulse radar tube that cannot sit keyed for 165 s.

================================================================================
3. THE DRIVE CHAIN
================================================================================
  B210 TX out --> attenuator/driver amp --> [bandpass 2304] --> TWTA in
  TWTA out --> [isolator/circulator + load] --> [harmonic filter] --> T/R relay --> feed
  feed --> T/R relay --> LNA --> ... --> B210 RX in
Notes:
- Driver/attenuator sets the TWTA input to the saturation level (section 4). Put a
  clean bandpass before the tube so you do not amplify B210 wideband noise/spurs.
- Isolator or circulator + load on the TWTA OUTPUT is mandatory (section 8): it protects
  the tube from reflected power if the feed/relay is mismatched or arcs.
- Harmonic filter (low-pass or band-pass) after the tube removes the 2f/3f harmonics a
  saturated tube produces.

================================================================================
4. SET THE OPERATING POINT (drive to saturation) -- exact procedure
================================================================================
Do this on a POWER METER into a DUMMY LOAD at reduced beam if possible, never into the
antenna, never near people.
  1. Terminate the TWTA output in a rated dummy load through a calibrated coupler/power
     meter. Confirm cooling and interlocks are satisfied.
  2. Drive a single CW tone from the B210 (e.g. one EVE tone, or CW at mid-band).
  3. Starting well below rated drive, raise input in 1 dB steps and record output power.
  4. Plot Pout vs Pin. It rises ~linearly, then the slope rolls off. SATURATION is where
     +1 dB of input gives < ~0.2 dB more output (the knee). 
  5. Set the input 0 to 1 dB INTO saturation (past the knee). That is your drive level.
     Because the signal is constant-envelope, sitting in compression is fine and gives
     max power. Do NOT back off "to be linear" -- that just wastes the tube.
  6. Lock the attenuator/driver at that setting. Record the number.
This drive level is what the B210 must produce for every burst. In software the burst
amplitude is fixed (0.8 full-scale in eve_tx_sigmf); calibrate the analog attenuation so
that full-scale lands at your saturation drive.

================================================================================
5. GATING: GATE THE DRIVE, NOT THE BEAM
================================================================================
Keep the TWTA beam ON continuously for the whole pass. Key transmit by turning the RF
DRIVE on and off, not by switching the high voltage.
  - Hard-keying the beam (HV) 11 times per pass stresses the cathode, causes big
    transients, and shortens tube life. Do not do it.
  - Instead, the B210 emits RF only during each TX burst (the schedule in
    eve_tx_gated.py) with raised-cosine ramps (--ramp-ms) at the edges so there are no
    key clicks and the RF is fully zero before/after each burst.
  - Between bursts the drive is zero -> the tube outputs (near) nothing -> safe to switch
    T/R. A tube with no drive still emits some broadband noise, so the T/R relay (not the
    absence of drive) is what actually protects the LNA (section 6).
The same PTT line that runs the T/R sequencer should also be wired so that RF drive is
physically inhibited unless the system is in the TX state (belt and suspenders): if the
sequencer is not in TX, the driver amp is muted.

================================================================================
6. T/R AND LNA PROTECTION WITH A TWT (unchanged ordering, higher stakes)
================================================================================
Same invariant as the solid-state case, enforced by the hardware sequencer (Kuhne SEQ 4
or equivalent): RF power is ZERO whenever the relay is in or moving toward RX.
  TX: PTT on -> relay DISH->PA(TWT), LNA isolated/grounded -> settle -> enable RF drive.
  RX: disable RF drive -> wait for tube output to decay (guard) -> PTT off -> relay
      DISH->LNA -> settle -> start RX.
Extra care with a tube:
  - A TWT can emit broadband noise even undriven. The T/R relay must give real isolation
    (>60-80 dB) between the tube output and the LNA. Verify isolation on the bench.
  - Use a relay/coax rated for your peak power with margin. A relay that fails closed
    while the tube is driven destroys the LNA instantly.
  - Consider a fast PIN limiter in front of the LNA as a last-ditch backstop.
The gating schedule already guarantees the tube is unkeyed ~107 s before the Venus echo
arrives (fig_gating.png), so there is generous time for switching.

================================================================================
7. PROTECT THE TUBE (this is how tubes die)
================================================================================
- ISOLATOR/CIRCULATOR + LOAD on the output, always. A TWT driven into an open, short,
  or arcing mismatch can be destroyed by reflected power in milliseconds. The isolator
  sends reflected power to a load instead of back into the helix.
- VSWR/reflected-power interlock: trip the drive if reflected power exceeds a threshold.
- Never key into an unknown/disconnected feed. Confirm the load/antenna match first.
- Respect the tube's helix-current and body-current limits and let the TWTA's own
  protection do its job; do not defeat interlocks.

================================================================================
8. HIGH-VOLTAGE AND RF SAFETY -- this can kill you
================================================================================
TWTAs run cathode voltages of several kV to >10 kV, with stored energy in HV capacitors
that remains lethal after power-off.
  - Never open a TWTA with HV present or caps undischarged. Follow the maker's lockout /
    discharge procedure. Use a grounding stick. Assume caps are charged.
  - Never work alone on HV. Know where the disconnect is.
  - Do not defeat interlocks or covers. They are why you go home.
  - X-rays: tubes above ~15 kV can emit X-rays; keep shielding intact.
  - RF exposure: hundreds of watts at 2.3 GHz will burn tissue and damage eyes. Never
    look into or put a hand near an open waveguide/feed when keyed. Enforce a keep-out
    zone around the feed during transmit. Observe RF exposure limits at the dish.
  - Cooling: TWTAs need their airflow/liquid cooling. An overheated tube fails fast.
    Confirm cooling interlocks before enabling HV.
If nobody on the team is experienced with kV supplies, bring in someone who is before
first power-up. This is not the place to learn by doing.

================================================================================
9. BENCH ACCEPTANCE TEST (before the sky)
================================================================================
  1. Cooling + interlocks verified; output into rated dummy load via coupler/meter.
  2. Warm up the tube per the maker's schedule (cathode warm-up before HV/drive).
  3. Pout-vs-Pin sweep -> set saturation drive (section 4). Record.
  4. Spectral purity: on a spectrum analyzer, drive one EVE tone at saturation; confirm
     a single clean tone in-band and that harmonics are killed by the output filter.
  5. Phase noise (optional but nice): measure the tube's added phase noise, or take it
     from the data sheet. Anything a normal comms TWTA does is fine (we tolerate >1 rad
     rms per 0.348 s frame; a real tube is ~0.02 rad). If the HV supply has 100/120 Hz
     ripple, look for ripple sidebands and confirm they are small.
  6. T/R isolation: measure tube-output-to-LNA isolation through the relay in RX state.
  7. Gating dry run: run  python3 eve_tx_rx_b210.py <schedule.json> --dry-run  and scope
     the PTT/drive/relay lines against the printed times. Confirm RF is zero whenever the
     relay moves.
  8. Only then, low duty, into the antenna with the keep-out zone enforced.

================================================================================
10. INTEGRATION WITH THE EVE SOFTWARE
================================================================================
Nothing about the TWT changes the waveform or the schedule.
  - Build the schedule:  python3 eve_tx_gated.py --distance-mkm 40.82 --tsym 164.794 ...
  - Run the station:     python3 eve_tx_rx_b210.py eve_gated_venus_schedule.json --rf 2304e6
    The controller's PTT/GPIO line drives your sequencer; wire it so PTT also gates the
    RF drive to the TWTA (section 5). Set B210 tx_gain + the analog attenuator so
    full-scale = saturation drive (section 4).
  - Decode the capture: python3 eve_rx.py eve_rx_capture.sigmf-meta --f-dopp .. --f-rate ..
Amplitude in the SigMF/burst is fixed and constant-envelope; do not add amplitude
shaping beyond the edge ramps.

================================================================================
11. OPERATING CHECKLIST (per pass)
================================================================================
  [ ] Cooling on, interlocks green, tube warmed up.
  [ ] Output isolator + load/antenna confirmed matched; reflected-power interlock armed.
  [ ] Harmonic filter + bandpass in line.
  [ ] Drive level set to saturation (recorded value); driver muted unless in TX.
  [ ] T/R sequencer verified: RF zero whenever relay moves; LNA isolation measured.
  [ ] Keep-out zone around the feed enforced; nobody near waveguide when keyed.
  [ ] Schedule loaded; dry-run scoped; Doppler (astropy) ready for the decode.
  [ ] Beam ON for the pass; key by DRIVE only.

================================================================================
12. WHAT NOT TO DO
================================================================================
  - Do NOT back the tube off "to be linear." Constant envelope -> saturation is correct.
  - Do NOT transmit multiple tones at once (no combining/OFDM) -> AM + IMD -> ruined.
  - Do NOT hard-key the beam/HV to gate. Gate the drive.
  - Do NOT key into an unmatched/unknown load or without the isolator.
  - Do NOT rely on "no drive" to protect the LNA. The relay protects the LNA.
  - Do NOT open the chassis or defeat interlocks. The HV will kill you.

Bottom line: one S-band TWTA, run saturated, with an isolator, a harmonic filter, the
same T/R sequencer, and drive-gating from the existing schedule -- replaces the entire
8 x 250 W combiner/phasing harness, delivers more clean power, and the link math is
identical (fig_twt_compliance.png). It closes at C/N0 = 0 dB-Hz.

Comments, corrections, critique WELCOME AND ENCOURAGED. 
