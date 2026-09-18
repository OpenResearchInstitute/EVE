# EVE project file manifest

What every file is, what it does, and how they depend on each other. Probably out of date.

================================================================================
0. import dependencies
================================================================================
The scripts import each other. Keep these together in one folder and run from there:
  eve_tx_sigmf.py   <- imported by almost everything. THE core module. Needs: numpy, galois.
  eve_rx.py         <- imports eve_tx_sigmf. Needs: numpy, galois.
  eve_twt.py        <- standalone (its self-test imports eve_tx_sigmf). Needs: numpy.
  eve_link_check.py <- standalone. Needs: numpy.
  eve_tx_gated.py   <- imports eve_tx_sigmf. Needs: numpy.
  eve_loopback.py   <- imports eve_tx_sigmf, eve_rx, eve_twt(optional). Needs: numpy, matplotlib.
  eve_b210.py       <- imports eve_tx_sigmf, eve_rx, uhd; uses eve_loopback.spectrogram.
If you see "ModuleNotFoundError: No module named 'eve_tx_sigmf'", you are missing a .py
file or running from the wrong folder. Ask me how I know. 

================================================================================
1. CORE DSP CHAIN seriously keep all of these
================================================================================
eve_tx_sigmf.py   TX generator. message -> CRC-16 -> BCH(127,106) -> 11 x M-ary(4096)
                  -> NCO (Pete's spiral) -> SigMF. The heart of everything. CLI + importable.
                  This is supposed to be useful to people with bigger dishes. 
eve_rx.py         Receiver/decoder. Doppler-correct -> frame at 1/R_bw -> FFT -> non-coherent
                  combine -> pick tone -> BCH decode -> CRC. Run: eve_rx.py --selftest.
eve_twt.py        TWT (Saleh AM/AM+AM/PM) + phase-noise model, and the constant-envelope
                  proof (one tone survives saturation; two tones make IMD). Run standalone.
                  Written to prove out that a TWTA can be used with this waveform. 
eve_link_check.py Monte-Carlo link compliance: FER vs C/N0, with a coherence-loss knob for
                  Doppler-spread and TWT phase noise. Run standalone for the table.
                  This is supposed to mirror Pete Wyckoff's MATLAB Monte-Carlo work. 

================================================================================
2. HARDWARE / STATION
================================================================================
eve_b210.py       *** CURRENT B210 station wrapper (UHD 4.11 Python API). ***
                  Subcommands: probe | txonly | loopback | run | ptt-test.
                  Bench loopback verified; run/ptt (on-air) not yet hardware-tested.
eve_loopback.py   Bench loopback harness. --sim = pure software (no radio), the DSP/decoder
                  test + spectrogram. (Its --hardware path is SUPERSEDED by eve_b210.py.)
eve_tx_rx_b210.py SUPERSEDED. Early raw-UHD controller written before a radio was in hand
                  (unverified). Keep for reference only; do NOT run on the B210.

================================================================================
3. GATING / SCHEDULING (monostatic DSES / EME / EVE)
================================================================================
eve_tx_gated.py   Builds the monostatic TX/RX/PTT schedule from the round-trip time
                  (checks W<RTT, LNA-safety ordering). Writes *_schedule.json/.csv used by
                  eve_b210.py run. Presets: --eme, --distance-mkm, --rtt.
                  If you have a way to do this better or more simply then please write us. 
                  This is the best we could come up with, using the simplest interface
                  possible to the B210. We'd like to improve it if there is a better way.

================================================================================
4. LINK-BUDGET / ANALYSIS SCRIPTS (from the DEF CON deck work)
================================================================================
eve_model.py             Reconstructed ORI link-budget engine (validated vs the notebook).
eve_conjunction_data.py  Verified dynamic-albedo / October-window scan data.
compute_sites.py         DSES / Dwingeloo / Effelsberg C/N0 comparisons.
eve_style.py             matplotlib terminal-green-phosphor style used by the deck figures.
(These may not be in Python_Implementation. They lived with the slide-deck work, and 
that is probably in a nearby folder.)

================================================================================
5. DOCUMENTATION (read these; they are the project's memory)
================================================================================
MANIFEST.md       This file. Who is who at the zoo.
README.md         Overview, core workflow, design provenance (waveform parameters, BCH/CRC).
MAC_STATION.md    macOS/conda setup, eve_b210.py usage, troubleshooting. START HERE on the Mac.
                  Use this for Hello Giggy station operation. 
BENCH.md          Bench loopback guide + the VERIFIED baseline (known-good B210 numbers).
                  Record of the stuff we learned, and was supposed to be general guide, 
                  but each OS and each machine is going to be a different adventure. 
EME_BRINGUP.md    Staged EME station bring-up: feed + SEQ 4 + LNA + PA + B210, LNA-safety
                  (septum + MSP2T-18-12+ + measured isolation), GPIO PTT. 13 cm / 2304 MHz.
                  This is station bringup for Hello Giggy EME station but is a very typical
                  procedure for this type of station. Your station may be different!
TWT_GUIDE.md      TWT operating manual (why constant-envelope + saturation; safety).
                  Done on request by DSES because they had challenges with the phased amplifiers.

================================================================================
6. BUILD ARTIFACTS -- regenerate, do NOT commit (per the "SigMF is a build product" rule)
================================================================================
*.sigmf-data, *.sigmf-meta      generated by eve_tx_sigmf.py / eve_b210.py / eve_loopback.py
*_schedule.json, *_schedule.csv generated by eve_tx_gated.py
*_spec.png (eve_loop_spec, eve_b210_cap_spec, ...)  capture spectrograms from a run
__pycache__/                    Python bytecode cache
Regenerate examples:
  python3 eve_tx_sigmf.py --smoke -o eve_spiral_smoke --freq-offset 25000
  python3 eve_tx_gated.py --eme --tsym 1.0 -o eve_gated_eme
  python3 eve_loopback.py --sim --nsym 11 --tsym 8 --cn0 45

This needs to all go into cleaner .gitignores. That's a TO-DO. 

================================================================================
7. FIGURES (for the talk / docs) -- regenerable from the scripts
================================================================================
fig_sigmf.png            the 11 M-ary tones stepping (from eve_tx_sigmf smoke + a plot)
fig_link_check.png       FER vs C/N0 compliance curve (eve_link_check)
fig_twt.png              TWT AM/AM, AM/PM, constellation, spectrum (eve_twt)
fig_twt_compliance.png   saturated TWT == linear PA (eve_link_check + eve_twt)
fig_gating.png           monostatic TX/echo/RX/PTT timeline (eve_tx_gated schedule)
fig_loopback_spec.png    example received-signal waterfall
(Deck-only figures: fig_rescue.png, fig_validation.png, etc. live with the slide work.)

A picture is worth a thousand words and these are not bad. They've been through a couple
rounds of revisions because contrast and all that sort of stuff can make a figure like this
essentially invisible. So, you'll see adaptive calculations for the figures that might be 
a bit messy looking. But, it works to make reliably good visuals. 

================================================================================
8. MINIMUM SETS (what you need in order to do a particular task)
================================================================================
Decode / DSP self-test (no radio):  eve_tx_sigmf.py + eve_rx.py  (+ galois, numpy)
Software loopback + spectrogram:    + eve_loopback.py + eve_twt.py (+ matplotlib)
Link compliance table/plot:         eve_link_check.py
Run the B210:                       eve_b210.py + eve_tx_sigmf.py + eve_rx.py + eve_loopback.py
                                    (in the conda 'eve' env with uhd)
Build a monostatic schedule:        eve_tx_gated.py + eve_tx_sigmf.py

More expected here. Hello Giggy can't do monostatic, but should be able to key-down 
for 30 minutes. We'll see very soon. 
