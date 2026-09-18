# EVE station on a Mac (Apple Silicon)
## A complete setup and operating guide 
## Works on Abraxas3d's computer! 

Everything needed to run the EVE transmit/receive station on one MacBook Air (Apple Silicon),
next to the K3NG rotator. Written to be followed step by step. ASCII only, but apparently
this wasn't enough to fend off suspicions of the entire thing being AI slop. It is not. 

Verified working on my machine: UHD 4.11.0.0 (conda, sadly), B210 serial 309AF9C, USB 3,
internal GPSDO, master clock 16 MHz, TX antenna TX/RX, RX antennas TX/RX + RX2,
TX gain 0..89.75 dB. All of that directly from the normal command line stuff that you are
supposed to do every time you fire one up. TX gain really does almost go up to 90. 

Why doesn't it go up to 90 dB? Glad you asked. Because there's a hardware limitation. 

359 steps times 0.25 dB step size is 89.75 dB
360 steps times 0.25 dB step size is 90.00 dB

This 360th step exceeds the maximum design attenuation threshold of the chip's internal 
resistor network. Why 0.25? This is exactly 2^-2, and is what the internal control
register can handle. OK now that that is out of the way.  

================================================================================
0. DAILY QUICK START (once setup in section 2 is done)
================================================================================
Open a fresh Terminal, then:
    conda activate eve
    cd ~/EVE/signal_design/Python_Implementation
    python3 eve_b210.py probe          # confirm the radio (serial 309AF9C)
Then whichever test you are running (details in section 5):
    python3 eve_b210.py txonly   ...   # TX only, watch the spectrum analyzer
    python3 eve_b210.py loopback ...   # cable loopback, capture + decode
    python3 eve_b210.py run --schedule ...   # EME / EVE burst-wait-capture
If probe cannot find the radio: external power on the B210, unplug/replug USB, retry
(section 3). When done for the day:  conda deactivate

This is all here because it's easy for me to forget how to use tools that I 
don't use every day. It's a very common situation for me to use something frequently
for a relatively short time period, get lots of muscle memory, and "know" how to do it,
then come back after a couple of months and forget the environment, the name of the script,
or a very important setup or argument.

================================================================================
1. WHY THIS WORKS ON A MAC (and why the usual warning does not apply)
================================================================================
- The "use Linux for SDR" warning is about USB THROUGHPUT at tens of MSps. EVE runs at
  ~250 kSps (a 23.5 kHz signal), four orders of magnitude below any bottleneck. Fine.
- The internal GPSDO is a bonus: a disciplined reference on the same box, useful later
  for the real Doppler / coherence work.
- The K3NG rotator is a separate serial/USB device. No conflict. One laptop is fine.

Why is this here? Because Linux is the default for a lot of things. Originally this was all
going to be in GNU Radio, which traditionally really needed to be done in the same Ubuntu 
version that the developers used, in order to not cry or hit a wall or find out that 
your GNU Radio can't run because brew or whatever doesn't install it right. The B210 I'm 
using has a GPSDO and that's documented here. The K3NG rotator controller and Moon Unit
doesn't interfere. Everything works well together (so far)

================================================================================
2. ENVIRONMENT SETUP (do this once)  -- and WHY it is conda, not venv
================================================================================
The station code needs "import uhd" (the UHD Python bindings). Two dead ends we already
hit, so you do not have to wonder:
  - There is NO 'uhd' package on PyPI, so pip/venv CANNOT install it. (Your orbital venv
    is great for everything else; it just cannot reach this one compiled package.)
  - Homebrew's uhd ships the driver + the uhd_* tools but NOT the Python bindings, and
    NOT the tx_samples_from_file / rx_samples_to_file example programs.
The one place with prebuilt, version-matched bindings for Apple Silicon is conda-forge.
So we use a small conda environment named 'eve' JUST for radio work. Your venv is
untouched; you switch with 'conda activate eve' / 'conda deactivate'. One active at a
time in a terminal. Do not stack them. Although I do.

Setup (once):
    brew install miniforge
    conda init bash            # then CLOSE and REOPEN Terminal (or: source ~/.bash_profile)
    conda create -n eve -c conda-forge python=3.14 uhd numpy scipy matplotlib sigmf
    conda activate eve
    pip install galois         # not on conda-forge; pip INTO the eve env is fine
    uhd_images_downloader      # fetch B210 firmware/FPGA for THIS uhd
    python3 -c "import uhd; print(uhd.__version__)"   # expect 4.11.0.0-release (or similar)

Notes:
  - New terminals may show (base). That is conda's default env, not eve. Always
    'conda activate eve' before radio work. Your machine might not have default conda.
  - Do NOT 'conda activate eve' while an orbital venv is active. If you see
    "(eve) (orbital)" in the prompt, open a fresh terminal and just activate eve.
    But, it worked for me when I did accidentally stack them. Just be cautions. 
  - Keep the .py files (eve_b210.py, eve_tx_sigmf.py, eve_rx.py, eve_loopback.py,
    eve_twt.py, eve_link_check.py) together in Python_Implementation and run from there.

================================================================================
3. THE B210 USB "MORPH" QUIRK -- read this before you panic (I panicked)
================================================================================
On power-up the B210 loads firmware/FPGA and RE-ENUMERATES on USB ("morphs"). On older
white B200/B210 this fails ~half the time and UHD then cannot find it. Normal on macOS;
not your setup being broken. It failed for me on an old but not that old B210. 
Fixes, most reliable first:
  1. Use the B210 EXTERNAL POWER SUPPLY (barrel jack), not USB power alone. Biggest help.
  2. If not found: unplug the USB cable, replug the SAME port, retry.
  3. Last resort: the USB reset button (S700) on the board. (didn't work for me). 
Once probe shows the B210, it is stable for the session. Usually. 

================================================================================
4. FIRST CONTACT (no RF)
================================================================================
    conda activate eve
    python3 eve_b210.py probe
Success looks like:
    B210 OK  |  master clock 16.000 MHz
    TX antennas: ['TX/RX']  RX antennas: ['TX/RX', 'RX2']
    TX gain range: 0.0 .. 89.75
(uhd_usrp_probe and uhd_find_devices also still work as raw checks.)

Copied directly from the terminal. This just makes sure that UHD is working and you can
command your B210 in a way that lets you transmit and receive. 

================================================================================
5. RUNNING THE STATION  (eve_b210.py)
================================================================================
General shape:   python3 eve_b210.py <command> [options]
Common options (all have safe defaults):
    --rf 1296e6         RF center (Hz). Use your bench / SA-friendly band.
    --rate 250000       sample rate (Hz). Leave at 250k.
    --freq-offset 25000 comb sits this far above the tune freq, off DC. Leave at 25k.
    --tx-gain 30        TX gain dB (0..89.75). START LOW.
    --rx-gain 30        RX gain dB (0..76).
    --tx-ant TX/RX      TX port (only choice on B210).
    --rx-ant RX2        RX port for loopback (RX2 recommended; TX/RX also valid).
    --nsym 11 --tsym 8  number of symbols and seconds/symbol (bench: 11 x 8 s).
    --message "ORI EVE TST"   up to ~11 characters.
    -o eve_b210_cap     output basename for captures.

--- 5a. TX ONLY into the spectrum analyzer (START HERE; RX never touched) ------
Zero risk to the receiver. Watch the 11 tones step on the SA. I did this it was 
awesome!
    python3 eve_b210.py txonly --nsym 11 --tsym 8 --rf 1296e6 --tx-gain 30 --repeat 5
What good looks like: on the SA, one tone at a time stepping in frequency, spanning
about +25 kHz to +48.5 kHz above the tune frequency (the 25 kHz offset keeps it off DC).
Raise --tx-gain only after you see a clean, unclipped tone. This is Alt Text style 
from the announcement about progress. You can speed up the symbol time from 8 seconds
if you want to see things change faster. 
(If the SA is busy, like it was for me, then you can skip to loopback.
eve_b210.py saves its own spectrogram of the capture, which is your received-signal view.)

The captures were made for slide deck presentations but are actually super useful for
showing the theory of what is going on here.

--- 5b. CABLE LOOPBACK (bench, ~0 delay): capture + decode -------------------
Wiring:  B210 TX/RX --> [40-60 dB attenuator] --> B210 RX2.  NEVER bare.
If you are doing this in ORI Remote Lab, you have 50 dB attentuation for loopback. 
The attenuation is chosen for you in order to not hurt the B210. 

Optional SA tee off TX via a directional coupler. Upon request, in Remote Labs. 
    python3 eve_b210.py loopback --nsym 11 --tsym 8 --rf 1296e6 --tx-gain 30 --rx-gain 30
It prints: captured N samples (peak |rx| = ...), the burst start, then the decode.
What good looks like:
    peak |rx| = 0.1 to 0.6      (well under 1.0; if > 0.98 it says CLIPPING then lower --rx-gain)
    detected d_m == the sent list, message 'ORI EVE TST', CRC OK, PASS
    a saved eve_b210_cap_spec.png showing 11 tones stepping (your SA-style view)
Tuning the level: peak |rx| too small (< 0.05) -> raise --rx-gain a little. Clipping ->
lower --rx-gain. With 50 dB fixed pad, tx-gain 30 / rx-gain 30 is a fine starting point. 
This is basic advice in general, but there's a compression knee in the B210. We want the cleanest
possible signal for the IF rig since Hello Giggy EME station has a transverter and then a PA. 
We don't want to start out with a crappy signal. 

--- 5c. EME / EVE scheduled run (burst -> wait RTT -> capture) ---------------
Uses a schedule from eve_tx_gated.py. TX a symbol, wait the round-trip, capture the echo.
    # EME (short symbols, ~2.56 s round trip):
    python3 eve_tx_gated.py --eme --tsym 1.0 -o eve_gated_eme
    python3 eve_b210.py run --schedule eve_gated_eme_schedule.json --rf 1296e6 \
        --tx-gain 40 --rx-gain 40
    # EVE (164.794 s symbols, ~4.5 min round trip):
    python3 eve_tx_gated.py --distance-mkm 40.82 --tsym 164.794 -o eve_gated_venus
    python3 eve_b210.py run --schedule eve_gated_venus_schedule.json --rf 2304e6 ...
Each symbol: it transmits, waits the round trip, captures a generous window, and syncs
onto the echo by energy. Prints detected d_m, message, CRC. NOTE: on-air monostatic
(one dish TX and RX) still needs the T/R sequencer for LNA safety -- see TWT_GUIDE.md /
BENCH.md; the schedule's gaps are when the relay switches. The cable bench has no relay.

Hello Giggy is bistatic by design, so we don't have to have this. Bigger stations probably
just want to generate the SigMF file and then handle splitting it up themselves, but in order
to be taken seriously at all, we went the extra mile here and made the script manage it.
This lets us demonstrate what can happen. Either full round trip or one symbol at a time or 
some other cadence can be chosen. It's parameterized. 

Hello Giggy has a sequencer for the transmit and a coax switch for the receive.

================================================================================
6. THE THREE TIMING REGIMES (same wrapper, round trip time parameterized!)
================================================================================
- BENCH (today): round trip ~ 0. loopback runs TX and RX concurrently over the cable.
- EME: round trip ~ 2.56 s. run --schedule with a short --tsym.
- EVE: round trip ~ 4.5 min. run --schedule with --tsym 164.794.
Timing is coarse (software-scheduled); this is ok because we capture generous windows and the
decoder locks onto the burst by energy. Sample-locked on-air T/R gating is a hardware
sequencer's job (GPIO), not software sleeps. 

================================================================================
7. SAFETY (cable loopback)
================================================================================
- NEVER wire B210 TX -> RX bare. Fixed, power-rated 40-60 dB attenuator in that path.
  This is a Remote Labs rule. 
  (You have 50 dB in remote lab and this is good. Even at max TX gain that keeps RX 
  far from any damage level. You have to manage this on your end with your own radio.)
- Watch peak |rx|: keep it well under 1.0; the script warns at > 0.98. Lower --rx-gain,
  not the pad, to fix clipping. We learned this the hard way several times. 
- Tee to the SA through a directional coupler (~ -30 dB coupled), not a bare tee.
- Start --tx-gain low. Confirm a clean tone (SA or the saved spectrogram) before nudging.

================================================================================
8. TROUBLESHOOTING (symptom -> fix)
================================================================================
- "No module named 'uhd'"             -> not in the eve env. conda activate eve.
- "(eve) (orbital)" in the prompt      -> stacked envs. Fresh terminal, conda activate eve.
- probe finds no device                -> B210 morph. External power, replug USB, retry (sec 3).
- AttributeError on some uhd.* name    -> a UHD API name differs on your build. Copy the FULL
                                          traceback and send it; it is a one-line fix.
- peak |rx| = 0 or "no samples"        -> RX not seeing TX. Check cable/attenuator path;
                                          confirm TX with txonly into the SA.
- peak |rx| ~ 1.0 / "CLIPPING"         -> lower --rx-gain (try 20, then 10).
- decode wrong but tones show in the
  saved eve_b210_cap_spec.png          -> level/sync, not signal. Adjust gains; make sure
                                          --tsym / --nsym match what you transmitted.
- matplotlib DeprecationWarning         -> harmless on Python 3.14. Ignore.
- people think your README is Claude    -> no idea what to do about that.

================================================================================
9. WHAT IS SUPERSEDED
================================================================================
Use eve_b210.py for all B210 work. The earlier hardware TEMPLATES -- eve_tx_rx_b210.py
and eve_loopback.py --hardware -- were written before a radio was in hand (raw UHD calls,
unverified). Keep them for reference only; do not run them on the B210 any more. 

eve_loopback.py --sim (pure software, no radio) is still useful for testing the DSP and
the decoder with no hardware, and eve_rx.py --selftest / eve_link_check.py remain the
memory-safe way to test full-length-symbol sensitivity at C/N0 = 0 dB-Hz.

Anyone updating the repo really should get safe scripts only... so...

================================================================================
10. TO DO
================================================================================

Review all of the actual code to make sure it doesn't do anything dumb. We've caught
a bunch so far but there probably are more really dumb bugs in there. The one that 
stood out to me the most was where the frequency offset was subtracted out after
it was put in. Moving the center frequency worked, but the offset (to get away from
DC spike territory or any LO problems) just didn't. It got missed until the week of
September 15th 2026. You're never too late with double-checking the code carefully, 
and going ahead and testing everything methodically pays off. 
