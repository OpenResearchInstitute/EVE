# ORI EVE transmit waveform -- Pete Wyckoff's "Spiral" as a SigMF file

FILE INDEX: see MANIFEST.md for every file, what it does, and dependencies.

Generates a SigMF recording of Pete Wyckoff KA3WCA "Venus Bounce Transmitter
Spiral #2" and plays it through a USRP B210 (GNU Radio or uhd). For the EME station
test and for sharing with Dwingeloo / Stockert / Effelsberg. ASCII only, as usual. 

## Files
(see MANIFEST.md for full list. This one will be removed soon.)
- eve_tx_sigmf.py            generator (numpy + galois + sigmf) used for monolithic SigMF files.
- eve_spiral_smoke.sigmf-data/.sigmf-meta  ready-to-play SMOKE TEST (22 s, 44 MB,
<<<<<<< HEAD
                            250 kSps, comb offset 25 kHz off DC). Validated the whole
                            B210 chain and shows the 11 tones. NOT the full frame.
=======
                            250 kSps, comb offset 25 kHz off DC). Validates the whole
                            B210 chain and shows the 11 tones. NOT the full frame. 
>>>>>>> 61b2c3f75dbe78bea2d3b63f5828c04477155647
- fig_sigmf.png             spectrogram proof (11 M-ary tones stepping through the band)

## The design (verified from Pete's June 2026 block diagram)
    106 payload bits (message + CRC) -> BCH(127,106) -> 11 x M-ary(4096) symbols -> NCO
    NCO:  s_n = exp( i * 2*pi * d_{floor(n/Tsym)} * (2*R_bw) * n / Fs )
    tone(d) = d * (2 * R_bw) = d * 5.74 Hz          d in [0, 4095]
    R_bw    = 2.87 Hz        (FFT bin = Doppler-spread forecast, so 1 guard bin between tones)
    M       = 4096 (12 bits/sym)   N_symbols = 11
    T_sym   = 164.794 s     (RX Monte Carlo for C/N0 = 0 dB)
    BW      = 4096 * 5.74   = 23.5 kHz  (Pete's "~22 kHz")
    T_tx    = 11 * T_sym    = 1812.734 s  (~30 min 12.7 s)
Verified in code: every symbol's synthesized tone lands on d*5.74 Hz; constant
envelope; BCH encode/decode round-trips; SigMF validates. Whew!

## Generate
    pip install numpy galois sigmf
    # smoke test (short, to validate the chain / B210):
    python3 eve_tx_sigmf.py --smoke -o eve_spiral_smoke --freq-offset 25000
    # FULL design frame (~30 min). At 250 kSps this is ~3.6 GB, make locally
    # you can resample in GNU Radio if you want, or use a shorter --tsym for EME (see below):
    python3 eve_tx_sigmf.py -o eve_spiral_full --freq-offset 25000 or more to get away from LO
    # EME test: the Moon link is ~223 dB stronger than Venus, so you do NOT need
    # 165 s symbols. A few seconds each is plenty and keeps the file small.
    python3 eve_tx_sigmf.py --tsym 5 -o eve_spiral_eme --rf 1296e6 --freq-offset 25000

Options: --fs (sample rate, default 250000), --tsym, --rf (metadata center freq),
--freq-offset (move comb off DC), --amplitude, --message (packed to 90 bits).

## Play through a B210
SigMF is complex float32, little-endian (cf32_le), interleaved I,Q.

GNU Radio (we use UHD plain so this is untested):
  SigMF Source (gr-sigmf)  ->  UHD: USRP Sink (B210)
  - USRP Sink: Ch0 center freq = your RF (e.g. 1296 MHz EME), samp_rate = file Fs
    (250 kSps for these files), gain to taste, antenna TX/RX.
  - If gr-sigmf is not installed: File Source (Complex Float 32, repeat off) reading
    the .sigmf-data, then set samp_rate manually to match the meta file.
  - If you generated at a low Fs (e.g. 48000) put a Rational Resampler before the
    sink (e.g. interp 250000/48000) so the B210 runs at a comfortable rate.
Command-line alternative: convert to a raw fc32 and use uhd_siggen / a small uhd
Python TX script at the same samp_rate. Whatever works for you. Remote Labs uses UHD.

## Practical notes (please read before keying up)
- DC / LO leakage: Pete's mapping puts d=0 at 0 Hz. On real hardware the B210 LO
  leaks at DC. These files use --freq-offset 25000 to move the whole comb to
  25.0 - 48.5 kHz above the tune frequency. Tune the B210 25 kHz LOW so the RF lands
  where you intend, and have the receiver subtract the same 25 kHz. Set offset 0 only
  if you want textbook picture on the Spectrum Analyzer for some reason. 
- Sample rate: 250 kSps is comfortable for the B210. The signal is only ~23.5 kHz
  wide, so most of the band is empty guard. This is fine.
- Amplitude is 0.8 (constant envelope). Set B210 gain conservatively and check your
  spectrum. This waveform is constant-envelope so it is friendly to a saturated PA.
- Time/frequency reference: Pete's chain assumes a shared H-maser via White Rabbit.
  For EME you can start with the B210's own reference. For the real Venus run use the
  disciplined reference. The Remote Labs B210 has a GPSDO. 

## Receiver side (documented from Pete's slide, implemented next)
  - astropy removes bulk Doppler shift + Doppler rate of the path.
  - FFT at R_bw (bin = 2.87 Hz, frame ~= 1/R_bw = 0.348 s). Every 0.348 chunk is
    coherently integrated. Take advantage of the coherence time!
  - Per symbol, combine ~440 frames NON-coherently (magnitude), pick the peak bin ->
    d_m; the tone spacing 2*R_bw keeps one guard bin so spread <= R_bw does not leak.
  - De-map 11 symbols -> 132 bits -> take 127 -> BCH(127,106) decode -> check CRC-16.

## Matched RX has been built
M=4096, R_bw=2.87, 2*R_bw spacing, 11 symbols, T_sym, BCH(127,106), the NCO equation. 
  - payload split: 90 message bits + CRC-16-CCITT (poly 0x1021, init 0xFFFF) = 106.
  - BCH: galois narrow-sense systematic, genpoly
    x^21+x^18+x^17+x^15+x^14+x^12+x^11+x^8+x^7+x^6+x^5+x+1.
  - bit->symbol: 127 coded bits zero-padded to 132, MSB-first groups of 12.
  - one guard offset (--freq-offset) for DC and baseband one-sided per his equation.
All parameters and the exact symbol values are written into each .sigmf-meta.

## Does it comply with our C/N0 = 0 to -1 dB-Hz channel?  (eve_link_check.py)

Short answer: YES at 0 dB-Hz, knife-edge at -1, needs Effelsberg below that. This
reproduces Pete's "RX Monte Carlo for C/N0 = 0 dB" that sized T_sym.

Verified Monte-Carlo results (M=4096, 473 non-coherent frames/symbol, AWGN):

  C/N0 (dB-Hz)   symbol err   frame err (11 sym)   verdict
  +0.65 (CAMRAS)   0.0007        0.007              closes
   0.00            0.0020        0.022              closes   <- Pete's design point
  -1.00            0.042         0.38               marginal
  -1.33 (Dwin Oct) 0.080         0.60               fails    <- why Effelsberg exists
  +11.94 (+Effel)  0.0000        0.000              closes

How it is computed: the link budget gives C/N0. Per-frame FFT-bin SNR is
gamma_f = (C/N0 linear)/R_bw (coherent within one 0.348 s frame = the coherence time).
Combining N=473 frames non-coherently, the true tone bin is 0.5*noncentral_chi2(2N,
2N*gamma_f), the other M-1 bins are 0.5*central_chi2(2N). 
FER = 1-(1-SER)^11 because a wrong M-ary symbol flips ~6 of its 12 bits, which
BCH(127,106) t=3 cannot fix. So the frame needs all 11 symbols right.

Cross-checked: the abstract chi-square model was validated against a genuine
synthesize->add-noise->FFT->combine->detect loop (agreed to ~1% SER). See
end_to_end_check(). This is based off of Pete's MATLAB work and digital comm theory.

- Real Doppler spread adds a few dB of intra-frame loss. 
  Pete set the frame length = coherence time to keep this small, but test it.
- FER assumes hard M-ary decisions and no BCH rescue of symbol errors.
- Frame count: 473 = T_sym*R_bw; Pete's slide says 440 (guard/overlap). 

Run it:
      python3 eve_link_check.py          # cross-check + the table above
      python3 -c "import eve_link_check as L; print(L.ser_fast(-1.0, coh_loss=0.7))"

The waveform is matched to the exact channel the link budget predicts, to the best of our
ability. At the CAMRAS-measured +0.65 dB-Hz it closes with margin; at 0 dB-Hz it closes; the
0-to-1 dB band is the design edge; and Dwingeloo-alone October (-1.33 dB-Hz) is below
the edge. Which is precisely the gap Effelsberg's +13 dB fills.

## Monostatic operation for DSES (they hear their own echo)

eve_tx_gated.py, eve_tx_rx_b210.py, eve_rx.py

DSES runs one dish and wants to receive its own signal. A dish cannot receive while its
PA is keyed, so each symbol is a separate TX burst. Between bursts the PA is off, the
sequencer switches the dish from PA to LNA, and DSES records its own echo one round trip
later. This works because a symbol (164.794 s) is shorter than the Venus round trip
(~272 s at 40.82 Mkm), so TX finishes ~107 s before the echo returns. Same situation for
Hello Giggy EME, assuming the chain can switch fast enough to send sub-2.5 second transmissions.

LNA SAFETY (non-negotiable ordering; a hardware sequencer like the Kuhne SEQ 4 enforces it):
  Straight from Kuhne handbuch:
  TX: PTT on -> PA on + relay DISH->PA (LNA isolated) -> settle -> RF on.
  RX: RF off -> PA unkeys (guard) -> PTT off -> relay DISH->LNA -> settle -> RX on.
  RF power is ZERO whenever the relay is in or moving to RX. Never reorder this.

Planned Workflow (needs to be updated):
  1. Build the schedule (prints TX/RX/PTT times, checks W<RTT and LNA safety):
        python3 eve_tx_gated.py --distance-mkm 40.82 --tsym 164.794 -o eve_gated_venus
     For the EME bench (shorter round trip), use short symbols and get a playable file:
        python3 eve_tx_gated.py --eme --tsym 1.0 -o eve_gated_eme --write-iq
  2. Run the station (HARDWARE BRING-UP REQUIRED; --dry-run prints the sequence first):
        python3 eve_tx_rx_b210.py eve_gated_venus_schedule.json --rf 2304e6 --dry-run
     It keys the sequencer via B210 GPIO, JIT-synthesizes each burst, then records the
     echo to eve_rx_capture.sigmf-* . Verify GPIO pin / sequencer wiring / timing on the
     bench at low power with a scope BEFORE going on the air.
  3. Decode the capture:
        python3 eve_rx.py eve_rx_capture.sigmf-meta --f-dopp <Hz> --f-rate <Hz/s>
     Doppler shift + rate come from astropy/helper script for the path/time.
     Output: the 11 d_m, the recovered message, and CRC OK/FAIL.

Receiver (eve_rx.py) -- what to do with a recording:
  remove Doppler (shift+rate) -> per symbol (this will be updated to be more often): 
  frame at 1/R_bw, FFT, sum frame powers non-coherently, pick the peak tone bin
  which gives d_m and then BCH(127,106) decode and then CRC-16 check.
  Self-test (TX -> simulated echo: Doppler + AWGN at C/N0=0 -> decode, recovers the
  message with 11/11 symbols and CRC OK):
        python3 eve_rx.py --selftest

Message capacity: the payload is 90 message bits + 16 CRC = 106. That is 11 ASCII
characters. Keep the message short (a callsign and a tag). Want more characters? 
It's a whole other frame to get another 11 characters. 

Bistatic note: Pete's own title is Dwingeloo-Venus-Stockert -- bistatic. If a second
site receives, it just listens continuously (no TX/RX conflict, no gating) and you also
get Effelsberg's big dish and the +13 dB. Gating is only needed when one dish must both
transmit and receive.
