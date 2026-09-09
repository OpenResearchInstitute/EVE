# ORI EVE transmit waveform

_Pete Wyckoff's "Spiral" as a SigMF file_

Generates a SigMF recording of Pete Wyckoff (KA3WCA) "Venus Bounce Transmitter
Spiral #2" and plays it through a USRP B210 (GNU Radio or uhd). For the EME station
test and for sharing with Dwingeloo / Stockert / Effelsberg. ASCII only.

## Files
- eve_tx_sigmf.py            generator (numpy + galois + sigmf)
- eve_spiral_smoke.sigmf-data/.sigmf-meta  ready-to-play SMOKE TEST (22 s, 44 MB,
                            250 kSps, comb offset 25 kHz off DC). Validates the whole
                            B210 chain and shows the 11 tones. NOT the full frame.
- fig_sigmf.png             spectrogram proof (11 M-ary tones stepping through the band)

## The design (verified from Pete's June 2026 block diagram)
    106 payload bits (message + CRC) -> BCH(127,106) -> 11 x M-ary(4096) symbols -> NCO
    NCO:  s_n = exp( i * 2*pi * d_{floor(n/Tsym)} * (2*R_bw) * n / Fs )
    tone(d) = d * (2 * R_bw) = d * 5.74 Hz          d in [0, 4095]
    R_bw    = 2.87 Hz        (FFT bin = Doppler-spread forecast; 1 guard bin between tones)
    M       = 4096 (12 bits/sym)   N_symbols = 11
    T_sym   = 164.794 s     (RX Monte Carlo for C/N0 = 0 dB)
    BW      = 4096 * 5.74   = 23.5 kHz  (Pete's "~22 kHz")
    T_tx    = 11 * T_sym    = 1812.734 s  (~30 min 12.7 s)
Verified in code: every symbol's synthesized tone lands on d*5.74 Hz; constant
envelope; BCH encode/decode round-trips; SigMF validates.

## Generate
    pip install numpy galois sigma
    # smoke test (short, to validate the chain / B210):
    python3 eve_tx_sigmf.py --smoke -o eve_spiral_smoke --freq-offset 25000
    # FULL design frame (~30 min). At 250 kSps this is ~3.6 GB; prefer a lower Fs
    # you will resample in GNU Radio, or a shorter --tsym for EME (see below):
    python3 eve_tx_sigmf.py -o eve_spiral_full --freq-offset 25000
    # EME test: the Moon link is ~223 dB stronger than Venus, so you do NOT need
    # 165 s symbols. A few seconds each is plenty and keeps the file small:
    python3 eve_tx_sigmf.py --tsym 5 -o eve_spiral_eme --rf 1296e6 --freq-offset 25000

Options: --fs (sample rate, default 250000), --tsym, --rf (metadata center freq),
--freq-offset (move comb off DC), --amplitude, --message (packed to 90 bits).

## Play through a B210
SigMF is complex float32, little-endian (cf32_le), interleaved I,Q.

GNU Radio (recommended):
  SigMF Source (gr-sigmf)  ->  UHD: USRP Sink (B210)
  - USRP Sink: Ch0 center freq = your RF (e.g. 1296 MHz EME), samp_rate = file Fs
    (250 kSps for these files), gain to taste, antenna TX/RX.
  - If gr-sigmf is not installed: File Source (Complex Float 32, repeat off) reading
    the .sigmf-data, then set samp_rate manually to match the meta file.
  - If you generated at a low Fs (e.g. 48000) put a Rational Resampler before the
    sink (e.g. interp 250000/48000) so the B210 runs at a comfortable rate.
Command-line alternative: convert to a raw fc32 and use uhd_siggen / a small uhd
Python TX script at the same samp_rate.

## Practical notes (please read before keying up)
- DC / LO leakage: Pete's mapping puts d=0 at 0 Hz. On real hardware the B210 LO
  leaks at DC. These files use --freq-offset 25000 to move the whole comb to
  25.0 - 48.5 kHz above the tune frequency. Tune the B210 25 kHz LOW so the RF lands
  where you intend, and have the receiver subtract the same 25 kHz. Set offset 0 only
  for a purely faithful bench test.
- Sample rate: 250 kSps is comfortable for the B210. The signal is only ~23.5 kHz
  wide, so most of the band is empty guard -- that is fine.
- Amplitude is 0.8 (constant envelope). Set B210 gain conservatively and check your
  spectrum; this waveform is constant-envelope so it is friendly to a saturated PA.
- Time/frequency reference: Pete's chain assumes a shared H-maser via White Rabbit.
  For EME you can start with the B210's own reference; for the real Venus run use the
  disciplined reference.

## Receiver side (documented from Pete's slide, not implemented here)
  - astropy removes bulk Doppler shift + Doppler rate of the path.
  - FFT at R_bw (bin = 2.87 Hz, frame ~= 1/R_bw = 0.348 s).
  - Per symbol, combine ~440 frames NON-coherently (magnitude), pick the peak bin ->
    d_m; the tone spacing 2*R_bw keeps one guard bin so spread <= R_bw does not leak.
  - De-map 11 symbols -> 132 bits -> take 127 -> BCH(127,106) decode -> check CRC-16.

## Provenance / choices (so a matched RX can be built)
FAITHFUL to Pete's slide: M=4096, R_bw=2.87, 2*R_bw spacing, 11 symbols, T_sym,
BCH(127,106), the NCO equation. CHOICES I made where his slide was silent (documented
in the .sigmf-meta ori:design block, change if you learn his intent):
  - payload split: 90 message bits + CRC-16-CCITT (poly 0x1021, init 0xFFFF) = 106.
  - BCH: galois narrow-sense systematic, genpoly
    x^21+x^18+x^17+x^15+x^14+x^12+x^11+x^8+x^7+x^6+x^5+x+1.
  - bit->symbol: 127 coded bits zero-padded to 132, MSB-first groups of 12.
  - one guard offset (--freq-offset) for DC; baseband one-sided per his equation.
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
2N*gamma_f), the other M-1 bins are 0.5*central_chi2(2N); symbol error if any beats it.
FER = 1-(1-SER)^11 because a wrong M-ary symbol flips ~6 of its 12 bits, which
BCH(127,106) t=3 cannot fix -- so the frame needs all 11 symbols right.

Cross-checked: the abstract chi-square model was validated against a genuine
synthesize->add-noise->FFT->combine->detect loop (agreed to ~1% SER). See
end_to_end_check().

Caveats to hammer on (this is an AWGN reference, i.e. optimistic):
- Real Doppler spread adds a few dB of intra-frame loss. Explore it:
      ser_fast(cn0_db, coh_loss=0.7)   # 0<coh_loss<=1 scales per-frame SNR
  Pete set the frame length = coherence time to keep this small, but test it.
- FER assumes hard M-ary decisions and no BCH rescue of symbol errors (true for
  orthogonal signaling, where an error goes to a uniformly-random tone).
- Frame count: 473 = T_sym*R_bw; Pete's slide says 440 (guard/overlap). Try both.

Run it:
      python3 eve_link_check.py          # cross-check + the table above
      python3 -c "import eve_link_check as L; print(L.ser_fast(-1.0, coh_loss=0.7))"
Knobs: n_frames, m, trials, coh_loss. This is the file to torture-test.

Bottom line: the waveform is matched to the exact channel the link budget predicts.
At the CAMRAS-measured +0.65 dB-Hz it closes with margin; at 0 dB-Hz it closes; the
0-to-1 dB band is the design edge; and Dwingeloo-alone October (-1.33 dB-Hz) is below
the edge, which is precisely the gap Effelsberg's +13 dB fills.
