# ORI EVE "Hack the Planet" -- figure + deck pipeline

Self-contained scripts that regenerate every figure and the 16-slide deck for the
RF Village / DEF CON "Hack the Planet" talk (ORI Earth-Venus-Earth). ASCII only.

## Requirements
- python3 with numpy, matplotlib  (pip install numpy matplotlib)
- node with pptxgenjs  (npm install pptxgenjs)  -- only for build_deck.js

## Files

Engine / data (imported by the figures; no editing needed to reproduce):
- eve_style.py            Shared terminal (green-phosphor) matplotlib style
                          (COL palette + apply_style() + terminal_frame() + SLIDE).
- eve_model.py            Link budget model, reconstructed formula-for-formula from
                          Link_Budget_Modeling.pdf. Running it prints a self-test that
                          reproduces the notebook's DSES numbers (+1.67, -0.66) and the
                          CAMRAS white-paper March-2025 value (-0.681 vs -0.678).
                          link_budget(...) takes tsys_override for the measured 65 K.
- eve_conjunction_data.py VERIFIED dynamic-albedo conjunction scan (date, rho_eff,
                          elevation, distance, cnr) transcribed from the notebook.
- compute_sites.py        Runs the validated model for DSES / Dwingeloo / Effelsberg
                          and prints the best-day C/N0 and the Effelsberg advantage.

Figures (each writes ../figs/<name>.png). 14 figures -> slides 2..15:
- fig_bochum.py           2009 dead-carrier origin                       (slide 2)
- fig_carrier_vs_comms.py carrier vs communications thesis               (slide 3)
- fig_crew.py             Dwingeloo / Effelsberg / Bochum + timing       (slide 4)
- fig_bridge.py           link budget bridge, Dwingeloo Oct attempt      (slide 5)
- fig_bounce.py           Venus ~13% reflector                          (slide 6)
- fig_why_1hz.py          why CNR is quoted in 1 Hz                      (slide 7)
- fig_validation.py       CAMRAS measured vs predicted, +0.645 vs +0.560 (slide 8)
- fig_rescue.py           Oct 2026 Dwingeloo alone vs +Effelsberg        (slide 9, hero)
- fig_handoff.py          the spin damages information (SO WHAT CLOSES IT) (slide 10)
- fig_coherence.py        Doppler spread -> coherence-time tax (~13 dB)  (slide 11)
- fig_zadoff.py           Zadoff-Chu chirp for acquisition               (slide 12)
- fig_pete_design.py      Pete's Spiral: BCH -> M-ary(4096) -> NCO       (slide 13)
- fig_mary.py             why M-ary orthogonal (power-efficient)         (slide 14)
- fig_eme.py              prove it on the Moon (EME testbed)             (slide 15)
- fig_noise.py            system noise 65 K meas vs 49 K modeled         (spare; not in deck)

Deck (16 slides: title + the 14 figures + close):
- build_deck.js           Assembles ../HackThePlanet_EVE.pptx from ../figs, with full
                          speaker notes per slide. Slide 1 title and slide 16 close are text.

## Reproduce everything
Run from THIS folder (figures go to ../figs, deck goes to ../HackThePlanet_EVE.pptx):
    python3 eve_model.py            # sanity: must match notebook values
    python3 compute_sites.py        # site numbers
    for f in fig_*.py; do python3 "$f"; done
    node build_deck.js              # -> ../HackThePlanet_EVE.pptx  (16 slides)

## Scripted deck vs the LIVE deck (important)
build_deck.js rebuilds the CLEAN scripted deck (HackThePlanet_EVE.pptx): figures +
speaker notes, no narration videos. The delivered talk (HackThePlanet_EVE_LIVE.pptx)
is a HAND-FINISHED artifact -- slides 11-16 were appended to the narrated deck and the
corner narration videos were later removed for the live version. No build_deck.js run
reproduces LIVE byte-for-byte. Treat LIVE as the presentation master; treat this pipeline
as the reproducible source of truth for the figures and structure.

## Provenance / caveats
- DSES outputs reproduce verified notebook values.
- Dwingeloo / Effelsberg outputs are COMPUTED with the validated model; not printed
  in the notebook. Dwingeloo T_sys = 65 K is MEASURED (Telkamp); Effelsberg ~49 K is
  MODELED and labeled as such.
- October uses the dynamic albedo (~0.117). Static 0.152 is only for reproducing the
  March 2025 CAMRAS validation.
- 23 cm albedo and Effelsberg T_sys are the two values to replace if better numbers
  become available; everything regenerates from the scripts.
