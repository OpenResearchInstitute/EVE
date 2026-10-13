# EME bring-up checklist -- 13 cm / 2304 MHz (DB6NT chain + B210)

Bringing the Kuhne/DB6NT 13 cm parts up safely so we can use them without
killing the LNA. 2304 MHz is also the Venus/EVE band, so this station does double duty.
Staged: each step proves one thing and leads directly to the next.

Status: feed working + septum isolation MEASURED (22.5 dB worse feed, Paul, 2026-09-12);
B210 TX output characterized (+14.11 dBm clean at tx-gain 80, 2026-09-16); IF driver amp in
hand; PA datasheet + 23 G4 handbuch now in hand (folded in below); radio chain not yet
cabled. Bench loopback (B210 direct, TX->50 dB->RX->decode, incl. Doppler) VERIFIED (see
BENCH.md). Control wiring = ARCHITECTURE B (new section below; drawn in the EVE Station
Wiring diagram in draw.io).

OUR PARTS (from the order):
  MKU 23 G4         13 cm transverter  (B210 IF  <->  2304 MHz)
  MKU PA 13CM-20W A2  20 W LDMOSFET PA
  MKU LNA 231 AH-SMA  super-low-noise 13 cm preamp (mast-mount)
  SEQ 4             SMD sequence controller, max 18 A, +12..32 V (built for 26..32 V PAs)
  RSP-150-27        27 V / 150 W supply (PA + transverter rail)
  SK 150-62 heatsink + 24 V fan   (PA cooling)
  KU BT 6000 SMA    bias tee (DC up the coax to power the mast LNA)

we bought a MKU IF SW but did not end up needing it. 

NOW KNOWN (from the datasheets):
  - MKU 23 G4 IF = 144..146 MHz for the 2304..2306 MHz band. So THE B210 TUNES TO ~144 MHz
    (LO = 2160 MHz). Every on-air --rf below is the IF (e.g. --rf 144e6), NOT 2304e6.
  - MKU 23 G4 has a SINGLE IF connector (common TX/RX) and SEPARATE 2304 TX-out / RX-in on
    the RF side. One IF cable to the B210's TX/RX port does both directions (SINGLE IF PORT).
  - MKU 23 G4 has a BUILT-IN adjustable sequencer + a control output for a coaxial relay /
    extra amp stages, and PTT = contact closure to GROUND (or +12 V on the IF cable). So
    the transverter can key and sequence from ONE PTT line; the B210 just grounds that pin.
  - MKU 23 G4 has an external 10 MHz reference input. Without it, LO is +/- 2..3 ppm =
    ~ +/- 5 kHz at 2304 -- fine for EME detection, but for EVE/Venus frequency precision testing
    we would want a 10 MHz GPS reference feeding the transverter (see "references" below).
  - MKU 23 G4 TX: IF input 0.5..5 W for ~1 W out (the G4's own window). The PA, though, wants
    only ~400 mW drive (see PA line), so the G4's ~1 W would OVERDRIVE it by ~4 dB. There is a
    procedure in the Handbuch where  you use the LEDs to get the drive level aligned.
  - MKU LNA 231 AH: NF 0.4 dB, gain 16 dB, supply 9..15 V @ 15 mA, and MAX INPUT 1 mW
    (0 dBm) -- a HARD limit. It has NO built-in coaxial relay (datasheet says so, we believe it).
    Powered by a Kuhne bias tee.
  - MKU PA 13CM-20W A2 (datasheet now in hand): 27 V rail (RSP-150-27), ~20 W out from
    ~400 mW TYP drive (+26 dBm) -- NOT ~1 W. The 23 G4's ~1 W would overdrive it, so drive into
    the PA must be held near ~400 mW. This is adjustable according to Handbuch.
    Datasheet also gives: >=40 dB harmonic suppression; forward + reflected
    power detectors; 1.8:1 VSWR foldback/shutdown; reverse-polarity protection; and NO built-in
    T/R relay (the septum's two ports handle T/R).
  - SEQ 4: two outputs. First is the PA supply switch (max 18 A) + coax-relay output (max 4 A, switched
    to GND), with a 100..150 ms delay between them. On TX it switches the relay first,
    then (after the delay) enables the PA supply; reverse on RX. That delay is what keeps
    RF off while the relay moves and the LNA (1 mW limit) isolated. Built for MKU transverters.

STILL TO CONFIRM (from the 23 G4 full manual):
  [x] LNA power path: via the KU BT 6000 bias tee up the coax, or its local solder pin.
  [x] PA drive level: ~400 mW TYP (datasheet). Supply current at 27 V for ~20 W: confirm the
      exact figure from the datasheet before sizing the SEQ 4 PA output (SEQ 4 rated 18 A, so
      ample; the number matters for the fuse/wire, not the sequencer).
  [x] Whether the 23 G4 output is adjustable down to ~400 mW, or a pad is needed G4->PA.
  [ ] 23 G4 IF mode: standard (0.5..5 W, needs the driver amp) vs LOW-power IF (mW, needs a
      pad). Picks DRIVE OPTION A vs B in GAP 2, confirm from the handbuch.

================================================================================
PARTS GAPS RESOLVED
================================================================================
GAP 1 -- COAX RELAY TO PROTECT THE LNA (OK1DFC septum feed). Needed; NOT in the order.
  The septum feed is DUAL-PORT (one circular sense each: TX on one port, RX on the other).
    - PA = 20 W = +43 dBm. LNA absolute max input = 1 mW = 0 dBm. We need ~43 dB of
      protection just to reach the limit; DB6NT's manual asks for ~50 dB, RX input <= 1 mW,
      and "strongly recommends a sequence controller." Which is why we bought one. 
    - Septum port-to-port isolation is only ~20..30 dB (and return loss is NOT isolation)
      Even a really great 30 dB leaves the LNA ~13 dB over its damage limit. No
      realistic septum isolation is enough on its own. The relay is the right thing to do.

  ARCHITECTURE (septum + sequenced RX-protection coax relay):
    See the draw.io diagram and the timing screenshots. 
    But here's some words that should help.  

    PA connects directly to the septum TX-sense port
    LNA input <-- [coax relay common]
        relay RX position: common -> septum RX-sense port     (receive)
        relay TX position: common -> 50 ohm LOAD              (LNA terminated + isolated)
        An absorptive coax switch instead of a reflective one would be a bit better, but
        is not anywhere close to being required. 
    SEQ 4 drives the relay (antenna-relay output, 4 A to GND) AND the PA 27 V supply:
        TX: relay -> TX (LNA on the load) -> wait 100..150 ms -> PA supply ON
        RX: PA supply OFF -> wait -> relay -> RX (LNA on the port) see timing diagram from Kuhne.
  During TX the LNA never sees the port: it is on a 50 ohm LOAD (never an open or short),
  and the port leakage must cross the relay's OFF isolation to reach it. The LNA input thus
  sees 50 ohm at all times. Antenna in RX, load in TX. Protection stacks up really nicely.
        LNA_input_dBm(TX) = +43 - S_septum - S_relay_off
  The relay's off-isolation alone (60..80 dB typical at S-band) already puts us far under
  1 mW; the septum adds margin. Terminating the LNA in 50 ohm (not open/short) is the
  right choice. The LNA always sees a good match, in RX or TX. This should be very reliable. 

  ACCEPTANCE (go / no-go before the LNA ever goes on the port):
    Require:  +43 - S_septum - S_relay_off  <=  -10 dBm   (i.e. >= 10x margin under 1 mW).
    *** MEASURED + PASSED (2026-09-12, Paul): septum S21 at 2304 MHz = 22.5 dB (feed 1,
    -22.46 dB) and 23.7 dB (feed 2, -23.66 dB); RSA5065N, CK106A cal, matches sim -22 dB.
    Using the worse feed S_septum = 22.5 dB and the MSP2T-18-12+ IN-J1 isolation:
      spec MIN 75 dB : LNA sees +43 - 22.5 - 75  = -54.5 dBm   (44 dB under the -10 bar) PASS
      typ ~113 dB    : LNA sees +43 - 22.5 - 113 = -92.5 dBm                              PASS
    Switch J1 power in TX = +43 - 22.5 = +20.5 dBm = 0.11 W, << its 10 W rating. PASS.
    The LNA is safe by a huge margin; the septum measurement also confirms J1 << 10 W. ***

  SELECTED RELAY: Mini-Circuits MSP2T-18-12+ (Paul bought these). VERIFIED from datasheet:
  mechanical SPDT, DC-18 GHz, SMA, REFLECTIVE FAILSAFE, 12 V coil (~180 mA), 20 ms.
  ISOLATION at 2304 MHz ~113 dB TYP (table: 115.7 dB @2 GHz, 107.3 dB @3 GHz; spec MIN
  75 dB for 1-8 GHz). Insertion loss ~0.06 dB at 2.3 GHz (negligible NF hit). RF power
  10 W COLD-switching; hot-switching only 1 W. Excellent for this role, with three rules:
    1. RX / LNA SIDE ONLY. 10 W rating < the 20 W PA -- NEVER in the PA path. (Septum
       architecture keeps the PA on the TX port, direct.) The switch's J1 port only sees
       the post-septum leakage = +43 - S_septum; with ~20 dB septum that is ~+23 dBm
       (0.2 W) << 10 W. Measure S_septum to confirm it is > 3 dB (so J1 stays under 10 W).
    2. COLD-SWITCH ONLY. The SEQ 4 moves the relay while RF is OFF (its whole purpose), so
       you get the 10 W rating and the long life (10M cycles). NEVER switch with the PA
       keyed (hot-switching is 1 W and wears it out).
    3. REFLECTIVE -> add an external 50 ohm LOAD on the throw the LNA uses in TX, so the LNA
       stays terminated (our standard). Wiring: LNA -> IN (common); J1 -> septum RX port;
       J2 -> 50 ohm load. De-energized = IN-J1 (RX, LNA on antenna -- failsafe/rest state);
       energized 12 V = IN-J2 (TX, LNA on load). SEQ 4 energizes the coil for TX, BEFORE the
       PA-supply step (100..150 ms delay). Power loss -> de-energized -> RX -> safe.
       The J2 load sees only the leakage through the switch (~-92 dBm, picowatts) -- so ANY
       real 50 ohm SMA terminator works; power rating irrelevant, just want a decent match
       at 2.3 GHz. Existing SMA dummy loads are fine; no need to buy a better one.
  BUDGET: during TX the LNA (IN) is isolated from the leaking J1 port by the switch's
  IN-J1 isolation. Even at the 75 dB spec MIN and ZERO septum help: +43 - 75 = -32 dBm,
  already past the -10 dBm acceptance. Typical (113 dB switch + ~20 dB septum) puts the LNA
  near -95 dBm. The LNA is safe by the switch alone; the septum is margin.
  (An ABSORPTIVE version, MSP2TA-18-12+, would internally absorb the off-port leakage
  instead of reflecting it. This is slightly cleaner, not required. The reflective + external
  load is safe.)

RF chain (MONOSTATIC because I don't think we can hear ourselves):
  IF side is ONE cable; the B210's bidirectional TX/RX port does BOTH directions (the 23 G4
  has a single IF port) so just make sure it's configured in code.
    B210 TX/RX to 23 G4 IF (144 MHz)
  RF side of the G4 SEPARATE 2304 TX-out and RX-in (confirmed by looking at the part):
    G4 TX-out -> PA 13CM-20W -> septum TX port -> dish                          (transmit)
    dish -> septum RX port -> MSP2T -> LNA 231 AH -> KU BT 6000 (bias) -> G4 RX-in   (receive)
  PTT (both key on contact-closure-to-GROUND, so drive ONE active-low line):
    B210 GPIO --(low = TX)--> [ SEQ 4 PTT in ] and [ 23 G4 PTT pin ], wired together.
    SEQ 4 sequences: coax relay -> TX, wait 100..150 ms, PA 27 V supply ON. (Reverse on RX:
    PA off, wait, relay -> RX.) The 23 G4 keys its own IF/mixer TX path on the same ground.
    Result: relay settled + LNA isolated BEFORE the PA can put out power. LNA (1 mW) safe.
  TIMING: the SEQ 4 delay is 100..150 ms. The gated schedule's PTT lead/lag and guard must
  be >= ~200 ms (SEQ 4 delay + relay travel + margin). At EME (RTT 2.56 s, ~1 s symbols)
  there is ~1.5 s of gap and this is plenty. RSP-150-27 = 27 V; SK150-62 + fan cool the PA.
  See timing diagram from Kuhne. 

================================================================================
SINGLE IF PORT one cable unites the B210 and the transverter
================================================================================
The 23 G4 has ONE IF connector (common TX/RX). The B210 has two SMA ports, but the TX/RX
port is BIDIRECTIONAL: TX comes out of it, and RX can come from it too (the B210's internal
T/R switch duplexes -- set rx antenna = TX/RX). So a single-port transverter pairs naturally
with the B210's single TX/RX port: ONE IF cable, no external IF switch. eve_b210.py --rx-ant
now DEFAULTS to TX/RX for exactly this (RX2 is only for a two-port bench loopback).

Do NOT put a one-way device in that shared IF line. A pad is bidirectional (TX down through
it, RX back up) and is fine. A driver AMP is one-way and would block receive so if the
drive stage must be an amp (standard IF mode), we then need an IF T/R switch (the MKU SW IF,
on hand) to route around it, OR use the low-power IF mode + pad and skip the switch entirely. 
The pad path is simplest: B210 +14 dBm -> pad -> G4 IF, same cable both ways, but I do not 
think we need to do this. The RF side is separate (G4 TX-out / RX-in), so the PA and LNA 
each get their own line and nothing in the 20 W path is switched.

================================================================================
MOON DOPPLER SPREAD & COHERENCE (this station is the low-coherence testbed)
================================================================================
Tone spacing and coherence time are set by the target's Doppler spread, computed per
(frequency, target) in eve_params.py. Run EME with --target moon.

Moon libration spread (K1JT 2010, "Frequency-Dependent Characteristics of the EME Path",
w10 = 90%-power full width, stations at +/-40 deg lat; varies through the month):
    1296 MHz : ~10 Hz  ->  R_bw 10 Hz, coherence ~0.10 s,  comb ~82 kHz  (fits 250 kHz)
    2304 MHz : ~20 Hz  ->  R_bw 20 Hz, coherence ~0.050 s, comb ~164 kHz (needs fs >= ~400 kHz)
For contrast, Venus at inferior conjunction is near-specular: ~2 Hz at 1299.5 (measured,
CAMRAS/Estevez), coherence ~0.5 s. So the MOON is ~5x more Doppler-spread than Venus at the
same band. Moon is a rough diffuse disk vs a specular glint, not a faster spin. That makes this EME
station a HARDER coherence test than the Venus target it was built to test for. The Moon's far
stronger link makes up for a lot of SNR.

CONSEQUENCE at 2304 Moon: spacing = 2*R_bw = 40 Hz, the 4096-tone comb is ~164 kHz wide, and
the sample-rate floor is ~400 kHz, which is now ABOVE the 250 kHz bench rate. Raise --rate (the B210 does
tens of MHz) or reduce M for a 2304 Moon run. eve_tx_sigmf warns if --fs is below the floor. 
This should be a solved problem in the code. 

================================================================================
ARCHITECTURE B -- CONTROL WIRING (DB6NT's intended split: G4 SW OUT + SEQ 4)
================================================================================
This is the wiring drawn in the EVE Station Wiring diagram. Two sequencers split the T/R job
on purpose; each drives only what it has the current to drive.

ONE PTT SOURCE, fanned out:
  B210 GPIO_0 (3.3 V, FP0 bank) -> opto/MOSFET -> a single PTT node = "contact closure to GND".
  That node drives BOTH, wired together:
    - 23 G4 PTT pin    (keys the transverter's own TX path + its SW OUT)
    - SEQ 4 PTT input  (starts the SEQ 4 sequence)
  GPIO_0 HIGH = TX (PTT pulled to GND); LOW / rest / crash = RX. eve_b210.py forces it LOW on
  start and exit, so a crash cannot leave you keyed.

WHO DRIVES WHAT (why two sequencers, not one):
  - 23 G4 SW OUT: a LOW-current control output (~0.4 A class but confirm exact figure from the
    handbuch). Fine for a coax-relay coil or an exciter-stage enable; TOO WEAK to switch the
    PA's 27 V / ~4 A drain.
  - SEQ 4: this is the muscle. PA output switches up to 18 A; relay (RL) output sinks a coil to GND at
    up to 4 A; 100..150 ms inter-stage delay. So the SEQ 4 owns the two jobs that matter for
    LNA safety -- the LNA-protect relay and the PA drain.

SEQ 4 SEQUENCE ON KEY-UP (hardware order):
    1. SEQ 4 RL out -> MSP2T coil: LNA swings onto its 50 ohm LOAD (LNA protected FIRST).
    2. (after 100..150 ms) SEQ 4 PA out -> PA +27 V drain ON.
    3. RF is LAST: eve_b210.py holds the samples ~0.30 s (GUARD) after PTT, so the B210 does not
       radiate until the relay has settled and the PA rail is up.
  KEY-DOWN reverses: RF stops -> PA drain off -> relay back to RX -> LNA on the port.

WHY NO HIGH-POWER T/R SWITCH? Because the OK1DFC septum has two ports (TX-sense, RX-sense). The PA sits
on the TX port directly. The MSP2T only ever handles the RX side (post-septum leakage, << its
10 W rating). Nothing in the 20 W path is switched. (Isolation math above)

CONFIRM ON THE SCOPE (Stage 2 dry run, PA UNPOWERED, no RF):
  [ ] 23 G4 SW OUT exact current rating + polarity from the handbuch (0.4 A is the class).
  [ ] Measured SEQ 4 inter-stage delay (100..150 ms is the spec -- measure it; it sets GUARD).
  [ ] Rest state AND power-loss state = RX (LNA on the port, PA off) on every line.

================================================================================
PROTECT THE LNA AT ALL COSTS
================================================================================
The LNA must never see transmit power. RF reaches the feed only when the relay is in TX
and the LNA is isolated. On every transition, RF is OFF while the relay moves. The SEQ 4
enforces this order in hardware.
  TX: inhibit RF -> relay feed->PA, LNA isolated -> settle -> PA key / RF on.
  RX: RF off / PA off -> settle -> relay feed->LNA -> LNA bias on -> settle -> RX.

================================================================================
STAGE 0 -- paper (no power)
================================================================================
[x] Pull all Kuhne manuals; fill the "WHAT I DO NOT KNOW" blanks.
[x] Draw block diagram
[x] Confirm dummy load (rated >= 20 W at 2304 which is not just any terminator)

================================================================================
STAGE 1 -- RECEIVE ONLY (zero TX risk; LNA cannot be hurt if you never transmit)
================================================================================
Chain:  feed -> LNA 231 AH -> KU BT 6000 (bias) -> MKU 23 G4 (RX) -> B210 RX (IF)
[ ] Power the LNA per its manual via the KU BT 6000 bias tee (confirm voltage/current)
    Confirm the transverter RX path and its IF.
[ ] In eve_b210.py set --rf to 144e6 (the 23 G4 IF -> 2304); NOT 2304e6. Look at received power:
        python3 eve_b210.py probe
        # RX-only look at the noise floor at the IF (a dedicated rx-only mode can be added)
[ ] SUN/MOON NOISE Y-FACTOR: received noise on cold sky vs Sun/Moon/warm ground should
    rise clearly. That proves feed+LNA+transverter+B210 RX end to end. No rise -> fix RX.
[ ] Point with the K3NG rotator and log the delta. Backlash? There is some.

================================================================================
STAGE 2 -- SEQUENCER (SEQ 4) DRY RUN (no RF, PA unpowered)
================================================================================
Verify LNA-protect ordering BEFORE any RF exists.
[ ] Wire PTT in -> SEQ 4 -> its outputs, per the SEQ 4 manual, to: LNA bias/enable,
    T/R relay (if used), PA key, transverter/exciter PTT.
[ ] With a scope/meter on each line (PA UNPOWERED, no RF), assert PTT and confirm the
    ORDER + delays match expected. Confirm on RX the PA/relay settle BEFORE the
    LNA is re-enabled; on TX the LNA is isolated + relay settled BEFORE PA key.
[ ] Set the SEQ 4 inter-stage delay >= your relay's switch time (from its datasheet).
[ ] PTT logic make sure active-low isn't dumb decision (contact closure to GROUND = TX) 
    for both the 23 G4 and the SEQ. Needed for automation.
[ ] MEASURE SEPTUM PORT-TO-PORT ISOLATION = S21 on the RSA5065N (we measured S11 already
    and S11 is MATCH / return loss and does NOT tell us leakage. Isolation is S21 between the
    two septum ports). Small-signal passive measurement, so the number is valid at 20 W.
    GOTCHA: the S11 calibration does not carry over to S21. You gotta do the calibration
    all over again every time and Rigol's manual says the S11 cal is valid only for S11.
      Setup: LNA and PA DISCONNECTED. Feed aperture at open space / absorber.
      RSA5065N steps:
        1. Mode -> VNA. (Tracking generator turns on automatically in VNA mode.)
        2. Meas -> S21 (transmission). Format = Log Mag (dB).
        3. Set frequency FIRST: center 2304 MHz, span ~200 MHz (or start 2.2 / stop 2.4 GHz).
        4. Cal -> S21 / Through: join the source-out cable to the RF-in cable with an SMA
           female-female adapter (measurement cables), and run the Through cal.
           The trace flattens to ~0 dB. This removes cable loss.
        5. Insert the feed: source-out -> septum TX port; septum RX port -> RF-in.
        6. Marker at 2304 MHz. Value is going to be negative and isolation = its magnitude
           (marker -24 dB -> 24 dB septum isolation).
      Then measure the MSP2T-18-12+ off-isolation the SAME way (source -> relay common,
      relay RX-port throw -> RF-in, relay held in TX state). We should see 
      +43 - S_septum - S_relay_off <= -10 dBm.
      DONE (2026-09-12): feed1 S21 = -22.46 dB, feed2 = -23.66 dB at 2304 (RSA5065N, CK106A).
      Switch IN-J1 isolation taken from the MSP2T-18-12+ datasheet (75 dB min / ~113 dB typ).
      Acceptance PASSES with 44..82 dB margin. 

================================================================================
STAGE 3 -- LOW-POWER TX INTO A DUMMY LOAD (through the sequenced chain)
================================================================================
Feed NOT connected. ... -> PA -> T/R -> DUMMY LOAD.
[ ] Bring the transverter + PA up per their manuals: 27 V from RSP-150-27, fan on the
    SK150-62, correct DRIVE level (set B210 IF output / transverter output so the PA sees
    its rated drive and try not ot overdrive the 20 W PA).
[ ] Key via SEQ 4 (manual/hardware PTT for now). Confirm on a power meter/SA that RF
    appears ONLY in TX state and the level is right.
[ ] After the first keydowns, repeat the Stage 1 Y-factor to confirm the LNA noise figure
    did not degrade (i.e., it was never hit by TX).

================================================================================
STAGE 4 -- ON THE FEED, RECEIVE THE MOON FIRST
================================================================================
[ ] Put the real feed in. Point at the Moon. Confirm Moon noise / a 2304 beacon / other
    stations. Proves the full station RX on-sky. Really optimistic it will work. 
[ ] Lock the B210 to its internal GPSDO (clock/time source = gpsdo; probe shows
    gps_locked/ref_locked). NOTE: the B210 GPSDO does not expose a 10 MHz OUTPUT, and the
    23 G4 LO is only +/- 2..3 ppm (~ +/- 5 kHz at 2304) on its own. For EVE/Venus frequency
    precision, feed the 23 G4 external 10 MHz ref input from a 10 MHz GPS standard (Kuhne
    suggests the ID Elektronik DK2DB). For EME detection the free-running LO is totally fine
    just measure and calibrate out the fixed offset.

================================================================================
STAGE 5 -- MONOSTATIC MOON ECHO (needs automated PTT) -- the ambitious step 
================================================================================
Round trip ~2.56 s, so SHORT symbols. But, there's a challenge. 20 W at 13 cm off the Moon 
is a hard link for this station. Whether we hear our own echo depends heavily on dish 
size/efficiency, and this is a smallish dish.
[ ] Schedule (short symbols, W < RTT with margin). --target moon sets R_bw/spacing from the
    Moon's libration spread (see MOON DOPPLER SPREAD); --rf is the transverter IF, 144e6:
        python3 eve_tx_gated.py --eme --tsym 1.0 --rf 144e6 --target moon -o eve_gated_eme
[ ] AUTOMATED PTT -- DONE in eve_b210.py, via B210 FP0 GPIO_0 (chosen: keep PTT on the
    radio, time-locked to the samples). Drill a hole in the case for a ribbon off J504.
    B210 FP0 GPIO (UHD bank "FP0", 3.3 V CMOS, <= 5 mA/pin -- never drive the sequencer
    directly). Header J504 (0.1" spacing; may be UNPOPULATED so we solder a header if bare).
    PINOUT DEPENDS ON BOARD REV -- confirm visually when open (our probe shows rev 4):
      Older (pre-rev-6, 12-pin): 1=3.3V 2=GPIO_0 3=GPIO_1 4=GPIO_2 5=GPIO_3 6=GND
                                 7=3.3V 8=GPIO_4 9=GPIO_5 10=GPIO_6 11=GPIO_7 12=GND
      Rev-6+ green (10-pin):     1=GPIO_0 2=GPIO_1 3=GPIO_2 4=GPIO_3 5=GPIO_4
                                 6=GPIO_5 7=GPIO_6 8=GPIO_7 9=GND 10=GND
    We use GPIO_0 (mask 0x01) and a GND pin. Interface (GPIO 3.3V -> contact-closure-to-GND):
      GPIO_0 --[1k]--gate  2N7000 (or opto e.g. PC817/6N137 for isolation -- preferred)
                          drain --> PTT line (SEQ 4 IN + 23 G4 PTT pin)
                          source--> GND (shared B210 <-> transverter)
      GPIO_0 HIGH -> device on -> PTT pulled to GND -> TX;  LOW -> PTT floats -> RX.
    VALIDATE WITH ZERO RF FIRST (this is the Stage-2 dry run, driven by the real code):
      python3 eve_b210.py ptt-test            # toggles GPIO_0 ~2 s on/off, NO RF
    Scope: GPIO_0 pin -> our MOSFET/opto -> PTT line -> SEQ 4 relay + PA-supply outputs.
    Confirm logic sense (HIGH=TX), the SEQ 4 ordering (relay before PA), and that rest/
    power-loss = RX. eve_b210.py forces PTT LOW on startup and exit, so a crash cannot leave
    us keyed on. Then the on-air run keys automatically:
      python3 eve_b210.py run --schedule eve_gated_eme_schedule.json --rf 144e6 --ptt ...
    (--ptt asserts GPIO_0 a GUARD before each burst and releases after; guard >= the SEQ 4
    100..150 ms delay + relay travel + margin. eve_b210.py now ships GUARD = 0.30 s; confirm
    it against your measured SEQ 4 delay on the Stage-2 scope dry run.)
[ ] Run against the Moon (remember --rf is the transverter IF, not 2304, so we have --sky-rf):
        python3 eve_b210.py run --schedule eve_gated_eme_schedule.json \
            --rf 144e6 --target moon   # 144 MHz IF -> 2304 via the 23 G4; set gains for driver+PA
    Feed the astropy Doppler to the decode. NOTE: at 2304 the Moon's spread is ~20 Hz (R_bw
    20 Hz, coherence ~0.05 s) vs ~10 Hz at 1296 we have MOON DOPPLER SPREAD for the comb/fs
    consequence (2304 Moon wants fs >= ~400 kHz, above the 250 kHz default quite possibly).

================================================================================
SAFETY THIRD
================================================================================
- LNA: never transmit into it, duh. SEQ 4 + the Stage-2 scope check is what protects it.
- PA (20 W): rated 27 V supply (RSP-150-27), fan-cooled heatsink, correct drive (do not
  overdrive), and an isolator/load if the manual says so. Never key into an unknown load.
- Resolve the T/R relay gap before any monostatic TX.
- Levels: bring the IF up LOW; raise only after confirming PA drive and downstream levels.
- GPSDO (and, if possible, transverter) reference before trusting frequency/Doppler.

================================================================================
CONJUNCTION JUNCTION WHAT IS YOUR FUNCTION? STAGE 5 AUTOMATION
================================================================================
- CONFIRMED: 23 G4 IF = 144 MHz -> --rf 144e6; PTT = active-low (ground to key); SEQ 4
  delay 100..150 ms (so schedule guard >= ~200 ms).
- Decide if B210 GPIO PIN 0 is what we will use for PTT (I think we need a transistor/opto to pull
  the PTT line to ground. The GPIO is 3.3 V logic, the PTT expects a contact closure).
We need to add GPIO PTT to eve_b210.py's run path, matched to our SEQ 4, for the first
automated Moon transmission. Then we have one line operation from the laptop. 
