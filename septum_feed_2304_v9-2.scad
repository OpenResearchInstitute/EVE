// OK1DFC Stepped Septum Feed for 2304 MHz — v9
// ORI / W5NYV — April 2026
//
// CONSTRUCTION: Sheet metal, 2mm 6061 aluminum throughout
// PIECES:
//   1. U-channel half A  — wide face + two flanges (bent from flat blank)
//   2. U-channel half B  — same, mirrors A
//   3. Septum plate      — stepped flat plate with flanges,
//                          captured between U-channel flanges when assembled
//   4. Back plate        — closes backshort end, flanged all four sides
//   + Flare              — aluminum flashing, square-to-circular, separate piece
//                          (KA1GT/KA1GT construction: rolled cylinder, one end deformed to square)
//
// ALL PROPORTIONAL CONSTANTS from Chen & Tsandoulas stepped septum paper
// Scaled via OK1DFC spreadsheet. Verified by W1GHZ HFSS simulation.
// Design proven at 1296, 2320, 3400 MHz by KI6DHU, KL6M, KA1GT, RA3AQ and others.
//
// ELECTRICAL SURFACES ONLY — flanges and bend allowances not modeled.
// This file is the ground truth for electrical dimensions.

// ============================================================
// PARAMETERS
// ============================================================
lambda    = 299792 / 2304;
t         = 2.0;             // sheet thickness — 2mm Al everywhere

wg_id     = 0.635 * lambda;  // inner dimension
// 0.635 is from the Chen & Tsandoulas paper on stepped septum polarizers
wg_od     = wg_id + 2*t;     // outer dimension

// step positions along the length of the septum
sA = 0.338 * lambda;
sB = 0.597 * lambda;
sC = 0.860 * lambda;
sD = 0.961 * lambda;

body_len = 1.600 * lambda;   // calculated minimum body length

// Body length needs to be long enough for the following things.
//1. Section before septum starts:    allow spurious modes to die out
//                                    so only TE10 reaches the septum.
//                                    This means we're single mode.
//2. The four stepped sections:       A through D = 0.961λ total
//                                    The A, B, C, D positions
//                                    create exactly 90° phase difference 
//                                    between the TE10 and TE01 modes 
//                                    at the design frequency.
//3. Full-height section after sD:    from sD to body_len = 0.639λ
//                                    allows modes to recombine cleanly.

//heights of each step of our septum profile
sF = 0.080 * lambda;
sG = 0.178 * lambda;
sH = 0.301 * lambda;
sI = 0.491 * lambda;
sJ = 0.626 * lambda;         

//The steps are carefully calculated to create exactly 90° of phase 
//difference between the two polarization components, which is 
//what produces circular polarization.


// Next, we get the probe position, measured from the back plate 
// toward the aperture. This is the distance that places the probe 
// at the correct location in the standing wave pattern inside 
// the waveguide for maximum good vibes. 

probe_z_from_end = 0.190 * lambda;

// SMA female bulkhead connector
// Huber+Suhner 21_SMA-50-2-15/111_NE (Paul Wade W1GHZ has multiples in stock)
// Straight bulkhead jack, solder attachment, 18 GHz, 50 ohm
// Designed for .086" semi-rigid cable (RG405)
// Center contact: solder cup, ~0.5-0.6mm hole for center conductor
// Body OD: 6.35mm (standard SMA)
// Panel hole: 6.4mm (6.35mm body + 0.05mm clearance)
// Nut: standard SMA hex nut, tightens on inside of wall
// Probe pin: ~0.5mm solid copper or brass wire, soldered into solder cup
//            total pin length = 24.1mm from inner wall face (M = 0.185*lambda)
//            cut wire to: 24.1mm + solder cup depth (~3mm) = ~27mm
sma_hole_d     = 6.4;    // panel hole (6.35mm body + 0.05mm clearance)
sma_body_d     = 6.35;   // SMA body OD (standard)
sma_nut_d      = 8.0;    // hex nut across flats (standard SMA)
sma_nut_t      = 3.0;    // hex nut thickness (protrudes this far into bore)

// Flare parameters — W1GHZ square pyramid flare
// From W1GHZ SEPTUM.pdf: 1.4λ square aperture, 15° half-angle
// specifically designed for offset dishes (our f/D = 0.83)
// "this size diagonal horn with linear polarization is a good feed for an offset dish"
flare_id  = 1.4 * lambda;    // 182.2mm inner aperture (square)
flare_ang = 15;              // half-angle degrees (W1GHZ)
flare_len = (flare_id - wg_id) / (2 * tan(flare_ang));  // 186mm
flare_od  = flare_id + 2*t;  // 186.2mm outer

// ============================================================
// BODY WALLS — four flat rectangular sheets
// ============================================================
module wall_top() {
    translate([-wg_od/2, wg_id/2, 0])
        cube([wg_od, t, body_len]);
}

module wall_bottom() {
    translate([-wg_od/2, -wg_id/2 - t, 0])
        cube([wg_od, t, body_len]);
}

module wall_right() {
    pz = body_len - probe_z_from_end;
    difference() {
        translate([wg_id/2, -wg_id/2, 0])
            cube([t, wg_id, body_len]);
        // TX probe hole — SMA bulkhead body passes through
        translate([wg_id/2 - 0.1, 0, pz])
            rotate([0, 90, 0])
                cylinder(h=t + 1, d=sma_hole_d, $fn=32);
    }
    // SMA bulkhead body on outside, nut on inside
    translate([wg_id/2 + t, 0, pz])
        rotate([0, 90, 0])
            sma_bulkhead();
    // Hex nut on inside of bore
    translate([wg_id/2, 0, pz])
        rotate([0, -90, 0])
            sma_nut();
}

module wall_left() {
    pz = body_len - probe_z_from_end;
    difference() {
        translate([-wg_id/2 - t, -wg_id/2, 0])
            cube([t, wg_id, body_len]);
        // RX probe hole
        translate([-wg_id/2 - t - 0.1, 0, pz])
            rotate([0, 90, 0])
                cylinder(h=t + 1, d=sma_hole_d, $fn=32);
    }
    // SMA bulkhead body on outside, nut on inside
    translate([-wg_id/2 - t, 0, pz])
        rotate([0, -90, 0])
            sma_bulkhead();
    // Hex nut on inside of bore
    translate([-wg_id/2, 0, pz])
        rotate([0, 90, 0])
            sma_nut();
}

module back_plate() {
    translate([-wg_od/2, -wg_od/2, body_len])
        cube([wg_od, wg_od, t]);
}

module body_assembly() {
    wall_top();
    wall_bottom();
    wall_right();
    wall_left();
    back_plate();
}

// ============================================================
// SEPTUM PLATE — stepped flat sheet
// Stands upright at Y=0 centerline, steps grow in +Y (Half A side)
// Step lengths run along Z (feed axis), thickness t in X
// Slides in from aperture (Z=0) before back plate is installed
// ============================================================
module septum_plate() {
    // Flat stepped plate — OK1DFC original values
    // Bottom edge rests on bore floor (Y = -wg_id/2)
    // Stepped top edge grows from sF (aperture) to sJ≈wg_id (backshort)
    // Polygon X = Z in world (feed axis), Polygon Y = plate height
    // Translate -wg_id/2 so bottom sits on bore floor wall
    translate([t/2, -wg_id/2, 0])
    rotate([0, -90, 0])
    linear_extrude(t) {
        polygon(points=[
            [0,        sF],
            [sA,       sF],
            [sA,       sG],
            [sB,       sG],
            [sB,       sH],
            [sC,       sH],
            [sC,       sI],
            [sD,       sI],
            [sD,       sJ],
            [body_len, sJ],
            [body_len, 0],
            [0,        0]
        ]);
    }
}


// ============================================================
// FLARE — square pyramid frustum (W1GHZ design for offset dishes)
// Aperture: 1.4λ square, half-angle: 15°
// From W1GHZ SEPTUM.pdf: designed specifically for offset dishes
// body end at Z=0, aperture end at Z=-flare_len
// ============================================================
module flare_outer() {
    // Wide end at aperture (Z=-flare_len), narrow end at body (Z=0)
    translate([0, 0, -flare_len])
    linear_extrude(flare_len, scale=[wg_od/flare_od, wg_od/flare_od])
        square([flare_od, flare_od], center=true);
}

module flare_bore() {
    translate([0, 0, -flare_len - 0.1])
    linear_extrude(flare_len + 0.2, scale=[wg_id/flare_id, wg_id/flare_id])
        square([flare_id, flare_id], center=true);
}

module flare_assembly() {
    difference() {
        flare_outer();
        flare_bore();
    }
}

// ============================================================
// SMA BULKHEAD CONNECTOR — body on outside, nut on inside
// KI6DHU approach: direct mount through wall, no backing plate
// Amphenol RF 132150 or similar SMA female bulkhead
// ============================================================
module sma_bulkhead() {
    // Connector body protruding outside wall
    cylinder(h=8, d=sma_body_d, $fn=32);
}

module sma_nut() {
    // Hex nut on inside of bore — protrudes ~3mm into bore
    // Small perturbation at 2304 MHz / 20W — acceptable
    cylinder(h=sma_nut_t, d=sma_nut_d, $fn=6);
}

// ============================================================
// RENDER — uncomment one section at a time
// ============================================================

// --- FULL ASSEMBLY ---
body_assembly();
septum_plate();
                      flare_assembly();

// --- INDIVIDUAL PARTS FOR EXPORT ---
// Body wall (all 4 identical rectangles):
//translate([0, 0, 0]) cube([body_len, wg_od, t]);

// Back plate:
//cube([wg_od, wg_od, t]);

// Septum plate (lying flat for printing):
// rotate([0, 0, 0]) septum_plate();

// Flare only:
// flare_assembly();

// ============================================================
echo("=== OK1DFC SEPTUM FEED 2304 MHz v7 — ORI / W5NYV ===");
echo("");
echo("--- SOURCE ---");
echo("All proportional constants: Chen & Tsandoulas stepped septum paper");
echo("Scaled by OK1DFC spreadsheet. Verified by W1GHZ HFSS simulation.");
echo("Built at 1296, 2320, 3400 MHz: KI6DHU KL6M KA1GT RA3AQ and others.");
echo("");
echo("--- FREQUENCY AND WAVELENGTH ---");
echo("Design frequency: 2304 MHz");
echo("c = 299792 km/s (precise)");
echo("lambda:", lambda, "mm");
echo("");
echo("--- WAVEGUIDE BODY ---");
echo("wg_id = 0.635*lambda:", wg_id, "mm  (fixed by Chen & Tsandoulas)");
echo("wg_od = wg_id + 2*t:", wg_od, "mm  (t=2mm sheet)");
echo("body_len = 1.600*lambda:", body_len, "mm  (minimum for transformer to work)");
echo("");
echo("--- SEPTUM STEP POSITIONS (from aperture, Chen & Tsandoulas) ---");
echo("sA = 0.338*lambda:", sA, "mm");
echo("sB = 0.597*lambda:", sB, "mm");
echo("sC = 0.860*lambda:", sC, "mm");
echo("sD = 0.961*lambda:", sD, "mm");
echo("");
echo("--- SEPTUM STEP HEIGHTS (from bore floor, Chen & Tsandoulas) ---");
echo("sF = 0.080*lambda:", sF, "mm  (narrowest, aperture end)");
echo("sG = 0.178*lambda:", sG, "mm");
echo("sH = 0.301*lambda:", sH, "mm");
echo("sI = 0.491*lambda:", sI, "mm");
echo("sJ = 0.626*lambda:", sJ, "mm  (full bore width, backshort end)");
echo("");
echo("--- PROBE (OK1DFC spreadsheet, from Chen & Tsandoulas) ---");
echo("L=0.190*lambda — probe Z from back plate:", probe_z_from_end, "mm");
echo("M=0.185*lambda — probe pin length (trim to this from inner wall):", 0.185*lambda, "mm");
echo("probe Z from aperture:", body_len - probe_z_from_end, "mm");
echo("");
echo("--- FLARE (W1GHZ square pyramid, designed for offset dishes) ---");
echo("Aperture: 1.4*lambda =", flare_id, "mm square inner");
echo("Half-angle:", flare_ang, "degrees");
echo("Flare length:", flare_len, "mm");
echo("Reference: W1GHZ SEPTUM.pdf — 1.4-lambda square aperture for offset dish f/D ~0.75-0.83");
echo("");
echo("--- FLAT BLANK CUT LIST (IMS San Diego, 2mm 6061 Al) ---");
echo("U-channel halves x2 (wide face + two flanges each):");
echo("  length:", body_len, "mm");
echo("  width: wg_od + 2*(wg_id/2 + t) =", wg_od + 2*(wg_id/2 + t), "mm");
echo("Septum plate x1 (with flanges, captured between U-channels):");
echo("  length:", body_len, "mm");
echo("  width: wg_id + 2*t =", wg_id + 2*t, "mm");
echo("Back plate x1 (flanged all four sides): approx", wg_od+20, "x", wg_od+20, "mm");
echo("");
echo("--- SMA CONNECTORS (confirmed in stock, W1GHZ Paul Wade) ---");
echo("Huber+Suhner 21_SMA-50-2-15/111_NE");
echo("Straight bulkhead jack, solder, 18 GHz, 50 ohm");
echo("Panel hole diameter:", sma_hole_d, "mm  (6.35mm body + 0.05mm clearance)");
echo("Nut protrusion into bore:", sma_nut_t, "mm");
echo("Probe pin: ~0.5mm solid wire, soldered into solder cup");
echo("Probe pin total length: 24.1 + 3mm cup depth = ~27mm");
echo("Probe extends into bore M=0.185*lambda:", 0.185*lambda, "mm from inner wall face");
