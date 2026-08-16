// OK1DFC Stepped Septum Feed for 2304 MHz -- v12 VERIFIED PARAMETER BUILD
// CONSTRUCTION: Sheet metal hat-channel clamshell, bending brake
// MATERIAL: 1mm (0.040") 3003-H14 Aluminum, Industrial Metal Supply, San Diego CA
//
// ============================================================
// ELECTROMAGNETIC PARAMETER PROVENANCE (v12 correction)
// ============================================================
// v11 wg_id=73mm and 5 equal 23.8mm steps came from an unverified source
// and are WRONG (0.561lambda guide, cutoff 2053 MHz; septum never reaches
// full height -> no circular polarization). v12 parameters are the OK1DFC
// spreadsheet 1296 MHz dimension set, frequency-scaled by exactly 1296/2304.
//
// SOURCES:
//  [1] OK1DFC spreadsheet output, 1296 MHz:
//      http://www.ok1dfc.com/EME/Technic/septum/23.pdf
//      Guide 144.9 | teeth A-E 78.2/138.2/200.9/222.5/370.4 from tip
//      heights F-J 18.5/41.2/69.7/113.7/144.9 | probe L=44, back M=43
//      total feed length 602 (=2.6 lambda) | septum thickness 0.008 lambda
//  [2] W1GHZ, "Septum Polarizers and Feeds" (HFSS verification of [1]):
//      http://www.w1ghz.org/antbook/conf/SEPTUM.pdf
//      Septum only valid at the Chen&Tsandoulas guide size; scale ALL dims.
//      Flare variant: 1.4 lambda square aperture, 15 deg half-angle,
//      best f/D ~0.7-0.85 (offset dish) -- retained from v8/v11.
//  [3] DL4MUP construction notes (independent as-built check):
//      https://www.qsl.net/dl4mup/Septum/RWST.htm
//      23cm internal 144.9mm; 13cm internal 82mm.
//  [4] W1GHZ, "Septum Feeds - Tolerances and Sensitivity" (2012):
//      dimension sensitivity is low; +/-1mm class errors are benign at 2304.
//
// KNOWN AMBIGUITIES (flagged, both benign per [4]):
//  * Guide ratio: spreadsheet 0.626 lambda vs W1GHZ prose "0.635 lambda"
//    (81.5 vs 82.6 at 2304). Using 0.626 -- the HFSS-verified set.
//  * Tooth boundary C: printout 200.9mm (0.868 lambda) vs ratio column
//    0.86 (199.1mm). Using 200.9 (the mm value builders used). ~1mm here.

// ============================================================
// MASTER PARAMETERS -- 1296 MHz GROUND TRUTH + SCALE
// (domain model: the verified 1296 set is the source of truth;
//  everything at 2304 is derived. Rescale by changing freq only.)
// ============================================================
freq      = 2304.0;                 // MHz
f_ref     = 1296.0;                 // MHz, reference design frequency
k         = f_ref / freq;           // 0.5625 exactly
lambda    = 299792.458 / freq;      // 130.118 mm

// --- Verified 1296 MHz dimensions [1] ---
wg_id_ref        = 144.9;                              // guide inside, square
tooth_z_ref      = [78.2, 138.2, 200.9, 222.5, 370.4]; // boundaries from tip
tooth_h_ref      = [18.5, 41.2, 69.7, 113.7, 144.9];   // heights from wall
probe_len_ref    = 44.0;                               // monopole length
probe_back_ref   = 43.0;                               // center to rear wall
probe_dia_ref    = 6.0;                                // monopole diameter
feed_total_ref   = 602.0;                              // aperture to rear wall
septum_t_ref     = 1.852;                              // 0.008 lambda

// --- Scaled to 2304 MHz ---
wg_id      = wg_id_ref * k;                 // 81.51
tooth_z    = [for (z = tooth_z_ref) z * k]; // 43.99 77.74 113.01 125.16 208.35
tooth_h    = [for (h = tooth_h_ref) h * k]; // 10.41 23.18 39.21 63.96 81.51
septum_len = tooth_z[4];                    // 208.35 -- septum tip to rear wall
probe_len  = probe_len_ref * k;             // 24.75 (cut long, trim to match)
probe_back = probe_back_ref * k;            // 24.19 from inside of rear wall
feed_spec  = feed_total_ref * k;            // 338.63 spec cavity length
// Septum thickness ideal = 1.042mm (0.008*lambda, scaled from OK1DFC
// 1.852 @ 23cm); 0.040" stock (-2.5%) is within tolerance [4].
// DO NOT substitute thick stock (0.125"=3.175mm would be 3x design --
// septum thickness is an RF dimension of the verified set, 0.008 lambda).

t          = 1.016;                  // wall sheet: 0.040" 3003-H14
sept_t     = 1.016;                  // septum: cut from same 0.040" stock
wg_od      = wg_id + 2*t;            // outside dim (both axes -- see below)

// --- Tube length (build convenience, anchored at the REAR wall) ---
// Sheet is 354 long: 15mm front collar-overlap zone + 339 cavity carried
// over from v11 (339 vs spec 338.6: +0.4mm, benign). Septum position is
// referenced to the REAR wall, so the front open-guide section becomes
// 354 - 208.35 = 145.65mm = 1.12 lambda, vs spec 1.00 lambda. Longer plain
// single-mode guide ahead of the septum is harmless, especially with the
// flare attached; all matching-critical dims (teeth, probe) are rear-anchored.
tube_len   = 354.0;                  // physical tube = flat sheet length
collar_z   = 15.0;                   // front & rear overlap zones
septum_tip = tube_len - septum_len;  // 145.65 from front edge
probe_z    = tube_len - probe_back;  // 329.81 from front edge

// --- Mode sanity (echoed below) ---
fc_te10 = 299792.458 / (2 * wg_id);         // 1839 MHz
fc_te11 = fc_te10 * sqrt(2);                // 2601 MHz -- single mode at 2304

// ============================================================
// PROBE / SMA
// ============================================================
// W1GHZ [2]: use a FAT probe instead of tuning screws (design intent: no
// tuning screws, dimensional precision). Scaled 6mm -> 3.38mm.
// K&S 1/8" (3.175mm) brass rod recommended. Cut 27mm, trim to match.
probe_dia   = 3.175;                 // K&S 1/8" brass rod
sma_hole_d  = 4.4;                   // #VERIFY center hole vs 132150 datasheet
// Amphenol RF 132150 is a 4-hole flange jack (not a bulkhead-nut part).
// #VERIFY flange pattern from the datasheet before laser release:
sma_flange_hole_d   = 2.3;           // #VERIFY (2-56 clearance)
sma_flange_pattern  = 8.64;          // #VERIFY hole-center square, 0.340"
sma_style   = "flange4";             // "flange4" | "bulkhead" (6.4 single hole)
sma_bulk_d  = 6.4;

// ============================================================
// SHEET-METAL BEND MODEL -- 3003-H14, t=1.0, air bend
// ============================================================
// ESTIMATES pending test coupon -- run printable_bend_test_coupon() first,
// measure, and adjust bend_R / k_factor. Formulas: Machinery's Handbook.
bend_R   = 1.0;    // inside radius. #MEASURE your brake nose radius
// --- COUPON CALIBRATION INPUT ---
// After bending the 80x30 coupon, enter the measured INSIDE width here
// and press F5. Leave at 0 to use the uncalibrated estimate below.
// Measure wall to wall, not flat part of the bottom. 

coupon_W = 0;      // mm, caliper inside jaws (e.g. 39.90). 0 = not yet run

k_factor = (coupon_W == 0)
    ? 0.42                                              // estimate: R/t=1, soft Al
    : ((40 + 2*bend_R - coupon_W)*(2/PI) - bend_R)/t;   // solved from coupon

// OK so what are we doing here?
// The construction condition ? value_A : value_B is the ternary conditional
// inherited from C. Read it aloud exactly as punctuated: "is coupon_W zero ?
// then 0.42 : otherwise, the formula." It's an expression, not a statement. 
// The whole thing evaluates to a single value, which then gets assigned.
    
    
function BA(theta) = (theta * PI/180) * (bend_R + k_factor * t);      // allowance
function BD(theta) = 2*(bend_R + t)*tan(theta/2) - BA(theta);          // deduction
BD90 = BD(90);     // 1.77
// Cross-check: v11's 173mm clamshell for wg=73 is reproduced by this BD:
// 2*15 + 2*(73/2+1) + (73+2) - 4*1.77 = 172.9. Methodology consistent.

// ============================================================
// CLAMSHELL FLAT GEOMETRY (each of 2 identical U-halves)
// ============================================================
// Formed section per half: flange 15 | half-wall (wg_id/2 + t outside)
// | floor (wg_od) | half-wall | flange 15. Four 90 deg bends.
// Flange outside dim 15 measured from wall OUTER surface to flange tip,
// so the flange tip lands flush with the 16mm septum ear edge and rail
// holes at 7.5-from-edge are coaxial by construction (v11 had a 1mm
// misalignment here: ears were 15, must be t+15).
flange_w   = 15.0;
// v12.1 FIX: the septum is SANDWICHED between the mated flanges, so it
// adds sept_t to the assembled interior across the mating axis. Each
// half channel must therefore be (wg_id - sept_t)/2 deep inside, so
// that (wg_id-sept_t)/2 + sept_t + (wg_id-sept_t)/2 = wg_id exactly.
// (Previous wg_id/2 depth would have built the guide sept_t oversize.)
// Note the outside stays square: across mating axis = wg_id + 2t = wg_od.
halfwall_o = (wg_id - sept_t)/2 + t; // outside dim
floor_o    = wg_od;                  // outside dim

b1 = flange_w - BD90/2;                        // 14.12
b2 = b1 + halfwall_o - BD90;                   // 54.10
b3 = b2 + floor_o - BD90;                      // 135.83
b4 = b3 + halfwall_o - BD90;                   // 175.82
clam_W = b4 + flange_w - BD90/2;               // 189.93
floor_ctr = (b2 + b3) / 2;                     // 94.97 == clam_W/2 (symmetric)

// ============================================================
// SHARED HOLE SCHEDULES (single source of truth -- item 2)
// ============================================================
hole_d      = 3.4;                   // M3 clearance, laser-cut final size
laser_kerf  = 0.0;                   // leave 0: IMS compensates in CAM
rail_edge_offset = 7.5;              // hole center from flange tip / ear edge

function rail_hole_zs()  = [for (z = [40 : 15 : 334]) z];   // 20 holes/rail
// Rear cap face-flange holes over the floors (deterministic: centerline
// symmetric; laser both parts):
cap_hole_z   = tube_len - 7.5;                 // 346.5
cap_hole_dy  = wg_id / 4;                      // +/-20.38 from floor center
// Front flare-collar holes over the floors:
collar_hole_z  = 7.5;
collar_hole_dy = wg_od/2 - flange_w;           // +/-26.75 from floor center
// NOTE: collar-tab-to-HALFWALL and cap-to-HALFWALL screws cross two bends
// of tolerance stack -> laser holes in the OUTER part only (tab / cap
// flange) and MATCH-DRILL the body through them at fit-up.

// ============================================================
// FLARE -- W1GHZ 1.4 lambda square pyramid, 15 deg half-angle [2]
// ============================================================
flare_id   = 1.4 * lambda;                     // 182.17
flare_od   = flare_id + 2*t;                   // 184.17
flare_ang  = 15;
flare_len  = (flare_id - wg_id) / (2*tan(flare_ang));   // 187.83 (was 203.7 @73)
slant      = flare_len / cos(flare_ang);                // 194.45
// Corner dihedral for a square pyramid with 15 deg face tilt:
// outward normals n1=(cosA,0,-sinA), n2=(0,cosA,-sinA); cos(angle)=sin^2 A
corner_normal_ang = acos(sin(flare_ang)*sin(flare_ang));  // 86.16
bracket_bend      = corner_normal_ang;   // bend 86.16 deg from flat
// (included angle between faces = 93.84 deg)
tab_bend          = flare_ang;           // collar tab: 15 deg bend, dihedral 165

// ============================================================
// BACKSHORT CAP -- wraps over tube exterior
// ============================================================
cap_fit_clr = 0.3;                   // slip fit over formed tube
cap_face    = wg_od + 2*cap_fit_clr; // 84.11 inside of formed cap

// ============================================================
// ECHO -- verification & cut list
// ============================================================
echo("==== VERIFIED EM PARAMETERS (2304 MHz, k=0.5625) ====");
echo(lambda=lambda, wg_id=wg_id, fc_te10=fc_te10, fc_te11=fc_te11);
echo(tooth_boundaries_from_tip=tooth_z);
echo(tooth_heights=tooth_h);
echo(septum_len=septum_len, septum_tip_from_front=septum_tip);
echo(probe_len=probe_len, probe_back=probe_back, probe_z=probe_z);
echo(feed_spec_len=feed_spec, tube_len=tube_len);
echo("==== BEND MODEL (ESTIMATE -- calibrate with coupon) ====");
echo(BA90=BA(90), BD90=BD90, bracket_bend_from_flat=bracket_bend);
echo(clamshell_bend_lines=[b1, b2, b3, b4], clamshell_flat_W=clam_W);
echo("==== FLARE ====");
echo(flare_id=flare_id, flare_len=flare_len, slant=slant);
echo("==== CUT LIST (bounding, mm) -- 0.040in (1.016mm) 3003-H14 ====");
echo("2x clamshell half", [tube_len, clam_W]);
echo("1x septum", [tube_len - collar_z, wg_id + 2*(t + flange_w)]);
echo("4x flare panel", [slant + flange_w, flare_od]);
echo("4x corner bracket", [slant, 25]);
echo("1x backshort cap", [cap_face + 2*flange_w, cap_face + 2*flange_w]);

// ============================================================
// BEND BUILD SHEET -- human-readable brake instructions.
// Recomputes from the CURRENT k_factor: after coupon calibration,
// update k_factor above, press F5, and read this table again.
// Registration convention (use IDENTICALLY on coupon and parts):
//   scribe line = center of the bend arc. Register the scribe to
//   your brake nose the same way every time, short leg protruding,
//   folding up. The coupon calibrates BD *for that habit*.
// ============================================================
function r1(x) = round(x * 100) / 100;   // 0.01 mm display rounding

echo("################ BEND BUILD SHEET ################");
echo(str("Material: t=", t, " mm 3003-H14 | assumed inside R=", bend_R,
         " | k=", k_factor));
echo(str("Per-90deg bend: allowance BA=", r1(BA(90)),
         " mm, deduction BD=", r1(BD90), " mm",
         (coupon_W == 0 ? "  [ESTIMATE until coupon run]" : "  [calibrated]")));
echo("--- STEP 0: COUPON (do this FIRST, 2x at 90deg grain) ---");
echo("Coupon flat: 80.00 x 30 mm; scribes at exactly 20.00 and 60.00.");
echo("Bend both up 90deg -> U-channel. Round numbers are on the CUT;");
echo(str("the calipers read the odd one: PREDICTED inside width = ",
         r1(40 - BA(90) + 2*bend_R), " mm at current k=", k_factor, "."));
echo(coupon_W == 0 ? "STATUS: UNCALIBRATED estimate -- set coupon_W after bend test" : str("STATUS: CALIBRATED from coupon_W = ", coupon_W, " mm"));
echo(str("CALIBRATE: just set coupon_W above and press F5. (Math: BA_actual = ",
         r1(40 + 2*bend_R), " - W;   k_factor = (BA_actual/(PI/2) - ",
         bend_R, ")/", t, " -- done for you.)"));
echo(str("Leg check: each outer leg ~", r1(20 + BD90/2), " mm outside."));
echo("--- CLAMSHELL HALF (make 2 identical) ---");
echo(str("Flat blank: ", r1(tube_len), " x ", r1(clam_W),
         " mm. All 4 bend lines parallel to the 354 mm edge,"));
echo("ACROSS the sheet rolling grain. All folds UP 90deg, same face.");
echo(str("Scribe from reference long edge:  B1=", r1(b1),
         "   B2=", r1(b2), "   B3=", r1(b3), "   B4=", r1(b4)));
echo("Bend order: B1, B4 (outer flanges first), then B2, B3.");
echo(str("CHECK after bending (outside dims): flange ", r1(flange_w),
         " | wall ", r1(halfwall_o), " | floor ", r1(wg_od),
         " | wall ", r1(halfwall_o), " | flange ", r1(flange_w)));
echo(str("CHECK inside channel: floor width ", r1(wg_id),
         " between walls, wall depth ", r1(halfwall_o - t),
         ". Two halves + ", sept_t, " mm septum between flanges -> ", r1(wg_id), " sq guide."));
echo("--- FLARE PANEL COLLAR TABS (4 panels) ---");
echo(str("15deg fold DOWN at ", r1(flange_w),
         " mm from aperture edge (loose tolerance; BD at 15deg = ",
         r1(BD(15)), " mm, already in pattern)."));
echo("--- CORNER BRACKETS (make 4) ---");
echo(str("Fold ", r1(bracket_bend), " deg along centerline of 25 mm strip",
         " (12.5 mm from either edge). Loose tolerance."));
echo(str("ASSEMBLY: panels have NO seam holes on purpose. Dry-fit ",
         "pyramid, clamp, verify mouth = ", r1(flare_id),
         " inside, MATCH-DRILL panels through bracket holes."));
echo(str("GAUGE: septum full-height section is cut to ", r1(wg_id),
         " mm -- it is your go/no-go gauge for the mated channel."));
echo("##################################################");

$fn = 64;

// ============================================================
// 3D VERIFICATION ENGINE
// (z=0 at tube front edge; rear wall inside face at z=tube_len)
// ============================================================
module wall_top()    { translate([-wg_od/2,  wg_id/2, 0]) cube([wg_od, t, tube_len]); }
module wall_bottom() { translate([-wg_od/2, -wg_id/2 - t, 0]) cube([wg_od, t, tube_len]); }

module wall_side(sgn) {   // sgn = +1 right (TX), -1 left (RX)
    difference() {
        translate([sgn > 0 ? wg_id/2 : -wg_id/2 - t, -wg_id/2, 0])
            cube([t, wg_id, tube_len]);
        translate([sgn * (wg_id/2 + t/2), 0, probe_z])
            rotate([0, 90, 0])
                cylinder(h = 3*t, d = sma_hole_d, center = true);
    }
    // probe render
    translate([sgn * wg_id/2, 0, probe_z])
        rotate([0, sgn * -90, 0])
            cylinder(h = probe_len, d = probe_dia);
}

module back_plate() { translate([-wg_od/2, -wg_od/2, tube_len]) cube([wg_od, wg_od, t]); }

module body_assembly() {
    wall_top(); wall_bottom(); wall_side(1); wall_side(-1); back_plate();
}

// Septum: plane x=0, teeth rise from bottom wall (y=-wg_id/2), tip set
// back 145.65 from the front edge. Step 5 IS the full-height section.
module septum_plate() {
    zs = [for (z = tooth_z) septum_tip + z];
    translate([sept_t/2, -wg_id/2, 0])
        rotate([0, -90, 0])
            linear_extrude(sept_t)
                polygon(points = [
                    [septum_tip, 0],
                    [septum_tip, tooth_h[0]],
                    [zs[0],      tooth_h[0]],
                    [zs[0],      tooth_h[1]],
                    [zs[1],      tooth_h[1]],
                    [zs[1],      tooth_h[2]],
                    [zs[2],      tooth_h[2]],
                    [zs[2],      tooth_h[3]],
                    [zs[3],      tooth_h[3]],
                    [zs[3],      tooth_h[4]],   // rises to full height 81.51
                    [zs[4],      tooth_h[4]],   // full height to rear wall
                    [zs[4],      0]
                ]);
}

module flare_assembly() {
    difference() {
        translate([0, 0, -flare_len])
            linear_extrude(flare_len, scale = [wg_od/flare_od, wg_od/flare_od])
                square([flare_od, flare_od], center = true);
        translate([0, 0, -flare_len - 0.1])
            linear_extrude(flare_len + 0.2, scale = [wg_id/flare_id, wg_id/flare_id])
                square([flare_id, flare_id], center = true);
    }
}

// ============================================================
// FLAT PATTERNS -- 2D, laser-ready
// ============================================================
module hole(d = hole_d) { circle(d = d + laser_kerf); }

// ---- SEPTUM ----
// y=0 at web bottom (bottom-wall inner surface). Ears are 16mm
// (flange 15 + wall pass-through t) so ear holes at 7.5-from-edge
// align with clamshell flange holes at 7.5-from-tip. (v12 fix)
module printable_septum() {
    ear = flange_w + t;   // 16.0
    difference() {
        union() {
            // bottom & top rails, full length (double as front spacers)
            translate([collar_z, -ear])  square([tube_len - collar_z, ear]);
            translate([collar_z, wg_id]) square([tube_len - collar_z, ear]);
            // stepped web (verified profile)
            zs = [for (z = tooth_z) septum_tip + z];
            polygon(points = [
                [septum_tip, 0],
                [septum_tip, tooth_h[0]], [zs[0], tooth_h[0]],
                [zs[0], tooth_h[1]],      [zs[1], tooth_h[1]],
                [zs[1], tooth_h[2]],      [zs[2], tooth_h[2]],
                [zs[2], tooth_h[3]],      [zs[3], tooth_h[3]],
                [zs[3], tooth_h[4]],      [zs[4], tooth_h[4]],
                [zs[4], 0]
            ]);
        }
        for (z = rail_hole_zs()) {
            translate([z, -ear + rail_edge_offset])        hole();
            translate([z,  wg_id + ear - rail_edge_offset]) hole();
        }
        // rear corner notches (v12.1 fix, caught by paper model): cut the
        // ears back to the wall OUTER surface over the last collar_z so
        // the backshort cap skirt slips over the tube. Leaves the t-thick
        // wall-slot sliver, flush with tube outside (cap_fit_clr clears it).
        // Mirrors the clamshell rear flange notches.
        translate([tube_len - collar_z, -ear])       square([collar_z, flange_w]);
        translate([tube_len - collar_z, wg_id + t])  square([collar_z, flange_w]);
    }
}

// ---- CLAMSHELL HALF (make 2 identical) ----
module printable_clamshell_half() {
    difference() {
        square([tube_len, clam_W]);
        // front notches: clear flare collar overlap on the flange rails
        translate([0, 0])                       square([collar_z, flange_w]);
        translate([0, clam_W - flange_w])       square([collar_z, flange_w]);
        // rear notches: clear backshort cap flange overlap
        translate([tube_len - collar_z, 0])                 square([collar_z, flange_w]);
        translate([tube_len - collar_z, clam_W - flange_w]) square([collar_z, flange_w]);
        // SMA on floor centerline (bend errors cancel by symmetry)
        if (sma_style == "flange4") {
            translate([probe_z, floor_ctr]) {
                hole(sma_hole_d);
                for (dx = [-1, 1], dy = [-1, 1])
                    translate([dx * sma_flange_pattern/2, dy * sma_flange_pattern/2])
                        hole(sma_flange_hole_d);
            }
        } else {
            translate([probe_z, floor_ctr]) hole(sma_bulk_d);
        }
        // rail holes (shared schedule -> coaxial with septum ears)
        for (z = rail_hole_zs()) {
            translate([z, rail_edge_offset])          hole();
            translate([z, clam_W - rail_edge_offset]) hole();
        }
        // rear cap holes over floor (laser both parts)
        for (dy = [-1, 1])
            translate([cap_hole_z, floor_ctr + dy * cap_hole_dy]) hole();
        // front collar holes over floor (laser both parts)
        for (dy = [-1, 1])
            translate([collar_hole_z, floor_ctr + dy * collar_hole_dy]) hole();
    }
    // scribe lines at the four bend centers (BD-corrected positions)
    for (b = [b1, b2, b3, b4])
        color("black") translate([collar_z, b]) square([tube_len - 2*collar_z, 0.3]);
}

// ---- FLARE PANEL (make 4) ----
module printable_flare_panel() {
    w_small = wg_od;      // 83.51 at throat
    w_large = flare_od;   // 184.17 at mouth
    difference() {
        union() {
            polygon(points = [
                [0,     -w_small/2], [0,      w_small/2],
                [slant,  w_large/2], [slant, -w_large/2]
            ]);
            translate([-flange_w, -w_small/2]) square([flange_w, w_small]); // collar tab
        }
        // tab holes: laser here, MATCH-DRILL body half-walls through these
        // for the two side panels; the two floor panels align with the
        // laser-cut collar holes in the clamshell floors.
        translate([-flange_w/2, -collar_hole_dy]) hole();
        translate([-flange_w/2,  collar_hole_dy]) hole();
    }
    // tab bend: 15 deg only (dihedral 165) -- BD(15) = negligible
    color("red") translate([0, -w_small/2]) square([0.3, w_small]);
}

// ---- CORNER BRACKETS (make 4) -- bend 86.16 deg from flat ----
// NOTE: flare panels have NO holes along their slant edges BY DESIGN.
// The bracket bend position, pyramid dihedral closure, and bracket
// placement along the seam all stack; pre-cut panel holes would demand
// they cancel. Instead the bracket holes are the drill jig: dry-fit the
// pyramid, clamp, verify mouth = flare_id inside, then MATCH-DRILL the
// panels through the bracket holes. (Same philosophy as collar/cap.)
module printable_flare_corner_brackets() {
    for (i = [0 : 3])
        translate([0, i * 35]) {
            difference() {
                square([slant, 25]);
                for (x = [25 : 50 : slant - 25]) {
                    translate([x,  6.25]) hole();
                    translate([x, 18.75]) hole();
                }
            }
            color("red") translate([0, 12.5]) square([slant, 0.3]);
        }
}

// ---- BACKSHORT CAP ----
module printable_backshort_cap() {
    f  = cap_face;                       // 84.11 formed inside
    fl = flange_w;
    d  = BD90;                           // one 90 bend per flange
    difference() {
        union() {
            translate([fl, fl])           square([f, f]);        // face
            translate([fl, fl - fl + 0])  square([f, fl]);       // -y flange
            translate([fl, fl + f])       square([f, fl]);       // +y flange
            translate([0,  fl])           square([fl, f]);       // -x flange
            translate([fl + f, fl])       square([fl, f]);       // +x flange
        }
        // 2 holes per flange at +/- wg_id/4 from face center,
        // rail_edge_offset from flange tip
        for (s = [-1, 1]) {
            // y flanges (land on floors -> matching laser holes in body)
            translate([fl + f/2 + s * cap_hole_dy, rail_edge_offset])          hole();
            translate([fl + f/2 + s * cap_hole_dy, 2*fl + f - rail_edge_offset]) hole();
            // x flanges (land on half-wall pairs -> MATCH-DRILL body)
            translate([rail_edge_offset,          fl + f/2 + s * cap_hole_dy]) hole();
            translate([2*fl + f - rail_edge_offset, fl + f/2 + s * cap_hole_dy]) hole();
        }
    }
    // bend scribes at face perimeter (bend line = face edge - d/2 inward
    // handled at the brake; scribe marks the mold line)
    for (p = [[fl, fl, f, 0.3], [fl, fl + f, f, 0.3]])
        color("red") translate([p[0], p[1]]) square([p[2], p[3]]);
    for (p = [[fl, fl], [fl + f, fl]])
        color("red") translate(p) square([0.3, f]);
}

// ---- BEND TEST COUPON -- run FIRST, calibrate bend_R / k_factor ----
// Bend at both scribes to 90. Target formed outside dims: 20 / 40 / 20.
// If the middle leg measures 40 + e, reduce k_factor by e/(2*BA(90)/1.42)
// or simply iterate; then re-export all patterns.
module printable_bend_test_coupon() {
    // ROUND-NUMBER COUPON (v12.2): blank 80 x 30, scribes at exactly
    // 20 and 60. Cut and mark to nice numbers; the ODD number is the
    // predicted inside width, read with calipers after bending:
    //   W_pred = 40 - BA(90) + 2*bend_R      (39.76 at current estimate)
    // Calibration from measured W:
    //   BA_actual = (40 + 2*bend_R) - W
    //   k_factor  = (BA_actual/(PI/2) - bend_R) / t
    // Leg check: each outer leg should measure 20 + BD/2 outside (~20.9).
    difference() { square([80, 30]); }
    color("red") translate([20, 0]) square([0.3, 30]);
    color("red") translate([60, 0]) square([0.3, 30]);
}

// ============================================================
// MASTER OUTPUT CONTROLLER
// ============================================================
// 3D check: uncomment the three lines below.
//body_assembly();
//septum_plate();
//flare_assembly();

// DXF EXPORT (IMS laser): uncomment ONE module, F6, File > Export > DXF.
// Quantities: coupon x1 (FIRST), clamshell x2, septum x1, flare panel x4,
// brackets x1 sheet, cap x1.
 printable_bend_test_coupon();
// printable_clamshell_half();
// printable_septum();
// printable_flare_panel();
// printable_flare_corner_brackets();
// printable_backshort_cap();
