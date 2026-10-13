// ============================================================
// MOON ENCLOSURE — K3NG rotator controller (Hello Giggy / ORI EVE)
// Arduino Mega 2560 + Seeed Relay Shield v3.0 + 20x4 I2C LCD + Wago junctions
//
// Two-part design:
//   base_plate() — flat mounting deck: Mega standoffs, Wago bay, zip anchors
//   moon_dome()  — cratered hemisphere lid with front facet (LCD mare)
//                  and rear facet (DIN / USB-B / barrel jack plain)
//
// Render one part at a time via the PART selector below.
// All dims mm. Values marked VERIFY need calipers before printing.
// Philosophy: print the test coupon FIRST (PART = "coupon") to check
// the Mega hole pattern and LCD window on a 20-minute print before
// committing to the multi-hour Moon.
// ============================================================

PART = "dome"; // "base", "dome", "coupon", "check", "deck", "assembly"

$fn = 96;

// ---------------- [ Body ] ----------------
moon_d        = 175;   // outer diameter of the lunar dome
wall          = 3.0;   // dome shell thickness
base_t        = 4.0;   // base plate thickness
base_lip_h    = 6;     // vertical skirt on base that the dome rim captures
fit_gap       = 0.25;  // clearance between dome rim and base lip (tune to printer)

// ---------------- [ Arduino Mega 2560 ] ----------------
mega_l        = 101.6;
mega_w        = 53.3;
// Hole coords from board corner (USB corner = origin). VERIFY with calipers
// or by printing the coupon — published values, not yet measured on this unit.
mega_holes    = [ [14.0,  2.5], [15.3, 50.7], [66.1, 35.5],
                  [66.1,  7.6], [96.6,  2.5], [90.2, 50.7] ];
standoff_h    = 7;     // clears solder points under the Mega
standoff_d    = 7;
screw_d       = 2.7;   // M3 self-tapping into plastic
mega_y_shift  = 9;     // aft shift away from the LCD facet: buys the
                       // backpack ~2.5mm clearance over the stack's
                       // front-top corner at lcd_lift=22. Aft headroom
                       // runs out ~17 (west-rear-top corner vs inner
                       // sphere at stack_h) — stay under that.
mega_pos      = [-mega_l/2 - 8, -mega_w/2 + mega_y_shift];
                       // Mega placement on deck (west, shifted aft;
                       // Wago bay east)

// ---------------- [ Stack height ] ----------------
stack_h       = 42;    // top of relay bodies above deck. VERIFY: standoff_h
                       // + Mega + header gap + shield + relay height.

// ---------------- [ 20x4 LCD module ] ----------------
// VERIFIED against Newhaven NHD-0420DZ mechanical drawing (industry-
// standard 2004 stack) + measured on the physical unit:
lcd_board     = [98.0, 60.0];  // PCB outline, ±0.5
lcd_bezel     = [96.8, 39.3];  // black metal frame, centered on PCB,
                               // face stands 9.0mm proud of PCB — it
                               // passes THROUGH the panel (wall=3mm)
lcd_va        = [77.0, 25.2];  // viewing area inside the bezel (ref)
lcd_hole_pitch= [93.0, 55.0];  // 4x Ø2.5 holes, pattern centered
lcd_screw_d   = 2.7;
lcd_win_clear = 0.4;           // per-side clearance around the bezel
lcd_win       = [lcd_bezel[0] + 2*lcd_win_clear,
                 lcd_bezel[1] + 2*lcd_win_clear];  // panel cut
lcd_tilt      = 15;            // facet tilt back from vertical, degrees —
                               // desk box read from above; adjust to taste
lcd_facet_d   = 49;            // facet plane distance from dome center.
                               // Lens radius sqrt(moon_r^2 - d^2):
                               // 49 on a 175 dome gives ~72.5mm.
lcd_lift      = 22;            // slide LCD up along the facet plane.
                               // At d=49/lift=22 on the 175 dome: top
                               // screws ~4.6mm inside the lens, bottom
                               // screws z=+7.4, board bottom clears deck.
                               // Backpack-vs-stack clearance comes from
                               // the mega_pos aft shift — re-verify with
                               // PART="check" after any mega_pos change.
lcd_pcb_t     = 1.6;           // module PCB thickness
lcd_solder_ko = 4;             // solder-side keepout behind PCB. VERIFY
bp_env        = [45, 20, 14];  // I2C backpack envelope (w x h x depth)
                               // riding the top edge of the PCB rear. VERIFY

// ---------------- [ Rear connector plain ] ----------------
din_hole_d    = 13.0;  // cable pass-through for DIN pigtail; upsize to ~12.5
                       // gland thread if using a PG7 cable gland. VERIFY.
usb_b_cut     = [14, 13];      // USB-B plug body clearance
barrel_hole_d = 11.5;  // panel-mount 2.1mm barrel jack bushing. VERIFY jack.
rear_port_z   = 22;    // port centerline height above deck

// ---------------- [ Wago bay ] ----------------
// 221-413 (3-port) and 221-415 (5-port) nominal envelopes. VERIFY measured.
wago3         = [13.0, 18.6, 8.4];   // w x l x h
wago5         = [20.4, 18.6, 8.4];
wago_fence_h  = 6;
wago_fence_t  = 1.6;
wago_slots    = 4;     // 3-port cradle count (AZ-FB, EL-FB, SDA, SCL)
wago5_slots   = 2;     // 5-port cradle count (GND junction, 5V junction)
ziptie_slot   = [4, 2];        // zip tie anchor slot cross-section

// ---------------- [ Craters ] ----------------
// [azimuth deg, elevation deg, radius mm] on the dome surface.
// Fixed list, not random: reproducible Moon. Sculpt at will.
craters = [ [ 20, 55, 14], [ 75, 30, 9], [130, 60, 11], [170, 25, 7],
            [215, 45, 16], [260, 65, 8], [300, 35, 12], [340, 70, 6],
            [ 55, 75, 5],  [110, 20, 6], [245, 20, 9],  [305, 78, 7] ];
crater_depth_ratio = 0.35;  // how deep each crater bites, x its radius

// ============================================================
// Derived
// ============================================================
moon_r  = moon_d/2;
dome_r  = moon_r;            // dome sphere radius
rear_facet_y  =  moon_r - 14; // plane position of the rear connector facet

// ============================================================
// Utility
// ============================================================
module m3_boss(h) {
    difference() {
        cylinder(d=standoff_d, h=h);
        translate([0,0,1]) cylinder(d=screw_d, h=h);
    }
}

module ziptie_anchor() {
    // bridge saddle (a STUB, not holes): the zip tie threads through the
    // tunnel under the bridge and cinches the harness down onto its top.
    difference() {
        cube([12, 8, 6], center=true);
        translate([0, 0, -1.5])
            cube([ziptie_slot[0]+2, 10, ziptie_slot[1]+1.2], center=true); // tunnel
    }
}

module wago_cradle(env, n) {
    // parallel pockets with fences; friction fit, lever side up
    for (i = [0:n-1])
        translate([i*(env[0]+wago_fence_t), 0, 0])
        difference() {
            cube([env[0]+2*wago_fence_t, env[1]+2*wago_fence_t, wago_fence_h]);
            translate([wago_fence_t, wago_fence_t, -1])
                cube([env[0], env[1], wago_fence_h+2]);
        }
}

// ============================================================
// BASE PLATE
// ============================================================
module base_plate() {
    difference() {
        union() {
            cylinder(r=moon_r - wall - fit_gap, h=base_t);
            // capture lip
            translate([0,0,base_t])
                difference() {
                    cylinder(r=moon_r - wall - fit_gap, h=base_lip_h);
                    translate([0,0,-1])
                        cylinder(r=moon_r - wall - fit_gap - 2, h=base_lip_h+2);
                }
            // Mega standoffs
            for (h = mega_holes)
                translate([mega_pos[0]+h[0], mega_pos[1]+h[1], base_t])
                    m3_boss(standoff_h);
            // Wago junctions: mounted loose (hot glue if needed) — the
            // wago_cradle() module survives below if cradles return
            // harness anchors, relocated to free parcels (SE + east
            // crescent) after the mega aft shift consumed the north strip
            translate([55, -30, base_t+3]) ziptie_anchor();
            translate([60,  18, base_t+3]) ziptie_anchor();
        }
        // ventilation: slots in the corridor between the standoff rows
        // (board-y 11..32), positioned RELATIVE to mega_pos so the vents
        // travel with the board (W5NYV fix, made shift-proof)
        for (i = [-2:2])
            translate([mega_pos[0]+30 + i*8, mega_pos[1]+12.5, -1])
                cube([3, 18, base_t+2], center=false);
    }
}

// ============================================================
// MOON DOME
// ============================================================
module dome_shell() {
    difference() {
        sphere(r=dome_r);
        sphere(r=dome_r - wall);
        translate([0,0,-dome_r]) cube(2*dome_r, center=true); // keep upper half
    }
}

module craters_cut() {
    for (c = craters)
        rotate([0,0,c[0]]) rotate([0, 90-c[1], 0])
            translate([0,0, dome_r + c[2]*(1-crater_depth_ratio)])
                sphere(r=c[2]);
}

// ---- facet frames: local z axis = facet outward normal, origin on plane ----
// A facet on a HOLLOW shell needs two operations, not one:
//   1. shear off the spherical cap outside the plane (leaves an opening)
//   2. weld a wall-thick lens-shaped plate into that opening, flush
// Windows/ports/screws are then drilled through the plate along local z.

module front_frame() {
    // origin = (depth along facet normal) + (lift along the in-plane up axis)
    translate([0,
               -lcd_facet_d*cos(lcd_tilt) + lcd_lift*sin(lcd_tilt),
                lcd_facet_d*sin(lcd_tilt) + lcd_lift*cos(lcd_tilt)])
        rotate([90 - lcd_tilt, 0, 0]) children();
}

module rear_frame() {
    translate([0, rear_facet_y, rear_port_z])
        rotate([-90, 0, 0]) children();
}

module facet_shear(frame) {
    // everything outside the facet plane (local z > 0)
    if (frame == "front") front_frame() translate([0,0,200]) cube(400, center=true);
    if (frame == "rear")  rear_frame()  translate([0,0,200]) cube(400, center=true);
}

module facet_plate(frame) {
    // wall-thick slab behind the plane, trimmed to the sphere: a flat lens
    intersection() {
        sphere(r=dome_r - 0.01);
        if (frame == "front")
            front_frame() translate([0,0,-wall/2]) cube([260,260,wall], center=true);
        if (frame == "rear")
            rear_frame()  translate([0,0,-wall/2]) cube([260,260,wall], center=true);
    }
}

module moon_dome() {
    difference() {
        union() {
            difference() {
                dome_shell();
                craters_cut();
                facet_shear("front");
                facet_shear("rear");
            }
            facet_plate("front");
            //facet_plate("rear");
        }

        // ---- LCD window + screws, drilled through the front plate ----
        front_frame() {
            cube([lcd_win[0], lcd_win[1], 8*wall], center=true);
            for (sx=[-1,1], sy=[-1,1])
                translate([sx*lcd_hole_pitch[0]/2, sy*lcd_hole_pitch[1]/2, 0])
                    cylinder(d=lcd_screw_d, h=8*wall, center=true);
        }

        // ---- rear ports, drilled through the rear plate ----
        rear_frame() {
            translate([-25, 0, 0]) cylinder(d=din_hole_d, h=8*wall, center=true);
            cube([usb_b_cut[0], usb_b_cut[1], 8*wall], center=true);
            translate([ 25, 0, 0]) cylinder(d=barrel_hole_d, h=8*wall, center=true);
        }

        // ---- FINAL operation: guillotine the southern hemisphere ----
        // (facet plates are full lenses of the sphere; this trims their
        // lower halves flush with the deck plane)
        translate([0, 0, -500]) cube(1000, center=true);
    }
}

// ============================================================
// TEST COUPON — print this first (~20 min, 2mm thick)
// Checks: Mega hole pattern, LCD window + screw pitch, port diameters
// ============================================================
module coupon() {
    difference() {
        cube([120, 75, 2]);
        for (h = mega_holes)
            translate([h[0]+8, h[1]+8, -1]) cylinder(d=screw_d, h=4);
        translate([60, 55, -1]) {
            cube([lcd_win[0]/2, lcd_win[1]/2, 4], center=false); // half window
        }
        translate([15, 60, -1]) cylinder(d=din_hole_d, h=4);
        translate([32, 60, -1]) cylinder(d=barrel_hole_d, h=4);
    }
}

// ============================================================
// INTERFERENCE CHECK — PART = "check"
// Models the LCD module's full envelope (bezel + PCB + solder
// keepout + I2C backpack) and the Mega/shield stack as boxes,
// all in the dome frame. Ghost outlines show the players; any
// SOLID RED geometry is a collision. Empty red = clear to print.
// ============================================================
module lcd_module_envelope() {
    front_frame() {
        // bezel: from PCB front face, 9.0mm forward, through the panel
        translate([0, 0, -wall + 4.5])
            cube([lcd_bezel[0], lcd_bezel[1], 9], center=true);
        // PCB flush against the back of the facet plate
        translate([0, 0, -wall - lcd_pcb_t/2])
            cube([lcd_board[0], lcd_board[1], lcd_pcb_t], center=true);
        // solder-side keepout across the whole board rear
        translate([0, 0, -wall - lcd_pcb_t - lcd_solder_ko/2])
            cube([lcd_board[0], lcd_board[1], lcd_solder_ko], center=true);
        // I2C backpack envelope along the top edge of the rear
        translate([0, lcd_board[1]/2 - bp_env[1]/2 - 2,
                   -wall - lcd_pcb_t - bp_env[2]/2])
            cube([bp_env[0], bp_env[1], bp_env[2]], center=true);
    }
}

module stack_envelope() {
    // Mega + shield + relays as a conservative full-footprint box
    translate([mega_pos[0], mega_pos[1], 0])
        cube([mega_l, mega_w, stack_h]);
}

module deck_envelope() {
    // the base plate plane and everything below it
    translate([0, 0, -10]) cylinder(r=moon_r, h=10);
}

// ============================================================
// PART selector
// ============================================================
if (PART == "base")      base_plate();
if (PART == "dome")      moon_dome();
if (PART == "coupon")    coupon();
if (PART == "deck") {
    // furniture layout view: deck real estate with the fixed occupants
    // ghosted in — use this to re-zone Wago cradles and saddles
    base_plate();
    translate([0,0,base_t]) {
        %stack_envelope();
        %lcd_module_envelope();
    }
}
if (PART == "check") {
    color("red")
        intersection() {
            lcd_module_envelope();
            union() { stack_envelope(); deck_envelope(); }
        }
    %stack_envelope();
    %lcd_module_envelope();
}
if (PART == "assembly") {
    base_plate();
    translate([0,0,base_t]) {
        %stack_envelope();
        %lcd_module_envelope();
        color("lightgray", 0.55) moon_dome();
    }
}
