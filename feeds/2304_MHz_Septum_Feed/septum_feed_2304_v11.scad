// OK1DFC Stepped Septum Feed for 2304 MHz — FULL RESTORED BUILD
// CONSTRUCTION: Sheet metal, 1mm (0.040") 3003-H14 Aluminum 
// MATERIAL SOURCE: Industrial Metal Supply, San Diego, CA

// ============================================================
// PARAMETERS (Optimized for 2304 MHz EME)
// ============================================================
lambda    = 299792 / 2304;   // Wavelength: 130.118 mm
t         = 1.0;             // IMS 0.040" Sheet Thickness

// Single-mode Waveguide Dimensions (OK1DFC Ground Truth)
wg_id     = 73.0;            // Inside width (Prevents higher-order modes)
wg_od     = wg_id + 2*t;     // Outer width including 1mm walls (75.0 mm)
body_len  = 339.0;           // Functional RF internal cavity length

// Step Positions along Z-axis (Measured from RF Aperture at Z=0)
sA = 23.8;                   
sB = 23.8 * 2;               
sC = 23.8 * 3;               
sD = 23.8 * 4;               
sE = 23.8 * 5;               

// Septum Step Heights (Measured from Bore Floor up)
sF = 5.5;   // Step 1: closest to aperture
sG = 13.8;  // Step 2
sH = 23.5;  // Step 3
sI = 33.3;  // Step 4
sJ = 43.1;  // Step 5: tallest step near probes

probe_z_from_end = 29.3; 

// SMA Bulkhead Parameters (W1GHZ Specs)
sma_hole_d     = 6.4;    
sma_body_d     = 6.35;   
sma_nut_d      = 8.0;    
sma_nut_t      = 3.0;    

// W1GHZ Square Pyramid Flare for Offset Dishes (f/D ~ 0.8)
flare_id  = 1.4 * lambda;    // 182.16 mm inner square aperture
flare_ang = 15;              // Half-angle degrees
flare_len = (flare_id - wg_id) / (2 * tan(flare_ang));  // ~203.7 mm
flare_od  = flare_id + 2*t;  

// ============================================================
// 3D VERIFICATION ENGINE (Restored)
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
        // TX probe hole
        translate([wg_id/2 - 0.1, 0, pz])
            rotate([0, 90, 0])
                cylinder(h=t + 1, d=sma_hole_d, $fn=32);
    }
    // SMA Hardware Renders
    translate([wg_id/2 + t, 0, pz]) rotate([0, 90, 0]) sma_bulkhead();
    translate([wg_id/2, 0, pz]) rotate([0, -90, 0]) sma_nut();
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
    // SMA Hardware Renders
    translate([-wg_id/2 - t, 0, pz]) rotate([0, -90, 0]) sma_bulkhead();
    translate([-wg_id/2, 0, pz]) rotate([0, 90, 0]) sma_nut();
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

module septum_plate() {
    // Extruded at center X=0, running along Z axis from 0 to body_len
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
            [sE,       sJ],       
            [sE,       wg_id],    // Full height to act as shielding wall
            [body_len, wg_id],    // Runs to back wall to isolate ports
            [body_len, 0],        
        ]);
    }
}

module flare_outer() {
    translate([0, 0, -flare_len])
    linear_extrude(flare_len, scale=[wg_od/flare_od, wg_od/flare_od])
        square([flare_od, flare_od], center=true);
}

// Fixed subtracting shape to ensure manifold alignment
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

module sma_bulkhead() { cylinder(h=8, d=sma_body_d, $fn=32); }
module sma_nut() { cylinder(h=sma_nut_t, d=sma_nut_d, $fn=6); }

// ============================================================
// PRODUCTION PRINTABLE FLAT PATTERNS (Collision-Free Updates)
// ============================================================

module printable_septum_with_drill_markers() {
    translate([-30, 0]) square(); // 20mm Calibration Check
    difference() {
        union() {
            translate([15, 0]) {
                polygon(points=[
                    [0, sF], [sA, sF], [sA, sG], [sB, sG], [sB, sH], 
                    [sC, sH], [sD, sI], [sD, sJ], [sE, sJ], [sE, wg_id], 
                    [body_len, wg_id], [body_len, 0],
                ]);
            }
            translate([0, -15])   square([body_len + 15, 15]);   
            translate([0, wg_id]) square([body_len + 15, 15]);   
        }
        for (z = [40 : 15 : body_len + 5]) {
            translate([z, -7.5])       circle(d=1.0, $fn=16); 
            translate([z, wg_id + 7.5]) circle(d=1.0, $fn=16); 
        }
    }
    color("red") square([body_len + 15, 0.2]);
    color("red") translate([0, wg_id]) square([body_len + 15, 0.2]);
}

module printable_symmetrical_clamshell_template_v4() {
    translate([-30, 0]) square(); 
    difference() {
        square([body_len + 15, 171]); 
        translate([0, 0])        square([15, 15]);
        translate([0, 171 - 15]) square([15, 15]);
        
        pz = 15 + (body_len - probe_z_from_end); 
        translate([pz, 86.5]) circle(d=sma_hole_d, $fn=32);
            
        for (z = [40 : 15 : body_len + 5]) {
            translate([z, 7.5])       circle(d=1.0, $fn=16); 
            translate([z, 171 - 7.5]) circle(d=1.0, $fn=16);
        }
    }
    color("red") translate([15, 15])                 square([body_len, 0.2]);       
    color("red") translate([0, 15 + 35])              square([body_len + 15, 0.2]);  
    color("red") translate([0, 15 + 35 + 73])         square([body_len + 15, 0.2]);  
    color("red") translate([15, 15 + 35 + 73 + 35])   square([body_len, 0.2]);       
}

module printable_final_flare_panel() {
    translate([-30, 0]) square(); 
    slant_height = flare_len / cos(flare_ang); 
    w_small      = wg_od;                     
    w_large      = flare_od;                   
    
    difference() {
        union() {
            polygon(points=[[0, -w_small/2], [0, w_small/2], [slant_height, w_large/2], [slant_height, -w_large/2]]);
            translate([-15, -w_small/2]) square([15, w_small]);
        }
        translate([-7.5, -w_small/2 + 15]) circle(d=1.0, $fn=16);
        translate([-7.5,  w_small/2 - 15]) circle(d=1.0, $fn=16);
    }
    color("red") translate([0, -w_small/2]) square([0.2, w_small]);
}

module printable_flare_corner_brackets() {
    translate([-30, 0]) square();
    slant_height = flare_len / cos(flare_ang); 
    for (i = [0 : 3]) {
        translate([0, i * 35]) {
            difference() {
                square([slant_height, 25]); 
                for (x = [25 : 50 : slant_height - 25]) {
                    translate([x, 6.25])  circle(d=1.0, $fn=16); 
                    translate([x, 18.75]) circle(d=1.0, $fn=16); 
                }
            }
            color("red") translate([0, 12.5]) square([slant_height, 0.2]);
        }
    }
}

module printable_backshort_cap_template() {
    translate([-30, 0]) square();
    difference() {
        union() {
            translate([15, 15])  square([75, 75]);
            translate([15, 0])   square([75, 15]);  
            translate([15, 90])  square([75, 15]);  
            translate([0, 15])   square([15, 75]);  
            translate([90, 15])  square([15, 75]);  
        }
        translate([15 + 37.5, 7.5])      circle(d=1.0, $fn=16); 
        translate([15 + 37.5, 105 - 7.5]) circle(d=1.0, $fn=16); 
        translate([7.5,       15 + 37.5]) circle(d=1.0, $fn=16); 
        translate([105 - 7.5, 15 + 37.5]) circle(d=1.0, $fn=16); 
    }
    color("red") translate([15, 15]) square([75, 0.2]);
    color("red") translate([15, 90]) square([75, 0.2]);
    color("red") translate([15, 15]) square([0.2, 75]);
    color("red") translate([90, 15]) square([0.2, 75]);
}

// ============================================================
// EXECUTION SWITCH BOARD
// ============================================================
// For standard 3D screen view, keep the top block uncommented:

body_assembly();
septum_plate();
flare_assembly();

// To print templates, comment out the 3D block above and uncomment ONE below:
// printable_symmetrical_clamshell_template_v4(); 
// printable_final_flare_panel();                    
// printable_septum_with_drill_markers();       
// printable_flare_corner_brackets();           
// printable_backshort_cap_template();          
