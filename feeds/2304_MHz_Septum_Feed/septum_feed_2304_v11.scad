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


// ============================================================
// FIXED OK1DFC SEPTUM TEMPLATE WITH DRILL MARKERS (SOLID RECTANGLE)
// ============================================================
module printable_septum_with_drill_markers() {
    translate([-30, 0]) square(); // 20mm Calibration Check
    
    difference() {
        union() {
            // THE COMPLETE INTERNAL STEPPED STAIRCASE (Fully Closed Symmetrical Loop)
            polygon(points=[
                [15,       sF],    // 1. Step 1 starts at Z=15 (mouth), height 5.5mm
                [15 + sA,  sF],    // 2. Flat segment of Step 1
                [15 + sA,  sG],    // 3. Rises to Step 2 (height 13.8mm)
                [15 + sB,  sG],    // 4. Flat segment of Step 2
                [15 + sB,  sH],    // 5. Rises to Step 3 (height 23.5mm)
                [15 + sC,  sH],    // 6. Flat segment of Step 3
                [15 + sC,  sI],    // 7. Rises to Step 4 (height 33.3mm)
                [15 + sD,  sI],    // 8. Flat segment of Step 4
                [15 + sD,  sJ],    // 9. Rises to Step 5 (height 43.1mm)
                [15 + sE,  sJ],    // 10. Flat segment of Step 5
                [15 + sE,  wg_id], // 11. Rises vertically up to full waveguide height (73.0mm)
                [354,      wg_id], // 12. Runs perfectly flat to the back wall at Z=354mm (No trailing comma!),     
                [354,      0],     // 13. Drops straight down the back short wall to the floor (Y=0)
                [15,       0]      // 14. Returns completely straight along the floor to the front mouth
            ]);
            
            // THE TWO EXTERNAL FLANGE CLAMPING BORDERS
            translate([15, -15])   square([324, 15]);   // Bottom screw margin
            translate([15, wg_id])  square([324, 15]);   // Top screw margin
        }
        
        // Symmetrical Screw Hole Crosshairs spaced every 15mm 
        for (z = [40 : 15 : 334]) {
            translate([z, -7.5])       circle(d=1.0, $fn=16); // Bottom flange holes
            translate([z, wg_id + 7.5]) circle(d=1.0, $fn=16); // Top flange holes
        }
    }
}

module printable_symmetrical_clamshell_template_v4() {
    translate([-30, 0]) square(); // 20mm Calibration Check
    
    // Sheet size is locked perfectly to 354mm long by 173mm wide
    difference() {
        square([354, 173]); 
        
        // FRONT NOTCHES (Z = 0 to 15) - Clears lips for front flare collar
        translate([0, 0])        square([15, 15]);
        translate([0, 173 - 15]) square([15, 15]);
        
        // REAR NOTCHES (Z = 339 to 354) - Clears lips for backshort cap collar
        translate([339, 0])        square([15, 15]);
        translate([339, 173 - 15]) square([15, 15]);
        
        // SMA Connector Drill Hole Center
        // Placed 29.3mm from back wall (354 - 15 - 29.3 = 309.7mm)
        translate([309.7, 86.5]) circle(d=sma_hole_d, $fn=32);
            
        // Flange Screw Holes (Synchronized to match the shortened septum ears)
        for (z = [40 : 15 : 334]) {
            translate([z, 7.5])       circle(d=1.0, $fn=16); 
            translate([z, 173 - 7.5]) circle(d=1.0, $fn=16);
        }
    }
    
    // Scribe Alignment Reference Marks (Fold Indicator Lines)
    color("black") translate([15, 15])         square([324, 0.5]); // Bend 1 (15mm)
    color("black") translate([0, 15 + 35])     square([354, 0.5]); // Bend 2 (50mm)
    color("black") translate([0, 15 + 35 + 73]) square([354, 0.5]); // Bend 3 (123mm)
    color("black") translate([15, 15 + 35 + 73 + 35]) square([324, 0.5]); // Bend 4 (158mm)
}

// ============================================================
// CORRECTED STRAIGHT-EDGE FLARE PANEL (100% Symmetrical)
// ============================================================
module printable_final_flare_panel() {
    translate([-30, 0]) square(); // 20mm Calibration Gauge
    
    slant_height = flare_len / cos(flare_ang); // ~210.9 mm long
    w_small      = wg_od;                     // 75.0 mm wide throat
    w_large      = flare_od;                   // 184.2 mm wide mouth
    
    difference() {
        union() {
            // Main face of the flare (Centered on Y-axis)
            polygon(points=[
                [0,            -w_small/2], 
                [0,             w_small/2], 
                [slant_height,  w_large/2], 
                [slant_height, -w_large/2]
            ]);
            
            // THE MOUNTING COLLAR TAB (Correctly Centered on Y-Axis)
            // Moves -15mm back on X, and centers perfectly from -37.5mm to +37.5mm on Y
            translate([-15, -w_small/2]) 
                square([15, w_small]);
        }
        
        // Simple mounting screw holes centered in the tab zone
        translate([-7.5, -w_small/2 + 15]) circle(d=1.0, $fn=16);
        translate([-7.5,  w_small/2 - 15]) circle(d=1.0, $fn=16);
    }
    
    // Scribe bend indicator line (A perfect straight line across the tab)
    color("red") translate([0, -w_small/2]) square([0.2, w_small]);
}

module printable_flare_corner_brackets() {
    translate([-30, 0]) square([20, 20]);
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
    translate([-30, 0]) square([20, 20]);
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
// MASTER PRODUCTION OUTPUT CONTROLLER
// ============================================================
// For normal 3D verification on screen, leave these 3 lines uncommented:
//body_assembly();
//septum_plate();
//flare_assembly();

// TO EXPORT A 1:1 PDF TEMPLATE: 
// 1. Comment out the 3 lines above using //
// 2. Uncomment ONLY ONE line below at a time
// 3. Press F6 (Render), then go to File > Export > Export as PDF...

// printable_septum_with_drill_markers();         // Print 3-4 copies
 printable_symmetrical_clamshell_template_v4(); // Print 2 copies
// printable_final_flare_panel();                 // Print 4 copies
// printable_flare_corner_brackets();             // Print 1 copy
// printable_backshort_cap_template();            // Print 1 copy
       
