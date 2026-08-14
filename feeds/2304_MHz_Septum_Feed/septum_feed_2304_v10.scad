// OK1DFC Stepped Septum Feed for 2304 MHz — v10 Optimized
// ORI / W5NYV & AI Collaborator — April 2026
//
// CONSTRUCTION: Sheet metal, 1mm (0.040") 3003-H14 Aluminum 
// MATERIAL SOURCE: Industrial Metal Supply, San Diego, CA
//
// ELECTRICAL SURFACES ONLY — flanges and bend allowances not modeled
// in this section. 

// ============================================================
// PARAMETERS (Optimized for 2304 MHz EME)
// ============================================================
lambda    = 299792 / 2304;   // Wavelength: 130.118 mm
t         = 1.0;             // IMS 0.040" Sheet Thickness

// Single-mode Waveguide Dimensions (OK1DFC Ground Truth)
wg_id     = 73.0;            // Inside width (Prevents higher-order modes)
wg_od     = wg_id + 2*t;     // Outer width including 1mm walls
body_len  = 339.0;           // Total optimized cavity length

// Step Positions along Z-axis (Measured from Open Aperture at Z=0)
// Each step is exactly 23.8 mm long, running sequentially
sA = 23.8;                   // End of Step 1
sB = 23.8 * 2;               // End of Step 2 (47.6 mm)
sC = 23.8 * 3;               // End of Step 3 (71.4 mm)
sD = 23.8 * 4;               // End of Step 4 (95.2 mm)
sE = 23.8 * 5;               // End of Step 5 (119.0 mm, Septum Ends)

// Septum Step Heights (Measured from Bore Floor up)
sF = 5.5;                    // Step 1: closest to aperture
sG = 13.8;                   // Step 2
sH = 23.5;                   // Step 3
sI = 33.3;                   // Step 4
sJ = 43.1;                   // Step 5: tallest step

// Probe Position (Measured from the Back Plate at Z=body_len)
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
// BODY WALLS — Four Flat Rectangular Sheets
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




// ============================================================
// REWRITTEN PRINTABLE TEMPLATE MODULES (COLLISION-FREE)
// Flanges and allowances included.
// ============================================================

// ============================================================
// FINAL OK1DFC SEPTUM TEMPLATE WITH DRILL MARKERS (FIXED 2D)
// ============================================================
module printable_septum_with_drill_markers() {
    // 1. 20mm Calibration Square (Verify your printer scale!)
    translate([-30, 0]) square([20, 20]);
    
    // 2. Main Sheet Geometry
    difference() {
        union() {
            // THE INTERNAL STEPPED STAIRCASE PROFILE (Pure 2D Grounded Ground-Plane)
            // Shifted exactly 15mm right to leave a clean front collar for the flare
            translate([15, 0]) {
                polygon(points=[
                    [0,        sF],       // Step 1: starts at Z=0 (mouth), height 5.5mm
                    [sA,       sF],
                    [sA,       sG],       // Step 2: height 13.8mm
                    [sB,       sG],
                    [sB,       sH],       // Step 3: height 23.5mm
                    [sC,       sH],
                    [sC,       sI],       // Step 4: height 33.3mm
                    [sD,       sI],
                    [sD,       sJ],       // Step 5: height 43.1mm (tallest step near probes)
                    [sE,       sJ],       
                    [sE,       wg_id],    // Full waveguide height (73.0mm)
                    [body_len, wg_id],    // Runs at full height all the way to the back wall
                    [body_len, 0],        // Drops down the back short wall to the floor
                    [0,        0]         // Returns back to the front mouth along the floor
                ]);
            }
            
            // THE TWO EXTERNAL FLANGE CLAMPING BORDERS (Where your hat channels sandwich together)
            // Total length matches the full 354mm length of the new clamshell collar (body_len + 15)
            translate([0, -15])   square([body_len + 15, 15]);   // Bottom screw margin
            translate([0, wg_id]) square([body_len + 15, 15]);   // Top screw margin
        }
        
        // Symmetrical Screw Hole Crosshairs spaced every 15mm 
        // Synchronized to match the clamshell template holes perfectly (starting at Z = 40mm)
        for (z = [40 : 15 : body_len + 5]) {
            translate([z, -7.5])       circle(d=1.0, $fn=16); // Bottom flange holes
            translate([z, wg_id + 7.5]) circle(d=1.0, $fn=16); // Top flange holes
        }
    }
    
    // 3. Scribe Alignment Reference Marks (Red Internal Guideline Boxes)
    // These lines match the inner roof and floor surfaces of your waveguide box
    color("red") square([body_len + 15, 0.2]);
    color("red") translate([0, wg_id]) square([body_len + 15, 0.2]);
}


module printable_symmetrical_clamshell_template_v3() {
    // 1. 20mm Calibration Square (Crucial Scale Check)
    translate([-30, 0]) square([20, 20]);
    
    // 2. Main Sheet Geometry with subtracted front notches
    // Total flat width remains 171mm. Total length expanded to 354mm to include 15mm collar.
    difference() {
        square([body_len + 15, 171]); 
        
        // FRONT COLLAR NOTCHES: Cut 15mm x 15mm away only from the vertical flanges
        translate([0, 0])        square([15, 15]);
        translate([0, 171 - 15]) square([15, 15]);
        
        // SMA Connector Drill Hole Center (Centered on the 73mm Side Wall section)
        pz = 15 + (body_len - probe_z_from_end); // 15 + 309.7 = 324.7mm
        translate([pz, 86.5])
            circle(d=sma_hole_d, $fn=32);
            
        // Screw Hole Crosshairs spaced every 15mm along the flanges
        for (z = [40 : 15 : body_len + 5]) {
            // Top Flange Hole Center (7.5mm from top edge)
            translate([z, 7.5]) circle(d=1.0, $fn=16); 
            // Bottom Flange Hole Center (7.5mm from bottom edge)
            translate([z, 171 - 7.5]) circle(d=1.0, $fn=16);
        }
    }
    
    // 3. Scribe Alignment Reference Marks (Red Bend Lines)
    color("red") translate([15, 15])                 square([body_len, 0.2]);       // Bend 1
    color("red") translate([0, 15 + 35])              square([body_len + 15, 0.2]);  // Bend 2
    color("red") translate([0, 15 + 35 + 73])         square([body_len + 15, 0.2]);  // Bend 3
    color("red") translate([15, 15 + 35 + 73 + 35])   square([body_len, 0.2]);       // Bend 4
}

// ============================================================
// SOLID STRAIGHT-EDGE FLARE PANEL (Print 4 Identical Copies)
// ============================================================
module printable_final_flare_panel() {
    // 1. 20mm Calibration Square
    translate([-30, 0]) square([20, 20]); 
    
    // Exact straight-line lengths for 2304 MHz
    slant_height = flare_len / cos(flare_ang); // ~210.9 mm long
    w_small      = wg_od;                     // 75.0 mm wide throat
    w_large      = flare_od;                   // 184.2 mm wide mouth
    
    // A completely clean, straight-edged trapezoid profile
    difference() {
        union() {
            // Main face of the flare
            polygon(points=[
                [0,            -w_small/2], 
                [0,             w_small/2], 
                [slant_height,  w_large/2], 
                [slant_height, -w_large/2]
            ]);
            
            // The 15mm attachment tab (Perfect 15mm x 75mm rectangle)
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


module printable_backshort_cap_template() {
    translate([-30, 0]) square([20, 20]);
    difference() {
        union() {
            translate([15, 15])  square([75, 75]);
            translate([15, 0])   square([75, 15]);  // Bottom tab
            translate([15, 90])  square([75, 15]);  // Top tab
            translate([0, 15])   square([15, 75]);  // Left tab
            translate([90, 15])  square([15, 75]);  // Right tab
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





// COMMENT OUT our normal 3D render and UNCOMMENT these lines to print:
printable_septum_with_drill_markers();
//printable_symmetrical_clamshell_template_v3();

//printable_final_flare_panel();
//printable_backshort_cap_template();



// ============================================================
// RENDER EXECUTION - RENDER ON SCREEN
// ============================================================
// Comment the PRINTABLE 1:1 PAPER TEMPLATE GENERATOR 
// and uncomment the lines below to see the rendering.

//body_assembly();
//septum_plate();
//flare_assembly();

// ============================================================
// CONSOLE DIAGNOSTICS
// ============================================================
echo("=== OK1DFC 2304 MHz OPTIMIZED FEED DIAGNOSTICS ===");
echo("Waveguide Inside Dimension (wg_id):", wg_id, "mm");
echo("Waveguide Outside Dimension (wg_od):", wg_od, "mm");
echo("Total Waveguide Length:", body_len, "mm");
echo("Probe Target Z from Back Wall:", probe_z_from_end, "mm");
echo("Probe Target Z from Aperture Mouth:", body_len - probe_z_from_end, "mm");
echo("Target W1GHZ Flare Length:", flare_len, "mm");
