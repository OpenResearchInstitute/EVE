//This is a mast adapter for Hello Giggy EME Station
//By Abraxas3d 19 August 2026
//Allows fitting of a Yaesu G-5500 style rotator
//To an ITE tripod (T6, serial 4170)

//============================================================
//Plan:
//scrap part is 116 mm in height and 57 mm in diameter.
//bore out 41 mm diamter centered hole all the way through.
//bore out 45 mm diamger centered hole 10 mm from base.
//drill 9 mm diamter hole 50 mm from top (for anti-spin bolt). 
//============================================================

$fn = 100;

//how high the mast adapter is suspended above stub.
//not critical - just to make it look good. 
part_lift = 200;

//base of stub
cylinder(130, 51/2, 51/2);

//step of stub
translate([0, 0, 130])
{
cylinder(10, 45/2, 45/2);
}
//stub itself
translate([0, 0, 140])
{
cylinder(42, 41/2, 41/2);
}

//stub is a total of 130 + 10 + 42 = 182 mm tall

difference()
{
//the rotator clamps to the created mast adapter.
//First, we create three cylinders using the
//measurements needed to clear the stub part.
//We take the difference of the two smaller
//cylinder from the cylinder that is the
//same size as the scrap piece of aluminum
//that we had on hand. 
{
translate([0, 0, part_lift])
{
difference()
{
//scrap aluminum piece to start with
cylinder(116, 57/2, 57/2);
//take away stub diameter, reamed out all the way through
cylinder(116, 41/2, 41/2);
//take away step diameter, carved out at bottom
cylinder(10, 45/2, 45/2);
}
}
}

//Second, we create the hole for the anti-spin 
//bolt. We create a cylinder, rotate, then translate.
//We take the difference of the mast adapter above
//and the anti-spin bolt hole below. 
{
translate([0, 30, part_lift+116-50])
{
rotate([90, 0, 0])
{
cylinder(20, 9/2, 9/2);
}
}
}

}