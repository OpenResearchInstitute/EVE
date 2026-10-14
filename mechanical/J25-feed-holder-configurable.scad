//Configurable dish feed mount for square posts

//make the curve smooth by setting number of faces
$fn = 100;

//inner dimensions in mm of the mounting hole
MountingHoleX = 5;
MountingHoleY = 7;

//inner dimensions of the square fitting
//this part sits on the square post
BodyZ = 20;
BodyY = 32;
BodyX = 31.5;

//dimension required for the section that sets
//the angle of the LNB with respect to dish face
BraceBodyZ = 20;

//mounting angle in degrees
MountingAngle = 135;

//diameter of the LNB
LNBDiameter = 40;

//wall thickness of the part
WallThickness = 4;





difference()
{
    difference() 
    {
        union()
        {
            //make block for outside
            cube([BodyX+WallThickness+WallThickness, BodyY, BodyZ+WallThickness]);
            
            //make brace block, the section that angles the LNB
            translate([0, BodyY,0]) rotate(a=180-MountingAngle, v = [1, 0, 0]) 
        
            difference()
            {
                //outside of brace block
                cube([BodyX+WallThickness+WallThickness, BodyY+WallThickness, BraceBodyZ+WallThickness]);
                
                //cutting block for cable tie wrap might better go here instead
        
                //remove excess material from inside brace block
                translate([WallThickness, 0, 0])
                cube([BodyX, BodyY-10, BraceBodyZ-(1.5*WallThickness)]);
            }
        }
        //cutting block for inside of holder
        translate([WallThickness, 0, 0])
        cube([BodyX, BodyY, BodyZ]);
    }
    
    difference()
    {
        //cutting block for mounting hole
        translate([BodyX/2 + WallThickness, 13+(MountingHoleY/2),0])
        cube([MountingHoleX, MountingHoleY, 60], true);
    }
    
    difference()
    {
        //cutting block for typical LNB body (e.g 40mm diameter)
        translate([BodyX/2+WallThickness, 3*BodyY+WallThickness,0]) 
        rotate(a=180-MountingAngle, v = [1, 0, 0])
        cylinder(100, LNBDiameter/2, LNBDiameter/2);
    
    }
    
    difference()
    {
        //cutting block for cable tie or similar fastener
        
        //this really needs to be done earlier
        //as in, with the brace block
        
        translate([-WallThickness, BodyY+10, BodyZ])
        rotate(a=180-MountingAngle, v = [1, 0, 0])
        cube([50, MountingHoleX/2, MountingHoleX]);
    }

}