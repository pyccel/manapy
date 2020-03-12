// Gmsh project created on Wed Feb 19 00:17:26 2020// Gmsh project created on Wed Jan 29 15:50:28 2020
SetFactory("OpenCASCADE");


lc = 3;



Point(1) = {0, 0, 0, lc};
Point(2) = {0, 3.6,  0, lc} ;
Point(3) = {36, 3.6, 0, lc} ;
Point(4) = {36,  0, 0, lc} ;

Line(1) = {1,2} ;
Line(2) = {3,2} ;
Line(3) = {3,4} ;
Line(4) = {4,1} ;



Line Loop(1) = {4,1,-2,3} ;


Plane Surface(1) = {1} ;

Physical Line(1) = {4} ;
Physical Line(2) = {2} ;
Physical Line(3) = {3};
Physical Line(4) = {1};



Physical Surface("1") = {4} ;
Physical Surface("2") = {2} ;
Physical Surface("3") = {3} ;
Physical Surface("4") = {1} ;





