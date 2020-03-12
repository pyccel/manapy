// Gmsh project created on Wed Feb 19 00:17:26 2020// Gmsh project created on Wed Jan 29 15:50:28 2020
SetFactory("OpenCASCADE");


lc = 3;



Point(1) = {0, 0, 0, lc};
Point(2) = {95, 0, 0, lc};
Point(3) = {95, 95,0, lc};
Point(4) = {100, 95, 0, lc};
Point(5) = {100, 0, 0, lc};
Point(6) = {400, 0, 0, lc};
Point(7) = {400, 200, 0, lc} ;
Point(8) = {100, 200, 0, lc};
Point(9) = {100, 162, 0, lc};
Point(10) = {95, 162, 0, lc};
Point(11) = {95, 200, 0, lc};
Point(12) = {0, 200, 0, lc};

Point(13) = {200,100, 0, lc};
Point(14) = {210, 90, 0, lc};
Point(15) = {260, 140, 0, lc};
Point(16) = {250, 150, 0, lc};


//Point(17) = {250,40, 0, lc};
//Point(18) = {260, 30, 0, lc};
//Point(19) = {310, 80, 0, lc};
//Point(20) = {300, 90, 0, lc};

//+
Line(1) = {12, 11};
//+
Line(2) = {11, 10};
//+
Line(3) = {10, 9};
//+
Line(4) = {9, 8};
//+
Line(5) = {8, 7};
//+
Line(6) = {7, 6};
//+
Line(7) = {6, 5};
//+
Line(8) = {5, 4};
//+
Line(9) = {4, 3};
//+
Line(10) = {3, 2};
//+
Line(11) = {2, 1};
//+
Line(12) = {1, 12};

//+
Line(13) = {16, 15};
//+
Line(14) = {15, 14};
//+
Line(15) = {14, 13};
//+
Line(16) = {13, 16};

//+
//Line(17) = {20, 19};
//+
//Line(18) = {19, 18};
//+
//Line(19) = {18, 17};
//+
//Line(20) = {17, 20};

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ;
Line Loop(2) = {13, 14, 15, 16};
//Line Loop(3) = {17, 18, 19, 20};


Plane Surface(1) = {1,2};//,3} ;



Physical Line(1) = {12} ;
Physical Line(2) = {6} ;
Physical Line(3) = {1, 2, 3, 4, 5, 13, 14, 15, 16};
Physical Line(4) = {7, 8, 9, 10, 11};//, 17, 18, 19, 20};



Physical Surface("1") = {12} ;
Physical Surface("2") = {6} ;
Physical Surface("3") = {1, 2, 3, 4, 5, 13, 14, 15, 16} ;
Physical Surface("4") = {7, 8, 9, 10, 11};// ,17, 18, 19, 20} ;





