#!/usr/bin/gnuplot
set term wxt 0
reset; set datafile separator " "; 
plot "sinusoidRMSE.txt" using 2 w l title "x", "" using 3 w l title "y", "" using 4 w l title "z"; set grid; show grid;

set term wxt 1
reset; set datafile separator " "; 
plot "sinusoidRMSE.txt" using 5 w l title "roll", "" using 6 w l title "pitch", "" using 7 w l title "yaw"; set grid; show grid;

set term wxt 2
set datafile separator " "; 
plot "sinusoidNEES.txt" using 2 w l title "p_{WS}", "sinusoidNEES.txt" using 3 w l title "alpha_{WS}", "sinusoidNEES.txt" using 4 w l title "T_{WS}"; set grid; show grid;

set term wxt 3
set datafile separator " "; 
splot "sinusoidMSCKF2_0.txt" using 3:4:5 w l, "sinusoidTruth.txt" using 3:4:5 w l; set view equal xyz; set grid; show grid;

# set terminal postscript portrait enhanced mono dashed lw 1 'Helvetica' 14
# set output '<SOME OUTPUT FILE>.ps'

pause -1




