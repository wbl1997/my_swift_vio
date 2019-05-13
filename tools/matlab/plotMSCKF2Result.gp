# usage: gnuplot plotMSCKF2Result.txt
# to close all windows: pkill -x gnuplot

filename= "/home/jhuai/Desktop/testHybridFilter.txt"
set term wxt 0
#set term postscript eps enhanced color
#set output "/home/jhuai/Desktop/pxy.eps" #if this line is enabled, the -p or --persist in gnuplot won't work
set size ratio -1;
set xlabel "x [m]"
set ylabel "y [m]"
plot filename using 3:4 lc rgb "#FF0000" title 'pxy' w lines
set term wxt 1
set xlabel "time [s]"
#set term postscript eps enhanced color
#set output "/home/jhuai/Desktop/qxyz.eps"
plot filename using 1:6 lc rgb "#FF0000" title 'qx' w lines, filename using 1:7 lc rgb "#00FF00" title 'qy' w lines, filename using 1:8 lc rgb "#0000FF" title 'qz' w lines

set term wxt 2
set xlabel "time [s]"
set ylabel "velocity [m/s]"
plot filename using 1:10 lc rgb "#FF0000" title 'vx' w lines, filename using 1:11 lc rgb "#00FF00" title 'vy' w lines, filename using 1:12 lc rgb "#0000FF" title 'vz' w lines

set term wxt 3
set xlabel "time [s]"
set ylabel "bg [rad/s]"
plot filename using 1:13 lc rgb "#FF0000" title 'bgx' w lines, filename using 1:14 lc rgb "#00FF00" title 'bgy' w lines, filename using 1:15 lc rgb "#0000FF" title 'bgz' w lines


set term wxt 4
set xlabel "time [s]"
set ylabel "ba [m/s^2]"
plot filename using 1:16 lc rgb "#FF0000" title 'bax' w lines, filename using 1:17 lc rgb "#00FF00" title 'bay' w lines, filename using 1:18 lc rgb "#0000FF" title 'baz' w lines

set term wxt 5
set xlabel "time [s]"
set ylabel "Tg [m/s^2]"
plot filename using 1:16 lc rgb "#FF0000" title 'bax' w lines, filename using 1:17 lc rgb "#00FF00" title 'bay' w lines, filename using 1:18 lc rgb "#0000FF" title 'baz' w lines



set term wxt 5
set xlabel "time [s]"
set ylabel "pcins [m]"
plot filename using 1:46 lc rgb "#FF0000" title 'px' w lines, filename using 1:47 lc rgb "#00FF00" title 'py' w lines, filename using 1:48 lc rgb "#0000FF" title 'pz' w lines
