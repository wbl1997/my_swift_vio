function curves()
% many 3d skew curves with expressions are found at
% https://mathcurve.com/courbes3d.gb/courbes3d.shtml
close all;
curveName='lissajousknot';
t0 = 0;
t1 = 10;
a = 1;
b = a;
c = 0.6;
q = 5;
p = 7;
r = 11;
phi = pi * 0.5;
psi = pi * 0.5;
[timestamps, xyz] = lissajousknot(t0, t1, a, b, c, p, q, r, phi, psi);
% figure;
% draw3dcurve(xyz, curveName);

curveName = 'eightcurve';
t0 = 0;
t1 = 10;
a = 1.0;
[timestamps, xyz] = eightcurve(t0, t1, a);
% figure;
% draw3dcurve(xyz, curveName);

curveName = 'grannyknot';
t0 = 0;
t1 = 10;
[timestamps, xyz] = grannyknot(t0, t1);
% figure;
% draw3dcurve(xyz, curveName);


curveName = 'toricsolenoid';
t0 = 0;
t1 = 100;
R = 3;
r = 1;
p = 11;
q = 13;
[timestamps, xyz] = toricsolenoid(t0, t1, R, r, p, q);
figure;
draw3dcurve(xyz, curveName);

end

function draw3dcurve(xyz, curveName)
    plot3(xyz(1, :), xyz(2, :), xyz(3, :), 'b-');
    axis equal; grid on;
    xlabel('x'); ylabel('y'); zlabel('z');
    title(curveName);
end

function [timestamps, xyz] = lissajousknot(t0, t1, a, b, c, p, q, r, phi, psi)
% https://mathcurve.com/courbes3d.gb/lissajous3d/noeudlissajous.shtml
timestamps = t0:0.02:t1;
xyz = zeros(3, length(timestamps));
index = 0;
for t=timestamps
    x = a*cos(q*t);
    y = b*cos(p*t + phi);
    z = c*cos(r*t + psi);
    index = index + 1;
    xyz(:, index) = [x; y; z];
end
end

function [timestamps, xyz] = eightcurve(t0, t1, a)
% https://mathcurve.com/courbes2d/gerono/gerono.shtml
timestamps = t0:0.02:t1;
xyz = zeros(3, length(timestamps));
index = 0;
for t=timestamps
    x = a * sin(t);
    y = a * sin(t) * cos(t);
    z = 0;
    index = index + 1;
    xyz(:, index) = [x; y; z];
end
end

function wavycicle()
% also give the surface equations.

end

function [timestamps, xyz] = toricsolenoid(t0, t1, R, r, p, q)
% The curve on a torus surface, eq. 4.63 in Huai thesis is a special case
% of the toric solenoid.
% https://mathcurve.com/courbes3d.gb/solenoidtoric/solenoidtoric.shtml
timestamps = t0:0.02:t1;
xyz = zeros(3, length(timestamps));
index = 0;
n = p/q;
for t=timestamps
    rxy = R + r * cos(n*t);
    x = rxy * cos(t);
    y = rxy * sin(t);
    z = r * sin(n*t);
    index = index + 1;
    xyz(:, index) = [x; y; z];
end
end

function satellitecurve()
% https://mathcurve.com/courbes3d.gb/satellite/satellite.shtml
end

function [timestamps, xyz] = grannyknot(t0, t1)
% https://mathcurve.com/courbes3d.gb/plat.vache/plat_vache.shtml
timestamps = t0:0.02:t1;
xyz = zeros(3, length(timestamps));
index = 0;
for t=timestamps
    x = 3*sin(t) + 2*sin(3*t);
    y = cos(t) - 2*cos(3*t);
    z = sin(10*t);
    index = index + 1;
    xyz(:, index) = [x; y; z];
end
end