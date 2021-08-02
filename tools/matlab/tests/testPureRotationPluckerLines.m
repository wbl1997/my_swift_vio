pure_rotation = 1;
s1 = [rand(2, 1) - 0.5; 1];
e1 = [rand(2, 1) - 0.5; 1];
s1 = normalize(s1, 'norm', 2);
e1 = normalize(e1, 'norm', 2);

theta = 15 * pi / 180;
R21 = [cos(theta), -sin(theta), 0;
    sin(theta), cos(theta), 0; 
    0, 0, 1];
if pure_rotation
    t21 = [0, 0, 0]';
else
    t21 = [-1, 0, 0]';
end

depth_s = 6;
depth_e = 6.5;
s2 = R21 * s1 * depth_s + t21;
e2 = R21 * e1 * depth_e + t21;
c1 = [0;0;0];
c2 = - R21' * t21;
s2 = normalize(s2, 'norm', 2)
e2 = normalize(e2, 'norm', 2)

pxyz1 = cross(s1, e1);
pxyz2 = cross(s2, e2);
pw1 = c1' * pxyz1;
pw2 = c2' * pxyz2;

p1 = [pxyz1; pw1];
p2 = [pxyz2; pw2];
L_star = p1 * p2' - p2 * p1'
% L_star = [cross(d), n; -n', 0];
% distance = norm(n) / norm(d);
% distance will be zero under pure rotation