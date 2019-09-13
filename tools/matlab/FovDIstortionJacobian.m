syms radius omega u v;
factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
                 (radius * omega);
simplify(taylor(factor, omega, 'order', 3))
simplify(taylor(factor, radius, 'order', 3))
disp('dfactor/domega')
domega = simplify(diff(factor, omega))
simplify(taylor(domega, omega, 'order', 3))
simplify(taylor(domega, radius, 'order', 3))
disp('dfactor/dradius')
dradius = simplify(diff(factor, radius))
simplify(taylor(dradius, omega, 'order', 3))
simplify(taylor(dradius, radius, 'order', 3))
