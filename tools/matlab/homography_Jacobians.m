function homography_Jacobians()
syms v w [3, 1] real
syms t theta [3, 1] real
syms R [3, 3] real

h = homography(R, theta, v, w);
disp('diff(homography) relative to fjbar');
assume(theta == 0);
jac = simplify([diff(h, v1), diff(h, v2), diff(h, v3)]);
jac_ref = - sym(eye(3));
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(3))));

disp('diff(homography) relative to fkbar');
jac = [diff(h, w1), diff(h, w2), diff(h, w3)];
jac_ref = R;
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(3))));

disp('diff(homography) relative to \delta\theta of R');
jac = [diff(h, theta1), diff(h, theta2), diff(h, theta3)];
jac_ref = - skew(R * w);
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(3))));
assume(theta, 'clear');
assume(theta, 'real');
end

function h = homography(R_est, theta, fjbar, fkbar)
R = (eye(3) + skew(theta)) * R_est;
h = R * fkbar - fjbar;
end