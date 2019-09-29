function epipolar_constraint_Jacobians()
syms v w [3, 1] real
syms t theta [3, 1] real
syms R [3, 3] real

h = epipolar_constraint(R, t, theta, v, w);
disp('diff(epipolar constraint) relative to fj');
assume(theta == 0);
jac = simplify([diff(h, v1), diff(h, v2), diff(h, v3)]);
jac_ref = (R * w)' * skew(t);
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(1, 3))));

disp('diff(epipolar constraint) relative to fk');
jac = [diff(h, w1), diff(h, w2), diff(h, w3)];
jac_ref = (skew(t) * v)' * R;
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(1, 3))));

disp('diff(epipolar constraint) relative to t');
jac = [diff(h, t1), diff(h, t2), diff(h, t3)];
jac_ref = - (R * w)' * skew(v);
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(1, 3))));

disp('diff(epipolar constraint) relative to \delta\theta of R');
jac = [diff(h, theta1), diff(h, theta2), diff(h, theta3)];
jac_ref = (R * w)' * skew(cross(t, v));
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(1, 3))));
assume(theta, 'clear');
assume(theta, 'real');

end

function h = epipolar_constraint(R_est, t, theta, fj, fk)
R = (eye(3) + skew(theta)) * R_est;
h = (R * fk)' * (skew(t) * fj);
end