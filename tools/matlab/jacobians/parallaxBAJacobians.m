% Compute Jacobians of the parallax angle parameterization with symbolic
% toolbox.
% Parallax angle parameterization is discussed in Zhao et al. Parallax BA.
% IJRR2015
syms x [3 1] real
ksi = x / sqrt(x' * x);
sym_diff = [diff(ksi, x1), diff(ksi, x2), diff(ksi, x3)];
man_diff = eye(3)/sqrt(x' * x) - x * x' / (sqrt(x' * x)^3);
offset = simplify(sym_diff - man_diff);
assert(isequaln(offset, sym(zeros(3))));

clear
syms a n z [3, 1] real
y = sqrt(z' * z);
dy_dz_sim = [diff(y, z1), diff(y, z2), diff(y, z3)];
dy_dz_man = z' / sqrt(z'*z);
offset = dy_dz_sim - dy_dz_man;
assert(isequaln(offset, sym(zeros(1, 3))));