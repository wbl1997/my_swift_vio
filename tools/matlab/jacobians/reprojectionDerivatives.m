function reprojectionDerivatives()
% Compute derivatives of a reprojection factor with anchored inverse depth
% parameterization.

syms x y [3 1] real;
z = skew(x) * y;
dz_dx = [diff(z, x1), diff(z, x2), diff(z, x3)];
dz_dx_expected = - skew([y1; y2; y3]);
offset = simplify(dz_dx - dz_dx_expected);
assert(isequaln(offset, sym(zeros(3))));

syms R [3 3] real
syms t theta [3 1] real
syms a b rho real

ab1rho = [a; b; 1; rho];
f = rho_pC(R, t, theta, ab1rho);
[df_dtheta, df_dt] = drho_pC(R, t, theta, ab1rho);
offset = simplify([diff(f, theta1), diff(f, theta2), diff(f, theta3)] - df_dtheta);
assert(isequaln(offset, sym(zeros(3))));

assume(theta == 0);
offset = simplify([diff(f, t1), diff(f, t2), diff(f, t3)] - df_dt);
assert(isequaln(offset, sym(zeros(3))));
assume(theta, 'clear');
assume(theta, 'real');

syms RBfBa [3 3] real
syms tBfBa [3 1] real

f = rho_pC_full(R, t, RBfBa, tBfBa, theta, ab1rho);
[df_dtheta, df_dt] = drho_pC_full(R, t, RBfBa, tBfBa, theta, ab1rho);
df_dtheta_anal = [diff(f, theta1), diff(f, theta2), diff(f, theta3)];
df_dt_anal = [diff(f, t1), diff(f, t2), diff(f, t3)];

assume(theta == 0);
% TODO(jhuai): the result is still too complex,
% simplify the function or add more assumptions
offset = simplify(df_dtheta_anal - df_dtheta)
% assert(isequaln(offset, sym(zeros(3))));

offset = simplify(df_dt_anal - df_dt)
assert(isequaln(offset, sym(zeros(3))));
assume(theta, 'clear');
assume(theta, 'real');
end


function f = rho_pC(R, t, theta, ab1rho)
Rbar = (eye(3) + skew(theta)) * R;
Rbar_t = Rbar';
f = [Rbar_t, - Rbar_t * t] * ab1rho;
end

function [df_dtheta, df_dt] = drho_pC(R, t, theta, ab1rho)
df_dtheta = R' * skew(ab1rho(1:3) - t * ab1rho(4));
df_dt = - R' * ab1rho(4);
end

function f = rho_pC_full(R_BC, t_BC, R_BfBa, t_BfBa, theta, ab1rho)
R_BC_bar = (eye(3) + skew(theta)) * R_BC;
T_BC_bar = [R_BC_bar, t_BC; 0, 0, 0, 1];
T_CB_bar = [R_BC_bar', - R_BC_bar' * t_BC; 0, 0, 0, 1];
T_BfBa = [R_BfBa, t_BfBa; 0, 0, 0, 1];
F = T_CB_bar * T_BfBa * T_BC_bar * ab1rho;
f = F(1:3);
end

function [df_dtheta, df_dt] = drho_pC_full(R_BC, t_BC, R_BfBa, t_BfBa, theta, ab1rho)
rho_pC = rho_pC_full(R_BC, t_BC, R_BfBa, t_BfBa, theta, ab1rho);
R_CfCa = R_BC' * R_BfBa * R_BC;
df_dtheta = simplify((skew(rho_pC) - R_CfCa * skew(ab1rho(1:3))) * R_BC');
df_dt = simplify(( R_CfCa - eye(3)) * R_BC' * ab1rho(4));
end