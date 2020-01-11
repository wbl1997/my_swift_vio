function basic_matrix_Jacobians()
syms w v [3, 1] real
h = w' * v;
disp(['diff(w''*v) relative to w = v''']);
jac = [diff(h, w1), diff(h, w2), diff(h, w3)];
jac_ref = v';
disp(jac);
dev = jac -jac_ref;
assert(isequaln(dev, sym(zeros(1, 3))));

disp(['diff(w''*v) relative to v = w''']);
jac = [diff(h, v1), diff(h, v2), diff(h, v3)];
jac_ref = w';
disp(jac);
dev = jac -jac_ref;
assert(isequaln(dev, sym(zeros(1, 3))));

syms R [3, 3] real
h = R * v;
disp(['diff(R * v) relative to v = R']);
jac = [diff(h, v1), diff(h, v2), diff(h, v3)];
jac_ref = R;
disp(jac);
dev = jac -jac_ref;
assert(isequaln(dev, sym(zeros(3, 3))));
end

