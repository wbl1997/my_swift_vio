function norm3dJacobians()
syms x y z positive
vec = [x;y;z];
e = normalized3d(vec);
r2 = x * x + y * y + z * z;
jac = diff(e, x);
jac_ref = r2 ^ (-1.5) * [y * y + z * z;
    -x * y;
    -x * z
    ];
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(3, 1))));

jac = diff(e, y);
jac_ref = r2 ^ (-1.5) * [- x * y;
    x * x + z * z;
    -y * z
    ];
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(3, 1))));
jac = diff(e, z);
jac_ref = r2 ^ (-1.5) * [-x * z;
    -y * z;
    x * x + y * y
    ];
disp(jac);
dev = simplify(jac - jac_ref);
assert(isequaln(dev, sym(zeros(3, 1))));

end

function res = normalized3d(vec)
  res = vec/norm(vec);
end
