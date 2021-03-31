function endpoints = fit3DLine(xyz)
% xyz[in] 3xn
% endpoints[out] 3x2
xyz0 = mean(xyz,2);
A = xyz-xyz0;
[U,S,~] = svd(A);
d = U(:,1);
t = d'*A;
t1 = min(t);
t2 = max(t);
endpoints = xyz0 + [t1,t2].*d;
end