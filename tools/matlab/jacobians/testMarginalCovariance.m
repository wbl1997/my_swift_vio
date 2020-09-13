function testMarginalCovariance()
% test computing marginal covariance with Schur complement.
testInvHessian();
testRecursiveSchurComplement();
end

function testInvHessian()
d = 12 + 3 * 2; % C0(camera pose 0), C1, p1(point1), p2
a = rand(d, d);
H = a * a';

SigmaC0 = schurComplement(schurComplement(H, 6), 6) \ eye(d - 12);
SigmaRef = H \ eye(d);
SigmaC0Ref = SigmaRef(1:6, 1:6);
assert(norm(SigmaC0 - SigmaC0Ref) < 1e-6);
end

function testRecursiveSchurComplement()
d = 12 + 3 * 2;
a = rand(d, d);
H = a * a';

% first marginalize 6
H1 = schurComplement(H, 6);
% then marginalize another 6
H2 = schurComplement(H1, 6);

% marginalize 12 at once
H2Ref = schurComplement(H, 12);

assert(norm(H2 - H2Ref) < 1e-7);
end

function Hs = schurComplement(H, r)
% r number of dimensions to remove from the bottom right corner.
k = size(H, 1) - r;
Hs = H(1:k, 1:k) - H(1:k, k+1:end) * (H(k+1:end, k+1:end) \ H(k+1:end, 1:k));
end
