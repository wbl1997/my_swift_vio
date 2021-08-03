
function [left, right] = twoSidedProbabilityRegionNees(Q, xdof, N)
% Estimation with applications to tracking and navigation: theory algorithms and software
% 3.7.6-4
dof = xdof * N;
left = chi2inv(Q * 0.5, dof) / N;
right = chi2inv(1 - Q * 0.5, dof) / N;
end