function testfminunc()
options = optimoptions('fminunc','Algorithm','trust-region',...
    'Display','iter','SpecifyObjectiveGradient',true);
problem.options = options;
problem.x0 = [-1,2];
problem.objective = @rosenbrockwithgrad;
problem.solver = 'fminunc';
[x,fval] = fminunc(problem);

% check
eps = 1e-8;
assert(abs(x(1) - 1.0) < eps);
assert(abs(x(2) - 1.0) < eps);
end

function [f,g] = rosenbrockwithgrad(x)
% Calculate objective f
f = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;

if nargout > 1 % gradient required
    g = [-400*(x(2)-x(1)^2)*x(1)-2*(1-x(1));
        200*(x(2)-x(1)^2)];
end
end
