function [f, g] = noise_fun_with_input_data(parameters, states, observations)
e = states - observations;
sigma2 = parameters(1)^2;
f = e' * e / sigma2 + size(e, 1) * 2 * log(parameters(1));
if nargout > 1
    g = -2 * e' * e / (parameters(1)^3) + size(e, 1) * 2 / parameters(1);
end
end

