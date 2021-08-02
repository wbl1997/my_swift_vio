function testNoiseIdentification()
% simulate observations
% x_i = x0 + n, n ~ N(0, \sigma^2)
% z_i = Hx_i, H = 1
% E = \Sigma_n ((z_i - x_i)^2 / \sigma^2 + log(\sigma^2))

x_hat = 0;
sigma = 0.5;
num_obs = 100;
x = normrnd(x_hat, sigma, num_obs, 1);
z = ones(num_obs, 1) * x_hat;

options = optimoptions('fminunc','Algorithm','trust-region',...
    'Display','iter','SpecifyObjectiveGradient',true);
problem.options = options;
problem.x0 = [0.1];
problem.objective = @(params)noiseFunctionWithInputData(params, x, z);
problem.solver = 'fminunc';
[hat_sigma,fval] = fminunc(problem)
assert(abs(hat_sigma - sigma) < 0.1);
end

