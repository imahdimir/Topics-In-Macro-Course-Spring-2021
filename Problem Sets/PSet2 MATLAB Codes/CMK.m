function Ksupply = CMK(r)
nk = 500; % number of k grid points
phi = 0.01;
kmin = phi;
kmax = 25;

k = linspace(kmin, kmax, nk); % capital grid
kver = k';

par.alpha = 0.4; % income share of capital
par.beta = 0.96; % Discount factor
par.gamma = 1.5; % CRRA coefficient
par.delta = 0.1; % depreciation rate
p_e = 0.9; % Pr(employed next period)
p_u = 1-p_e;

% initial guess for the distribution and value function
lambda = (1/nk)*ones(nk,1);
lambda_new = lambda;
wtemp = (1-par.alpha)*(r/par.alpha)^(par.alpha/(par.alpha-1));
ctemp = wtemp*(1-p_u);
V_init = (ctemp^(1-par.gamma))/((1-par.gamma)*(1-par.beta));
TV = V_init*ones(nk,1);
% In the following, vertical grid of k is current k, horizontal grid of k
% is future k

% Implied capital/labor ratio
kn = (r/par.alpha).^(1/(par.alpha-1));
% Implied wage
w = (1-par.alpha)*kn^par.alpha;

C = max((r+1-par.delta)*kron(kver, ones(1,nk) ) + w*p_e*ones(nk) - kron(k,ones(nk,1)) ,0); % c_t matrix
U = (C.^(1-par.gamma))./(1-par.gamma); % u_t matrix

iter_val = 0;
iter_dist = 0;
crit_val = 10;
crit_dist = crit_val;
while  (iter_val <= 1500) && (crit_val >= 1e-7) % This while loop finds the value function
    V = transpose(TV);
    [TV,I]  = max( U + par.beta*kron(V, ones(nk,1)),[],2);
    crit_val = max(abs(TV-V'));
    iter_val = iter_val + 1;
end

while (iter_dist <= 1500) && (crit_dist >= 1e-6) ; % this while loop finds the stationary distribution
    for i=1:nk
        tmp=ones(nk,1)-min(abs(100*(I-i*ones(nk,1))),ones(nk,1));                   % zero-one vectors to indicate which agents choose capital level k(i)
        lambda_new(i,1)= tmp'*lambda   ;   % compute new mass at grid points
    end
    crit_dist=sqrt(mean((lambda_new-lambda).^2));        % compute convergence criterion
    lambda=lambda_new;       % update distribution
    iter_dist = iter_dist + 1;
end

Ksupply = k*lambda;
end