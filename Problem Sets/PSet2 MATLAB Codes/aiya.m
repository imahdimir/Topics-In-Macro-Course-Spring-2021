function Ksupply = aiya(r)
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
P = [p_e p_u;p_e p_u];
Pstat = P^1000;
uss = Pstat(2,2);

% initial guess for the distribution and value function
lambda = [((1-uss)/nk)*ones(nk,1) (uss/nk)*ones(nk,1)];
lambda_new = lambda;
wtemp = (1-par.alpha)*(r/par.alpha)^(par.alpha/(par.alpha-1));
ctemp = wtemp*(1-uss);
V_init = (ctemp^(1-par.gamma))/((1-par.gamma)*(1-par.beta));
TV = V_init*ones(2*nk,1);
% In the following, vertical grid of k is current k, horizontal grid of k
% is future k

% Implied capital/labor ratio
kn = (r/par.alpha).^(1/(par.alpha-1));
% Implied wage
w = (1-par.alpha)*kn^par.alpha;

Ce = max((r+1-par.delta)*kron(kver, ones(1,nk) ) + w*ones(nk) - kron(k,ones(nk,1)) ,0);
Cu = max((r+1-par.delta)*kron(kver, ones(1,nk) ) - kron(k,ones(nk,1)) ,0);
C = [Ce;Cu]; % c_t matrix
U = (C.^(1-par.gamma))./(1-par.gamma); % u_t matrix

iter_val = 0;
iter_dist = 0;
crit_val = 10;
crit_dist = crit_val;
while  (iter_val <= 1500) && (crit_val >= 1e-7) % This while loop finds the value function
    V = transpose(reshape(TV, nk, 2));
    [TV,I]  = max( U + par.beta*kron(P*V, ones(nk,1)),[],2);
    crit_val = max(abs(TV-reshape(V',2*nk,1)));
    iter_val = iter_val + 1;
end

I_e = I(1:nk,1); % policy for the employed
I_u = I(nk+1:2*nk,1); % policy for the unemployed

while (iter_dist <= 1500) && (crit_dist >= 1e-6) ; % this while loop finds the stationary distribution
    for i=1:nk
        tmp1=ones(nk,1)-min(abs(100*(I_e-i*ones(nk,1))),ones(nk,1));                   % zero-one vectors to indicate which (employed) agents choose capital level k(i)
        tmp0=ones(nk,1)-min(abs(100*(I_u-i*ones(nk,1))),ones(nk,1));                      % same for unemployed agents
        lambda_new(i,1)= sum(  (tmp1.*lambda(:,1))*p_e + (tmp0.*lambda(:,2))*p_e  ) ;   % compute new mass at grid points: employed agents
        lambda_new(i,2)= sum(  (tmp1.*lambda(:,1))*p_u + (tmp0.*lambda(:,2))*p_u  )  ;   % compute new mass at grid points: unemployed agents
    end
    crit_dist=sqrt(mean([(lambda_new(:,1)-lambda(:,1)).^2 ; (lambda_new(:,2)-lambda(:,2)).^2 ]));        % compute convergence criterion
    lambda=lambda_new;       % update distribution
    iter_dist = iter_dist + 1;
end
Ksupply = sum(k*lambda);
end