
% e) Find the equilibrium using bisection method on R

clc
clear all

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
nss = Pstat(1,1);

% inital guess
epsilon = 1e-15;
R = 1/par.beta - 1 - epsilon;
r = R+par.delta; % set initial r close to rate of time pref
Rmax = R;
lambda = [((1-uss)/nk)*ones(nk,1) (uss/nk)*ones(nk,1)];
lambda_new = lambda;
wtemp = (1-par.alpha)*(r/par.alpha)^(par.alpha/(par.alpha-1));
ctemp = wtemp*(nss);
V_init = (ctemp^(1-par.gamma))/((1-par.gamma)*(1-par.beta));
TV = V_init*ones(2*nk,1);
% In the following, vertical grid of k is current k, horizontal grid of k
% is future k

iter = 1;
ES = 999*ones(50,1);
crit_R = 9;
Rhist = zeros(50,1);
while (crit_R> 1e-7) && (iter <= 49)
    iter_val = 0;
    iter_dist = 0;
    crit_val = 100;
    crit_dist = 100;
    
    % Implied capital/labor ratio
    r = R+par.delta;
    kn = (r/par.alpha).^(1/(par.alpha-1));
    Kdemand = kn*nss;
    % Implied wage
    w = (1-par.alpha)*kn^par.alpha;
    Ce = max((r+1-par.delta)*kron(kver, ones(1,nk) ) + w*ones(nk) - kron(k,ones(nk,1)) ,0);
    Cu = max((r+1-par.delta)*kron(kver, ones(1,nk) ) - kron(k,ones(nk,1)) ,0);
    C = [Ce;Cu];
    U = (C.^(1-par.gamma))./(1-par.gamma);
    
    while  (iter_val <= 1000) && (crit_val >= 1e-7) % This while loop finds the value function
        V = transpose(reshape(TV, nk, 2));
        [TV,I]  = max( U + par.beta*kron(P*V, ones(nk,1)),[],2);
        crit_val = max(abs(TV-reshape(V',2*nk,1)));
        iter_val = iter_val + 1;
        if iter_val >= 1000
            disp(['WARNING: No convergence under 1500 iterations'])
        end
    end
    
    kprime = k(1,I);
    I_e = I(1:nk,1);
    I_u = I(nk+1:2*nk,1);
    kprime_e = k(1,I_e);
    kprime_u = k(1,I_u);
    
    while (iter_dist <= 1000) && (crit_dist >= 1e-6) ; % this while loop finds the stationary distribution
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
    
    
    % Find the implied interest rate
    Ksupply = sum(k*lambda);
    r_implied = par.alpha*((Ksupply/nss)^(par.alpha-1));
    ES(iter) =  Ksupply - Kdemand;
    %     crit_ES = abs(ES(iter));
    R_implied = r_implied-par.delta;
    if iter == 1;
        Rmin = max(R_implied,0);
    else
        if R>R_implied % i.e.  Ksupply > Kdemand;
            Rmax = R;
        else
            Rmin = R;
        end
    end
    Rnew = (Rmax + Rmin) / 2;
    Rhist(iter,1) = R;
    iter = iter+1;
    
    %     crit_R = abs(Rnew-R)/(1+abs(R));
    crit_R = Rmax-Rmin;
    R = Rnew;
    disp(['Interest rate bracket' '  ' 'Conv. metric'])
    disp([Rmin Rmax crit_R])
end

save PS4e lambda k I_e I_u ES Rhist Rmax Rmin R par nss uss nk

%% Print the results for e)
load PS4e

figure(3)
plot(k,lambda(:,1)+lambda(:,2),k,lambda(:,1),k,lambda(:,2))
title('Wealth Distribution')
xlabel('k');ylabel('Density')
legend('total','emp','unemp','Location', 'Best')
figure(4)
plot(k,k,'--',k,k(I_e),k,k(I_u))
title('Policy Functions for Future Capital Holdings')
xlabel('k');ylabel('kprime')
legend('45^o','kprime_e','kprime_u','Location', 'Best')

%% e) Complete market case
R_CMK = 1/par.beta - 1;
r_CMK = R_CMK + par.delta;
kn_CMK = (r_CMK/par.alpha).^(1/(par.alpha-1));
K_CMK = kn_CMK*nss;

%% f) Lorenz Curve and Gini coeff
load PS4e
K = sum(k*lambda);
lambda1 = sum(lambda,2);
cumu_lambda = cumsum(lambda1);
cumu_wealth = cumsum(k.*lambda1')/K;
figure(5)
plot(cumu_lambda,cumu_lambda,'--',cumu_lambda,cumu_wealth)
title('Lorenz Curve')
xlabel('% of the population');ylabel('% of the total wealth')
legend('45^o','Lorenz Curve','Location', 'Best')
axis([0,1,0,1])

disp([num2str(sum(lambda(1,:))*100) ' % of the population are at the borrowing constraint' ])

x = cumu_lambda-[0; cumu_lambda(1:end-1)];
y = cumu_wealth+[0 cumu_wealth(1:end-1)];

gini = 1- y*x;