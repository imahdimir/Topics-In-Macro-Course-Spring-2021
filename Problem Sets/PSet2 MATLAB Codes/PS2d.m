

clc
clear all

eps = 1e-15;
nR = 10;
par.alpha = 0.4; % income share of capital
par.beta = 0.96; % Discount factor
par.gamma = 1.5; % CRRA coefficient
par.delta = 0.1; % depreciation rate
R = linspace(eps, 1/par.beta - 1 - eps, nR);  % grid for R
r = R+par.delta;
r_CMK = r;
% r_CMK = [r(1:end-1) r(end)-0.001 r(end)];  % grid for r 
% create blank matrices for K demand and supply
Ksupply_CMK = zeros(nR,1);
% find the supply for capital for each value of r in the Complete Market
% version
for i = 1:nR
Ksupply_CMK(i) = CMK(r_CMK(i));
end

save PS4d Ksupply_CMK r_CMK
%% Print the results and compare with the incomplete market case
load PS4c
load PS4d

figure(2)
plot(Ksupply,r,'-o',Kdemand,r,'-o',Ksupply_CMK,r_CMK,'o')
title('Capital Market')
xlabel('K');ylabel('r = R+\delta')
legend('K Supply','K Demand','K Supply (CMK)','Location', 'Best')