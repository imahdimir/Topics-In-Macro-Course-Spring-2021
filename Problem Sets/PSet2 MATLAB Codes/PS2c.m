% Topics in Macro: Solution for PS2 

clc
clear all
close all

eps = 1e-15;
nR = 10;
par.alpha = 0.4; % income share of capital
par.beta = 0.96; % Discount factor
par.gamma = 1.5; % CRRA coefficient
par.delta = 0.1; % depreciation rate
R = linspace(eps, 1/par.beta - 1 - eps, nR); % grid for R
r = R+par.delta; % grid for r
p_e = 0.9;

% create blank matrices for K demand and supply
Ksupply = zeros(nR,1);
Kdemand = Ksupply;

% find the demand and supply for capital for each value of r
for i = 1:nR
Ksupply(i) = aiya(r(i));
Kdemand(i) = p_e*(r(i)/par.alpha)^(1/(par.alpha-1));
end

save PS4c r Ksupply Kdemand
%% Print the results

load PS4c

figure(1)
plot(Ksupply,r,'-o',Kdemand,r,'-o')
title('Capital Market')
xlabel('K');ylabel('r = R+\delta')
legend('K Supply','K Demand','Location', 'Best')