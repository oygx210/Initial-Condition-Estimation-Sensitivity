% MINI-PROJECT 3
% DATE: November 30st, 2020
% AUTHOR: Alex Nguyen
% DESCRIPTION: Estimation Error and Covariance Plots with Two Different
% Cases While Varying Initial Conditions

clc; clear; close all;

%----- Simulation Parameters
% Reciever:
x_rx0 = [150, 100, -15, 0]';        % Initial State [m, m, m/s, m/s]'
qx = 0.1; qy = qx;                 % Process-Spectral Density Noise [m^2/s^4]

% Radio Frequency (RF) Transmitters:
x_s1 = [25, 0]';           % Initial States [m, m]'
x_s2 = [100, 0]';

% Speed of Light [m/s]:
c = 299792458;  

% Simulation Time:
T = 10e-3;                                  % Sampling Period [s]
t = (0:T:10)';                              % Experiment Time Duration [s]
SimL = length(t);                           % Simulation Time Length

%----- RF Transmitter Dynamics     
% "Jacobian" for RF Dynamics:
Fs = eye(2);    
 
% White Noise Covariance:
Qs = zeros(2);  

%----- Reciever Dynamics                
% "Jacobian" for Receiver Dynamics:
Fpv =   [eye(2), T*eye(2); ...  
         zeros(2), eye(2)];  

% P.V. Process Noise Covariance (Random Walk Velocity):
Qpv = [qx*T^3/3,     0,     qx*T^2/2,  0; ...  
          0,      qy*T^3/3,    0,     qy*T^2/2; ...
       qx*T^2/2,     0,       qx*T,    0; ...
          0,      qy*T^2/2,    0,     qy*T];

% White Noise Covariance:
Qr = Qpv;  

%----- EKF State Estimation
% Number of States:
nx = 4;                       % Full System States (Rx & RF Tx 1-5)
nz = 2;                       % RF Tx 1 - 5 Measurement States

% Augmented System:
Fk = Fpv;
f = @(x) Fk*x;

% Noise Covariance Matrices and Standard Deviations (e.g. wk & vk):
R = 20*eye(nz);               % Measurment Noise Covariance   
r = sqrt(diag(R));            % Measurement Noise St. Dev. 
Q = Qr;                       % Process Noise Covariance
q = sqrt(diag(Q));            % Process Noise St. Dev.

% Estimation Error Matrices:
P_rx0 = 1e3*blkdiag(1, 3, 1, 1);     % Initial Rx Covariance
P_est0 = blkdiag(P_rx0);             % Full System Covariance

% EKF State Initialization:
x_rxest0 = [100, -5, -10, 5]';                  % Initial Reciever Estimate
x_0 = x_rx0;                                    % Reciever System States 
x_s = [x_s1; x_s2];                             % RF Transmitter States
xz = x_0 + sqrt(diag(P_est0)).*randn(nx, 1);    % Estimate System States  
% xz = x_rxest0 + sqrt(diag(P_est0)).*randn(nx, 1);    % Estimate System States 
xz0 = xz;                                       % Initial Estimate State

% Preallocation:
z = zeros(nz, SimL);                              
P_est = zeros(nx, SimL);
x_est = P_est; 
x_true = x_est;
x_rw = zeros(4, SimL); 
ep = zeros(SimL, 1);

for k = 1:SimL
    % RF Transmitter 1 & 2 Measurement Equations:
    h1 = @(x) sqrt((x(1) - x_s(1)).^2 + (x(2) - x_s(2)).^2);
    h2 = @(x) sqrt((x(1) - x_s(3)).^2 + (x(2) - x_s(4)).^2);
    
    % Observation Jacobian (nz x nx):
    Hk = @(x) [(x(1) - x_s(1))./h1(x), ...
        (x(2) - x_s(2))./h1(x), ...
        0, ...
        0;
        
        (x(1) - x_s(3))./h2(x), ...
        (x(2) - x_s(4))./h2(x), ...
        0, ...
        0];
    
    % True Pseudorange Measurment RF Tx 1 & 2:
    z_true = [h1(x_0); h2(x_0)];
    z(:, k) = z_true + r.*randn(nz, 1);
%         z(:, k) = z_true;
    
    % True State Values (Rx and SOP 2):
    x_true(:, k) = x_0;
    
    if k == 1
        % Initial Prediction:
        x_estn = xz;
        P_estn = P_est0;
    else
        % Prediction:
        x_estn = f(xz);
        P_estn = Fk*P_est0*Fk' + Q;
    end
    
    % Update:
    H = Hk(x_estn);
    z_est = [h1(x_estn); h2(x_estn)];
    yk_res = z(:, k) - z_est;
    Sk = H*P_estn*H' + R;
    Kk = P_estn*H'*inv(Sk);
    
    % Correction:
    xz = x_estn + Kk*yk_res;
    P_est0 = (eye(nx) - Kk*H)*P_estn;
    
    % Save Values:
    x_est(:, k) = xz;
    P_est(:, k) = diag((P_est0));
    
    % Normalized Estimation Error Squared (NEES):
    xbar = x_true(:, k) - xz;
    ep(k) = xbar'*inv(P_est0)*xbar;
    
    % Next Step:
    x_0 = f(x_0) + q.*randn(nx, 1);
%         x_0 = f(x_0);
end

% Full System Error Analysis:
x_tilde = x_true - x_est;    % Error Trajectories
x_P = sqrt(P_est);           % Error Variance Bounds
ep_avg = sum(ep)/SimL;       % Average NEES for Nth Trial

% Print Results:
fprintf('NEES for the One Trial = %4.4f\n', ep_avg)
fprintf('Initial Estimate Vector, xz0: \n'); disp(xz0);

%----- Plot Results
% Random Walk Dynamics:
figure;
plot(x_true(1, :), x_true(2, :), 'Color', '#D95319', 'linewidth', 3); hold on;
plot(x_true(1, 1), x_true(2, 1), 'go', 'linewidth', 6);
plot(x_true(1, end), x_true(2, end), 'ro', 'linewidth', 6);
plot(x_s(1), x_s(2), 'ks', 'linewidth', 6); 
plot(x_s(3), x_s(4), 'ks', 'linewidth', 6); hold off;
xlabel('x [m]'); ylabel('y [m]')
legend('Random-Walk Receiver', 'Receiver Start Point', 'Receiver End Point', 'RF Transmitter 1 & 2', 'location', 'best');
title('COpNav Reciever Random-Walk Dynamics')
axis equal;

% Simulated Environment:
figure; 
hold on;
plot(x_est(1, :), x_est(2, :), 'linewidth', 2);
plot(x_true(1, :), x_true(2, :), ':', 'linewidth', 2.5);
plot(x_s(1), x_s(2), 'rs', 'markersize', 10);
plot(x_s(3), x_s(4), 'rs', 'markersize', 10); hold off;
xlabel('x [m]'); ylabel('y [m]');
legend('Estimated Rx Trajectory', 'True Rx Trajectory', 'RF Tx 1 & 2', 'location', 'best')
title('COpNav Reciever State Observer Tracking')
axis equal; grid on;

% Estimation Trajectories:   
figure; 
leg1 = {'$\tilde{x}_1(t_k)$', '$\tilde{x}_2(t_k)$', ...  % Legend Labels
            '$\tilde{x}_3(t_k)$', '$\tilde{x}_4(t_k)$'};
leg2 = {'$\pm 2 \sigma_1(t_k)$', '$\pm 2 \sigma_2(t_k)$', ...
            '$\pm 2 \sigma_3(t_k)$', '$\pm 2 \sigma_4(t_k)$'};
ylab = {'$\tilde{x}_r$ [m]','$\tilde{y}_r$ [m]', ...     % ylabel Labels
            '$\tilde{\dot{x}}_r$ [m/s]', '$\tilde{\dot{y}}_r$ [m/s]'};
for ii = 1:4
    % Plot Error Estimation Trajectories:
    subplot(2,2,ii)
    plot(t, x_tilde(ii, :), 'linewidth', 2); hold on;
    plot(t, -2*x_P(ii, :), 'r--',  'linewidth',1.5);
    plot(t, 2*x_P(ii, :), 'r--', 'linewidth',1.5); hold off;
    xlabel('Time [s]'); ylabel(ylab(ii), 'interpreter', 'latex');
    legend(leg1{ii}, leg2{ii},'interpreter','latex')
    
    % Resize Subplots:
    sph = subplot(2,2,ii); 
    dx0 = -0.04;  dy0 = -0.03;
    dwithx = 0.03; dwithy = 0.03;
    set(sph,'position',get(sph,'position') + [dx0, dy0, dwithx, dwithy])
end
sgtitle('Estimation Error Trajectories with $\pm 2\sigma$ Bounds','interpreter','latex')
