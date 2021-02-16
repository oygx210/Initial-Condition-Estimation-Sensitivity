% MINI-PROJECT 3
% DATE: November 30st, 2020
% AUTHOR: Alex Nguyen
% DESCRIPTION: Estimation Error and Covariance Plots with Two Different
% Cases

clc; clear; close all;

%----- Simulation Parameters
% Reciever:
x_rx0 = [250, 250, -10, 0]';   % Initial State [m, m, m/s, m/s]'
qx = 0.01; qy = qx;            % Process-Spectral Density Noise [m^2/s^4]

% Radio Frequency (RF) Transmitters:
x_s1 = [0, 0]';                % Initial State [m, m]'
x_s2 = [500, 0]';

% Speed of Light [m/s]:
c = 299792458;  

% Simulation Time:
T = 0.1;                       % Sampling Period [s]
t = (0:T:5)';                 % Experiment Time Duration [s]
SimL = length(t);              % Simulation Time Length 

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
R = 25*eye(nz);               % Measurment Noise Covariance   
r = sqrt(diag(R));            % Measurement Noise St. Dev. 
Q = blkdiag(Qr);              % Process Noise Covariance
q = sqrt(diag(Q));            % Process Noise St. Dev.

% Estimation Error Matrices:
P_est0 = 1e3*eye(nx);          % Initial Reciever Covariance            

% Initial Reciever Estimates:
xRx_est0 = [150, 150, -5, 5]';

% EKF State Initialization:
x_0 = x_rx0;                                     % Reciever System States 
x_s = [x_s1(:); x_s2(:)];                        % RF Transmitter States 
% xz = xRx_est0 + sqrt(diag(P_est0)).*randn(nx, 1);  % Estimate System States  

xz = x_0 + sqrt(diag(P_est0)).*randn(nx, 1);  % Estimate System States 
xz0 = xz;                                        % Initial Estimate States

% Preallocation:
z = zeros(nz, SimL);                              
x_est = zeros(nx, SimL); 
P_est = x_est;
x_true = x_est;

for k = 1:SimL
    % Reciever Vectors:
    Xu = @(x) [x(1); x(2)]; 
    
    % RF Transmitter 1 & 2 Measurement Equations:
    r1 = @(x) norm(Xu(x) - x_s1); 
    r2 = @(x) norm(Xu(x) - x_s2);

    % Observation Jacobian (nz x nx):
    Hk = @(x) [(x(1) - x_s(1))./r1(x), (x(2) - x_s(2))./r1(x), zeros(1, 2);
               (x(1) - x_s(3))./r2(x), (x(2) - x_s(4))./r2(x), zeros(1, 2)];  
        
    % True Pseudorange Measurment RF Tx 1 -5
    z_true = [r1(x_0); r2(x_0)];
    z(:, k) = z_true + r.*randn(nz, 1);  

    % True State Values (Rx and SOP 2)
    x_true(:, k) = x_0;                   
    
    if k == 1       
        % Prediction
        x_estn = xz;
        P_estn = P_est0;
        
        % Update
        H = Hk(x_estn);
        z_est = [r1(x_estn); r2(x_estn)];
        yk_res = z(:, k) - z_est;
        Sk = H*P_estn*H' + R;
        Kk = P_estn*H'*inv(Sk);
         
        % Correction
        xz = x_estn + Kk*yk_res;
        P_est0 = (eye(nx) - Kk*H)*P_estn;
        
        % Save Estimate Values
        x_est(:, k) = xz;
        P_est(:, k) = diag(P_est0);
        
    else
        % Prediction
        x_estn = Fk*xz;
        P_estn = Fk*P_est0*Fk' + Q;
        
        % Update
        H = Hk(x_estn);
        z_est = [r1(x_estn); r2(x_estn)];
        yk_res = z(:, k) - z_est;
%         Sk = H*P_estn*H' + R;
%         Kk = P_estn*H'*inv(Sk);

        P_xy = P_estn*H';
        P_yy = H*P_estn*H' + R;
        Kk = P_xy/P_yy;
         
        % Correction
%         xz = x_estn + Kk*yk_res;
%         P_est0 = (eye(nx) - Kk*H)*P_estn;
        
        xz = x_estn + Kk*yk_res;
        A = eye(nx) - Kk*H;
        P_est0 = A*P_estn*A' + Kk*R*Kk';         

        % Save Values
        x_est(:, k) = xz;
        P_est(:, k) = diag(P_est0);
    end
    
    % Next Step
%     x_0 = f(x_0) + q.*randn(nx, 1);
    x_0 = f(x_0);

end

% Estimation Error 
x_tilde = x_true - x_est;    % Full System's Error Trajectories
x_P = sqrt(P_est);           % Full System's Error Variance Bounds

%----- Plot Results
% Estimation Trajectories:
figure;  
subplot(2,2,1) 
x = plot(t, x_tilde(1, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(1, :), 'r--',  'linewidth',1.5); 
plot(t, -2*x_P(1, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{x}_r$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_1(t_k)$','$\pm 2 \sigma_1(t_k)$','interpreter','latex')

subplot(2,2,2)
x = plot(t, x_tilde(2, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(2, :), 'r--', 'linewidth',1.5); 
plot(t, -2*x_P(2, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{y}_r$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_2(t_k)$','$\pm 2 \sigma_2(t_k)$','interpreter','latex')

subplot(2,2,3) 
x = plot(t, x_tilde(3, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(3, :), 'r--', 'linewidth',1.5); 
plot(t, -2*x_P(3, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{\dot{x}}_r$ [m/s]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_3(t_k)$','$\pm 2 \sigma_3(t_k)$','interpreter','latex')

subplot(2,2,4)
x = plot(t, x_tilde(4, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(4, :), 'r--' , 'linewidth',1.5); 
plot(t, -2*x_P(4, :), 'r--' , 'linewidth',1.5); hold off;
ylabel('$\tilde{\dot{y}}_r$ [m/s]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_4(t_k)$','$\pm 2 \sigma_4(t_k)$','interpreter','latex')
sgtitle('System $\Sigma$: Estimation Error Trajectories with $\pm 2\sigma$ Bounds','interpreter','latex')

for ii = 1:4
    sph = subplot(2,2,ii); % Resize Subplots
    dx0 = -0.05;
    dy0 = -0.025;
    dwithx = 0.03;
    dwithy = 0.03;
    set(sph,'position',get(sph,'position') + [dx0, dy0, dwithx, dwithy])
end

