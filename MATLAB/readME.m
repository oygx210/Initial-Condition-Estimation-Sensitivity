% Thoughts for the Mini-Project 3 Script:
%
% SIMULATION: 
% - Generate the reciever dynamics in the ekf for loop (rather than
%   external to the script) since we only know the inital conditions.
% - Try the ekf for reciever dynamics with and without noise
% - Generate true measurment values for the two sops (try r(k) model
%   written on paper and the true pseudorange model in paper)
% - Estimate reciever dynamics and sop transmitter positions (and clock
%   dynamics too after) in ekf
% - Play around with different process and measurement noise values 
% - Try different values for the initial states (Rx and SOP) and the error
%   covariance matrix estimates. 
% - Plot the estimation error trajectory and confidence bounds for the system
% 
% RESULTS:
% - Latex up the problem statement and a summary of results from this
%   simulation. 
%       - Did I notice a pattern from trying different initial conditions?
%       - How well do the observers track the actual state of the system?
%       - "I should notice some nonlinear phenomena when running the simulation"  
% - Display the plots and inital conditions used for various plots (include
%   a table for the different cases). 
%
% FINALLY:
% - Clean up the MATLAB and Latex script then submit to Prof. Kassas 
 