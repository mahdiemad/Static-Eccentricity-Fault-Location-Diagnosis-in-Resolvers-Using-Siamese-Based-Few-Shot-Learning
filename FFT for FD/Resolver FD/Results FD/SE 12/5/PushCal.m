clc
clear all
FEMSigal = csvread('Winding Plot 1.csv',1);
time = FEMSigal(:,1)*1e-3;
V_s = FEMSigal(:,2);
V_c = FEMSigal(:,3);
timeStep = time(2)-time(1);

Fs = 5000;

% Envelope detection
step = round(1/Fs/timeStep);
TimePush = time(1:step:end);
Ec = V_s(1:step:end);
Es = V_c(1:step:end);

Es = Es / max(Es);


