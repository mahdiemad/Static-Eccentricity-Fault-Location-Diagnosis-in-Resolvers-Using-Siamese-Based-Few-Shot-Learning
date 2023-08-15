Signal = Es;
Freq = 8;
TimeStep = TimePush(2)-TimePush(1);

[SpectAmp,SpectPhase,DCComponent,FirstHamonicLoc,f,THD,RMS] = spectrum(Signal,Freq,TimeStep);

F = f';

% y = DCComponent;
% for i=1:311
%    y = y+ SpectAmp(i)*sin(2*pi*f(i)*time'+SpectPhase(i));
% end
