function [SpectAmp,SpectPhase,DCComponent,FirstHamonicLoc,f,THD,RMS] = spectrum(Signal,Freq,TimeStep)
N = 1/TimeStep/Freq;
len = size(Signal,1);
NumberOfPeriod = floor((len+1)/N);
if NumberOfPeriod == 0
    error('Number of sample in signal less than a period.')
end
len = floor(NumberOfPeriod*N);
Signal(len+1:end) = [];
%Frequency Axis
Fmin = 1/TimeStep/len;
f = Fmin:Fmin:(len/2-1)*Fmin;%Hz
%FFT
FFT = fft(Signal,len)/len;
%DC Component
DCComponent = FFT(1,1);
%Spectrum_Amplitude Of Harmonics
SpectAmp = 2*abs(FFT(2:floor(len/2)));
%Spectrum_Phase Of Harmonics
SpectPhase = angle(FFT(2:floor(len/2)))+pi/2;
%RMS
RMS=sqrt(sum(abs(FFT).^2));
% %THD
THD=100*sqrt((sum(SpectAmp.^2)-SpectAmp(NumberOfPeriod,1)^2)/SpectAmp(NumberOfPeriod,1)^2); %THD without DC
%First Harmonic Location On Spect
FirstHamonicLoc = NumberOfPeriod;
end