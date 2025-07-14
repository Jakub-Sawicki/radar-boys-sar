clear
clc
close all

% monopulsePolynom=getMonopulsePolynom(getConfig().M);

SignalAmplitude=[20];
TargetAngle=[-20];

measure=getMeasure(TargetAngle,SignalAmplitude);

% BeamConv(measure)
% BeamMVDR(measure)
MLE(measure)

% monopulse polynom can be calculated once (i.e. at startup), but it varies with M

% MIMP(measure,monopulsePolynom)
