SignalAmplitude = [20];
TargetAngle = [30]; % [deg]

step_mm = 0.5;           % przesunięcie radaru (suwnicy)
numSteps = 20;           % liczba pozycji
cfg = getConfig();
lambda = 3e8 / 24e9;     % długość fali (dla 24 GHz)
d_mm = lambda * 1.94 * 1e3;   % odległość między elementami anteny = lambda/2 (w mm)

M = cfg.M;               % liczba elementów (antenn) w jednej pozycji (np. 2)
virtualArray = [];       % macierz pomiarowa
elementPositions = [];   % pozycje fizyczne każdego elementu w mm

for k = 0:numSteps-1
    shift_mm = k * step_mm;
    
    % pozycje anten dla tej iteracji
    pos_k = shift_mm + (0:M-1) * d_mm;
    elementPositions = [elementPositions, pos_k]; % dodaj pozycje anten do całości
    
    % wykonaj pomiar
    measure = getMeasure_sar(TargetAngle, SignalAmplitude, pos_k);
    virtualArray = [virtualArray; measure]; % każdy pomiar to 2 anteny
end

MLE_sar(virtualArray, elementPositions);
