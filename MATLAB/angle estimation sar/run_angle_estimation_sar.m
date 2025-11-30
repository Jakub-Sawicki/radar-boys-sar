SignalAmplitude = [20];
TargetAngle = [-20]; % [deg]

cfg = getConfig();
M = cfg.M;               % liczba elementów (antenn) w jednej pozycji (np. 2)
virtualArray = [];       % macierz pomiarowa
elementPositions = [];   % pozycje fizyczne każdego elementu w mm

for k = 0:cfg.numSteps-1
    shift_mm = k * cfg.step_mm;
    
    % pozycje anten dla tej iteracji
    pos_1 = shift_mm;
    pos_2 = shift_mm + cfg.d_mm;
    elementPositions = [elementPositions, pos_1, pos_2]; % dodaj pozycje anten do całości
    
    % wykonaj pomiar
    measure_1 = getMeasure_sar(TargetAngle, SignalAmplitude, pos_1);
    measure_2 = getMeasure_sar(TargetAngle, SignalAmplitude, pos_2);
    virtualArray = [virtualArray; measure_1; measure_2]; % każdy pomiar to 2 anteny
end

MLE_sar(virtualArray, elementPositions, cfg.freq);
% kroki = [15, 40, 330]; % Przykładowe liczby kroków do porównania (zmień wg potrzeb, max to cfg.numSteps)
% MLE_comp(virtualArray, kroki, elementPositions, cfg.freq);
