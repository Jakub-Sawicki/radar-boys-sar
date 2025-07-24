function cfg=getConfig()
    cfg.M=2; % number of antenna elements
    cfg.freq = 10.3943359e9; % signal freqency
    cfg.lambda = 3e8/cfg.freq; % lambda
    cfg.d = cfg.lambda * 1.94; % antenna distance
    cfg.d_mm = cfg.d * 1e3;
    
    cfg.step_mm = 0.5; % przesunięcie radaru (suwnicy)
    cfg.numSteps = 50; % liczba pozycji
end
