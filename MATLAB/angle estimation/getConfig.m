function cfg=getConfig()
    cfg.M=2; % number of antenna elements
    cfg.freq = 10.3943359e9; % signal freqency
    cfg.lambda = 3e8/cfg.freq; % lambda
    % cfg.d = cfg.lambda*1.94; % antenna distance
    cfg.d = cfg.lambda*1/2; % antenna distance
end
