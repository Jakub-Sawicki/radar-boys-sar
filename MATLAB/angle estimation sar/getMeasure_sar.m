function out = getMeasure_sar(targetDeg, ampl, elementPositions_mm)
    cfg = getConfig();
    lambda_mm = cfg.lambda * 1e3;
    out = a_sar(targetDeg, elementPositions_mm, lambda_mm) * ampl.';
    out = out + crandn([length(elementPositions_mm), 1]) / sqrt(2); % dodanie szumu
end