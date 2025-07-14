function out = getMeasure_sar(targetDeg, ampl, elementPositions_mm)
    lambda_mm = 3e8 / 24e9 * 1e3;  % długość fali w mm dla 24 GHz
    out = a_sar(targetDeg, elementPositions_mm, lambda_mm) * ampl.';
    out = out + crandn([length(elementPositions_mm), 1]) / sqrt(2); % dodanie szumu
end