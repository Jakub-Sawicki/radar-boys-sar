function angleDetEstimated = MLE_sar(Y, elementPositions_mm, freq)
    [M, N] = size(Y);
    R = Y * Y' / N;
    lambda_mm = 3e8 / freq * 1e3;

    function J = CostFunction(angleDeg)
        aTemp = a_sar(angleDeg, elementPositions_mm, lambda_mm);
        Pv = eye(M) - aTemp * ((aTemp') * aTemp)^(-1) * aTemp';
        J = abs(trace(Pv * R));
    end
    
    minAngle = -45;
    maxAngle = 44;
    angleVec = minAngle:0.1:maxAngle;
    pval = zeros(size(angleVec));
    for k = 1:length(angleVec)
        pval(k) = CostFunction(angleVec(k));
    end
    figure;
    plot(angleVec, pval);
    title("MLE z syntetycznej anteny");
    xlabel("Kąt [deg]");
    ylabel("Funkcja kosztu");
    
    angleDetEstimated = fminsearch(@CostFunction, minAngle);
    angleCost = CostFunction(minAngle);
    for i = (minAngle + 10):10:45
        if CostFunction(i) < angleCost 
            angleDetEstimated = fminsearch(@CostFunction, i);
            angleCost = CostFunction(i);
        end
    end
    angleDetEstimated = fminsearch(@CostFunction, -20);
    disp("Oszacowany kąt:");
    disp(angleDetEstimated);
end
