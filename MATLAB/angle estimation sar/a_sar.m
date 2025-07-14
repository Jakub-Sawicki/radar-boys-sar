function steering = a_sar(PhiDeg, positions_mm, lambda_mm)
    % pozycje elementów w mm
    k = 2 * pi / lambda_mm; % liczba falowa
    steering = exp(1j * k * positions_mm(:) * sind(PhiDeg));
end