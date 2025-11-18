function angleDetEstimated = MLE_comp(Y, stepsToCompare, allElementPositions, freq)
    % Y - wektor danych pomiarowych z wirtualnego szyku [M_total x 1]
    % stepsToCompare - wektor z liczbą kroków SAR do porównania, np. [10, 50]
    % allElementPositions - wektor pozycji wszystkich wirtualnych anten w mm
    % freq - częstotliwość sygnału w Hz

    % Obliczenie długości fali w mm (potrzebne do a_sar)
    lambda_mm = 3e8 / freq * 1e3;

    % Przygotowanie wykresu
    figure;
    hold on;
    colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y']; % Paleta kolorów
    legendEntries = {};
    
    % Pełny zakres kątów do przeszukiwania
    angleVec = -45:0.1:45;
    
    % Definicja funkcji kosztu (lokalnie dla danej pętli)
    % Używamy a_sar zamiast a
    function J = CostFunctionLocal(angleDeg)
        aTemp = a_sar(angleDeg, pos_sub, lambda_mm);
        % Macierz projekcji na podprzestrzeń szumu
        % Pv = I - a * (a'a)^-1 * a'
        % Dla pojedynczego celu (a to wektor) (a'a) to skalar
        Pv = eye(M) - aTemp * ((aTemp' * aTemp)^(-1)) * aTemp';
        J = abs(trace(Pv * R));
    end

    % Pętla po wybranych liczbach kroków (stepsToCompare to liczba pozycji suwnicy)
    % Uwaga: Każdy krok suwnicy generuje 2 elementy antenowe (zgodnie z run_angle_estimation_sar)
    for i = 1:length(stepsToCompare)
        N_steps = stepsToCompare(i);
        
        % Liczba wirtualnych anten dla danej liczby kroków
        % W Twoim kodzie: 1 krok = 2 anteny (pos_1 i pos_2)
        M_antennas = N_steps * 2; 
        
        % Sprawdzenie czy mamy wystarczająco dużo danych
        if M_antennas > length(Y)
            warning('Wymagana liczba anten (%d) większa niż dostępna w danych (%d). Pomijam krok N=%d.', M_antennas, length(Y), N_steps);
            continue;
        end
        
        % Wybór podzbioru danych i pozycji anten
        % Y jest wektorem kolumnowym [M_total x 1], traktujemy to jako 1 snapshot
        Y_sub = Y(1:M_antennas);         % Sygnał z pierwszych M anten
        pos_sub = allElementPositions(1:M_antennas); % Pozycje tych anten
        
        % W przypadku SAR, mamy zazwyczaj jeden 'snapshot' (N=1), ale bardzo duży wektor obserwacji (M)
        [M, N_snapshots] = size(Y_sub); 
        
        % Obliczenie macierzy kowariancji (dla N=1 to po prostu iloczyn zewnętrzny)
        R = (Y_sub * Y_sub') / N_snapshots;
        
        
        
        % Obliczanie wartości funkcji kosztu dla całego wektora kątów
        pval = zeros(size(angleVec));
        for k = 1:length(angleVec)
            pval(k) = CostFunctionLocal(angleVec(k));
        end
        
        % Normalizacja wykresu do 1 dla czytelności porównania (opcjonalne)
        if max(pval) ~= 0
            pval = pval / max(pval);
        end

        % Rysowanie wykresu
        plot(angleVec, pval, 'LineWidth', 1.5, 'Color', colors(mod(i-1, length(colors))+1));
        legendEntries{end+1} = sprintf('%d kroków (%d anten)', N_steps, M_antennas);
        
        % Znalezienie minimum (estymacja) dla tego przypadku
        % Szukamy w okolicy prawdziwego kąta (-20) lub globalnie
        Jsq = @(x) CostFunctionLocal(x);
        
        % Proste przeszukiwanie minimum startując od -20 (lub 0)
        estAngle = fminsearch(Jsq, -20); 
        fprintf('Dla %d kroków (%d anten), estymowany kąt: %.4f stopni\n', N_steps, M_antennas, estAngle);
        
        % Zwracamy estymatę dla ostatniego przypadku
        if i == length(stepsToCompare)
            angleDetEstimated = estAngle;
        end
    end
    
    % Formatowanie wykresu
    % title('Porównanie funkcji kosztu MLE dla różnej długości apertury syntetycznej');
    xlabel('Kąt [deg]');
    ylabel('Znormalizowana funkcja kosztu J(\theta)');
    legend(legendEntries, 'Location', 'best');
    grid on;
    hold off;
end