import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound
from adi import ad9361
from adi.cn0566 import CN0566

ESP32_IP = "192.168.0.103"
ESP32_PORT = 3333
MEASUREMENTS = 30  # ile kroków i pomiarów
STEP_SIZE_M = 0.00018  # rozmiar kroku w metrach (0.18mm)

def connect_esp32():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ESP32_IP, ESP32_PORT))
    s.settimeout(5)
    print("[INFO] Połączono z ESP32")
    return s

def send_step_and_wait(sock):
    sock.sendall(b"STEP_CCW\n")
    data = sock.recv(1024)
    if b"DONE_CCW" in data:
        print("[INFO] ESP32 wykonał krok")
    else:
        print("[WARN] Otrzymano:", data)

def acquire_data(sdr):
    samples = sdr.rx()
    ch0 = samples[0]
    ch1 = samples[1]
    return ch0, ch1

def analyze_angle_estimation(data_ch0, data_ch1, freq_hz=10.3943359375e9):
    """
    Podstawowa estymacja kąta przylotu sygnału używając różnicy fazowej
    """
    c = 3e8  # prędkość światła
    lambda_m = c / freq_hz  # długość fali w metrach
    d_m = 0.014  # rozstaw anten w metrach (14mm)
    d_over_lambda = d_m / lambda_m  # normalizowany rozstaw
    
    # Oblicz korelację między kanałami
    correlation = np.mean(data_ch0 * np.conj(data_ch1))

    calibrated_correlation = correlation# * np.exp(-1j * np.deg2rad(125))
    
    # Różnica fazowa między kanałami
    phase_diff = np.angle(calibrated_correlation)
    
    # Estymacja kąta na podstawie różnicy fazowej
    sin_theta = phase_diff / (2 * np.pi * d_over_lambda)
    
    # Sprawdź czy wartość jest w zakresie [-1, 1]
    if abs(sin_theta) > 1:
        sin_theta = np.clip(sin_theta, -1, 1)
    
    estimated_angle_rad = np.arcsin(sin_theta)
    estimated_angle_deg = np.rad2deg(estimated_angle_rad)
    
    return estimated_angle_deg, phase_diff, calibrated_correlation

def analyze_mle_method(measurment_data, freq_hz=10.3943359375e9):
    """
    Poprawiona implementacja MLE dla estymacji kąta
    """
    c = 3e8
    lambda_m = c / freq_hz
    d_m = 24.25e-3  # rozstaw anten w metrach
   
    # Wiele próbek
    ch0 = measurment_data['ch0']  # lista 3 arrayów
    ch1 = measurment_data['ch1']  # lista 3 arrayów

    # Połącz pomiary w jeden długi wektor na kanał
    ch0_combined = np.concatenate(ch0)  # shape (N,)
    ch1_combined = np.concatenate(ch1)  # shape (N,)

    # Stwórz macierz Y: każdy wiersz to jeden kanał
    Y = np.vstack([ch0_combined, ch1_combined])  # shape (2, N)

    # Teraz można policzyć macierz korelacyjną (lub macierz kowariancji)
    N = Y.shape[1]  # liczba pomiarów (czyli długość sygnału)
    M = 2

    R = np.dot(Y, Y.conj().T) / N 

    # measurment_data['positions'] = [x + d_m for x in measurment_data['positions']]
    positions_ch1 = [x + d_m for x in measurment_data['positions']]

    # print(measurment_data['positions'])

    def steering_vector(angle_deg, measurment_data):
        angle_rad = np.deg2rad(angle_deg)
        k = 2 * np.pi / lambda_m  # liczba falowa w metrach
        positions = np.array([measurment_data['positions'], measurment_data['positions'])  # pozycje anten
        print(positions)
        print("\n")
        return np.exp(1j * k * positions * np.sin(angle_rad)).reshape(-1, 1)
    
    def cost_function(angle_deg, measurment_data):
        a = steering_vector(angle_deg, measurment_data)
        aH = a.conj().T
        
        # Sprawdź czy macierz jest odwracalna
        aH_a = np.dot(aH, a)
        if np.abs(aH_a) < 1e-12:
            return np.inf
        
        inv_aH_a = 1.0 / aH_a[0, 0]  # dla macierzy 1x1
        P_perp = np.eye(M) - np.dot(a, aH) * inv_aH_a
        
        J = np.abs(np.trace(np.dot(P_perp, R)))
        return J
    
    steering_vector(5, measurment_data)
    # print(cost_function())
    
    # min_angle = -45
    # max_angle = 45
    
    # # Najpierw przeszukaj siatkę jak w MATLAB (co 0.1 stopnia)
    # angle_vec = np.arange(min_angle, max_angle + 0.1, 0.1)
    # pval = np.array([cost_function(angle) for angle in angle_vec])
    
    # # Znajdź najlepszy punkt startowy
    # best_grid_idx = np.argmin(pval)
    # best_grid_angle = angle_vec[best_grid_idx]
    
    # # Dokładna optymalizacja wokół najlepszego punktu (odpowiednik fminsearch)
    # try:
    #     result = fminbound(cost_function, 
    #                       max(min_angle, best_grid_angle - 5), 
    #                       min(max_angle, best_grid_angle + 5), 
    #                       xtol=1e-6)
    #     best_angle = result
    #     best_cost = cost_function(result)
    # except:
    #     best_angle = best_grid_angle
    #     best_cost = pval[best_grid_idx]
    
    # # Dodatkowa optymalizacja z różnych punktów startowych (jak w MATLAB)
    # start_points = [min_angle, min_angle + 10, min_angle + 20, min_angle + 30, max_angle]
    
    # for start_angle in start_points:
    #     try:
    #         result = fminbound(cost_function, min_angle, max_angle, xtol=1e-6)
    #         current_cost = cost_function(result)
    #         if current_cost < best_cost:
    #             best_cost = current_cost
    #             best_angle = result
    #     except:
    #         continue
    
    # return best_angle

def analyze_coordinate_estimation_method(measurements_data, positions, freq_hz=10.3943359375e9):
    """
    METODA 1: Estymacja współrzędnych z uwzględnieniem położenia radaru
    
    Args:
        measurements_data: lista krotek (ch0, ch1, angle_estimate) dla każdego pomiaru
        positions: lista pozycji radaru dla każdego pomiaru [m]
        freq_hz: częstotliwość sygnału
    
    Returns:
        estimated_target_coords: estymowane współrzędne celu (x, y)
        refined_angle: poprawiony kąt
    """
    print("[INFO] Analiza metodą estymacji współrzędnych...")
    
    # Zbierz estymaty kątów i pozycje
    angles = []
    radar_positions = []
    
    for i, (ch0, ch1, pos) in enumerate(zip(measurements_data['ch0'], 
                                          measurements_data['ch1'], 
                                          positions)):
        # Podstawowa estymacja kąta dla tego pomiaru
        angle_deg, _, _ = analyze_angle_estimation(ch0, ch1, freq_hz)
        angles.append(angle_deg)
        radar_positions.append(pos)
    
    angles = np.array(angles)
    radar_positions = np.array(radar_positions)
    
    # Estymuj współrzędne celu używając triangulacji
    # Załóżmy, że radar porusza się wzdłuż osi X, a cel jest w punkcie (target_x, target_y)
    
    # Metoda najmniejszych kwadratów dla estymacji pozycji celu
    def estimate_target_position(radar_pos, angles_deg):
        """Estymuj pozycję celu na podstawie pomiarów kątowych"""
        angles_rad = np.deg2rad(angles_deg)
        
        # Macierz dla rozwiązania liniowego
        A = []
        b = []
        
        for i in range(len(radar_pos)-1):
            x1, x2 = radar_pos[i], radar_pos[i+1]
            theta1, theta2 = angles_rad[i], angles_rad[i+1]
            
            # Równanie: y = tan(theta) * (target_x - radar_x)
            # Dla dwóch pomiarów: tan(theta1)*(target_x - x1) = tan(theta2)*(target_x - x2)
            
            if np.abs(np.cos(theta1)) > 1e-6 and np.abs(np.cos(theta2)) > 1e-6:
                tan1, tan2 = np.tan(theta1), np.tan(theta2)
                
                if np.abs(tan1 - tan2) > 1e-6:
                    # target_x = (tan2*x2 - tan1*x1) / (tan2 - tan1)
                    A.append([tan1 - tan2, -1])
                    b.append(tan1*x1 - tan2*x2)
        
        if len(A) < 2:
            # Fallback - użyj średniego kąta
            avg_angle = np.mean(angles_rad)
            avg_pos = np.mean(radar_pos)
            # Załóż cel w odległości 1m
            target_x = avg_pos + 1.0 * np.cos(avg_angle)
            target_y = 1.0 * np.sin(avg_angle)
            return target_x, target_y
        
        A = np.array(A)
        b = np.array(b)
        
        # Rozwiązanie metodą najmniejszych kwadratów
        try:
            coords = np.linalg.lstsq(A, b, rcond=None)[0]
            return coords[0], coords[1]  # target_x, target_y
        except:
            # Fallback
            avg_angle = np.mean(angles_rad)
            avg_pos = np.mean(radar_pos)
            target_x = avg_pos + 1.0 * np.cos(avg_angle)
            target_y = 1.0 * np.sin(avg_angle)
            return target_x, target_y
    
    # Estymuj pozycję celu
    target_x, target_y = estimate_target_position(radar_positions, angles)
    
    # Popraw estymaty kątów używając estymowanej pozycji celu
    refined_angles = []
    for pos in radar_positions:
        # Oblicz rzeczywisty kąt do celu z tej pozycji
        dx = target_x - pos
        dy = target_y
        refined_angle = np.rad2deg(np.arctan2(dy, dx))
        refined_angles.append(refined_angle)
    
    # Ostateczna estymacja kąta jako średnia ważona
    weights = np.exp(-np.abs(angles - np.mean(angles)))  # wagi oparte na odchyleniu
    final_angle = np.average(refined_angles, weights=weights)
    
    return (target_x, target_y), final_angle, refined_angles

def analyze_synthetic_aperture_method(measurements_data, positions, freq_hz=10.3943359375e9):
    """
    METODA 2: Syntetic Aperture - "udawana duża antena"
    
    Używa wszystkich pomiarów do stworzenia wirtualnej dużej anteny
    """
    print("[INFO] Analiza metodą Synthetic Aperture...")
    
    c = 3e8
    lambda_m = c / freq_hz
    
    # Zbierz wszystkie sygnały
    all_signals_ch0 = []
    all_signals_ch1 = []
    
    for ch0, ch1 in zip(measurements_data['ch0'], measurements_data['ch1']):
        all_signals_ch0.append(ch0)
        all_signals_ch1.append(ch1)
    
    # Stwórz wirtualną antenę z N pozycji
    N_positions = len(positions)
    virtual_aperture_size = positions[-1] - positions[0]  # całkowity rozmiar apertura
    
    print(f"[INFO] Rozmiar wirtualnej apertury: {virtual_aperture_size*100:.1f} cm")
    print(f"[INFO] Liczba pozycji: {N_positions}")
    
    # Beamforming dla różnych kątów
    angles_to_test = np.arange(-90, 91, 1)  # kąty co 1 stopień
    beamformer_output = []
    
    for angle_deg in angles_to_test:
        angle_rad = np.deg2rad(angle_deg)
        
        # Oblicz opóźnienia dla każdej pozycji
        beam_sum = 0
        total_power = 0
        
        for i, pos in enumerate(positions):
            # Opóźnienie względem pozycji referencyjnej (pierwszej)
            delay_samples = int(2 * pos * np.sin(angle_rad) / c * 30e6)  # sample rate = 30MHz
            
            # Pobierz sygnały z odpowiednim opóźnieniem
            ch0_delayed = all_signals_ch0[i]
            ch1_delayed = all_signals_ch1[i]
            
            # Suma sygnałów z dwóch kanałów
            signal_sum = ch0_delayed + ch1_delayed
            
            # Dodaj do wiązki z odpowiednim przesunięciem fazowym
            phase_shift = np.exp(1j * 2 * np.pi * pos * np.sin(angle_rad) / lambda_m)
            beam_sum += np.mean(signal_sum) * phase_shift
            total_power += np.mean(np.abs(signal_sum)**2)
        
        # Moc wiązki dla tego kąta
        beam_power = np.abs(beam_sum)**2 / total_power if total_power > 0 else 0
        beamformer_output.append(beam_power)
    
    beamformer_output = np.array(beamformer_output)
    
    # Znajdź kąt z maksymalną mocą
    max_idx = np.argmax(beamformer_output)
    estimated_angle = angles_to_test[max_idx]
    
    # Popraw estymację używając interpolacji wokół maksimum
    if 0 < max_idx < len(angles_to_test)-1:
        # Interpolacja paraboliczna
        y1, y2, y3 = beamformer_output[max_idx-1:max_idx+2]
        x1, x2, x3 = angles_to_test[max_idx-1:max_idx+2]
        
        # Znajdź maksimum paraboli
        denom = (x1-x2)*(x1-x3)*(x2-x3)
        if abs(denom) > 1e-10:
            A = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denom
            if A < 0:  # maksimum istnieje
                estimated_angle = (x1*x1*(y2-y3) + x2*x2*(y3-y1) + x3*x3*(y1-y2)) / (2*denom)
    
    return estimated_angle, beamformer_output, angles_to_test

def analyze_mle_sar_method(measurements_data, positions, freq_hz=10.3943359375e9):
    """
    METODA 3: MLE dla Synthetic Aperture Radar - zgodna z MATLAB
    """
    print("[INFO] Analiza metodą MLE SAR...")
    
    c = 3e8
    lambda_m = c / freq_hz
    d_m = 0.014  # bazowy rozstaw anten
    
    # Zbuduj macierz pomiarową Y i pozycje elementów (jak w MATLAB)
    virtualArray = []
    elementPositions = []
    
    for i, (ch0, ch1, pos) in enumerate(zip(measurements_data['ch0'], 
                                          measurements_data['ch1'], 
                                          positions)):
        # Pozycje anten dla tej iteracji (w metrach)
        # Każda pozycja ma 2 anteny: pos i pos+d_m
        pos_k = pos + np.array([0, d_m])
        elementPositions.extend(pos_k)
        
        # Dodaj pomiar - każdy pomiar to 2 anteny
        measure = np.array([ch0, ch1])
        virtualArray.append(measure)
    
    # Przekształć do macierzy jak w MATLAB
    Y = np.array(virtualArray).T  # M x N gdzie M = liczba elementów, N = liczba pomiarów
    elementPositions = np.array(elementPositions)
    
    M, N = Y.shape
    print(f"[INFO] Macierz Y: {M} x {N}")
    print(f"[INFO] Liczba elementów: {len(elementPositions)}")
    
    # Macierz korelacji
    R = np.dot(Y, Y.conj().T) / N
    
    def steering_vector_sar(angle_deg):
        """Wektor kierunkowy dla SAR - zgodny z MATLAB a_sar"""
        angle_rad = np.deg2rad(angle_deg)
        k = 2 * np.pi / lambda_m
        # Dokładnie jak w MATLAB: exp(1j * k * positions_mm(:) * sind(PhiDeg))
        return np.exp(1j * k * elementPositions * np.sin(angle_rad)).reshape(-1, 1)
    
    def cost_function_sar(angle_deg):
        """Funkcja kosztu MLE dla SAR - dokładnie jak w MATLAB"""
        a = steering_vector_sar(angle_deg)
        aH = a.conj().T
        
        aH_a = np.dot(aH, a)
        if np.abs(aH_a) < 1e-12:
            return np.inf
        
        # Projektor ortogonalny - dokładnie jak w MATLAB
        # Pv = eye(M) - aTemp * ((aTemp') * aTemp)^(-1) * aTemp'
        inv_aH_a = 1.0 / aH_a[0, 0]
        P_perp = np.eye(M) - np.dot(a, aH) * inv_aH_a
        
        # Funkcja kosztu - abs(trace(Pv * R))
        J = np.abs(np.trace(np.dot(P_perp, R)))
        return J
    
    # Optymalizacja dokładnie jak w MATLAB
    min_angle = -45
    max_angle = 45
    
    # Przeszukiwanie siatki co 0.1 stopnia
    angle_vec = np.arange(min_angle, max_angle + 0.1, 0.1)
    pval = np.array([cost_function_sar(angle) for angle in angle_vec])
    
    # Znajdź najlepszy punkt z siatki
    best_grid_idx = np.argmin(pval)
    best_grid_angle = angle_vec[best_grid_idx]
    
    # Dokładna optymalizacja (odpowiednik fminsearch)
    try:
        result = fminbound(cost_function_sar, 
                          max(min_angle, best_grid_angle - 5), 
                          min(max_angle, best_grid_angle + 5), 
                          xtol=1e-6)
        best_angle = result
        best_cost = cost_function_sar(result)
    except:
        best_angle = best_grid_angle
        best_cost = pval[best_grid_idx]
    
    # Dodatkowa optymalizacja z różnych punktów startowych
    start_points = [min_angle, -20, 0, 20, max_angle]
    
    for start_angle in start_points:
        try:
            result = fminbound(cost_function_sar, min_angle, max_angle, xtol=1e-6)
            current_cost = cost_function_sar(result)
            if current_cost < best_cost:
                best_cost = current_cost
                best_angle = result
        except:
            continue
    
    return best_angle, pval, angle_vec

def main():
    print("===============================")
    print("    RADAR CN0566 + ESP32       ")
    print("  Rozszerzone metody estymacji  ")
    print("  Z POPRAWIONĄ IMPLEMENTACJĄ MLE")
    print("===============================")
    
    # Połączenie z urządzeniami
    try:
        print("[INFO] Próba połączenia z CN0566...")
        phaser = CN0566(uri="ip:phaser.local")
        sdr = ad9361(uri="ip:phaser.local:50901")
    except:
        print("[INFO] Próba połączenia po localhost")
        phaser = CN0566(uri="ip:localhost")
        sdr = ad9361(uri="ip:192.168.2.1")
    
    phaser.sdr = sdr
    
    # Konfiguracja urządzeń (bez zmian)
    time.sleep(0.5)
    phaser.configure(device_mode="rx")
    sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
    sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
    sdr._ctrl.debug_attrs["initialize"].value = "1"
    
    sdr.rx_enabled_channels = [0, 1]
    sdr._rxadc.set_kernel_buffers_count(1)
    rx = sdr._ctrl.find_channel("voltage0")
    rx.attrs["quadrature_tracking_en"].value = "1"
    
    # Parametry próbkowania
    sdr.sample_rate = int(30e6)
    sdr.rx_buffer_size = int(4 * 1024)
    sdr.rx_rf_bandwidth = int(10e6)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = 0
    sdr.rx_hardwaregain_chan1 = 0
    sdr.rx_lo = int(2.0e9)
    sdr.filter = "LTE20_MHz.ftr"
    
    # Wyłącz nadawanie
    sdr.tx_hardwaregain_chan0 = -80
    sdr.tx_hardwaregain_chan1 = -80
    
    # Konfiguracja phasera
    phaser.SignalFreq = 10.3943359375e9
    phaser.lo = int(phaser.SignalFreq) + sdr.rx_lo
    
    # Ustawienie wzmocnienia na wszystkich 8 antenach
    gain_list = [64] * 8
    for i in range(len(gain_list)):
        phaser.set_chan_gain(i, gain_list[i], apply_cal=False)
    
    phaser.set_beam_phase_diff(0.0)
    phaser.Averages = 16
    
    # Wyświetl parametry systemu
    c = 3e8
    lambda_m = c / phaser.SignalFreq
    d_m = 24.25e-3
    d_over_lambda = d_m / lambda_m
    
    print(f"[INFO] Częstotliwość: {phaser.SignalFreq/1e9:.3f} GHz")
    print(f"[INFO] Długość fali: {lambda_m*100:.2f} cm")
    print(f"[INFO] Rozstaw anten: {d_m*100:.1f} cm")
    print(f"[INFO] Rozmiar kroku: {STEP_SIZE_M*1000:.1f} mm")
    print(f"[INFO] d/λ = {d_over_lambda:.3f}")
    
    time.sleep(0.5)
    
    # Połączenie z ESP32
    sock = connect_esp32()
    time.sleep(1)
    
    # Struktury danych dla nowych metod
    measurements_data = {
        'ch0': [],
        'ch1': [],
        'positions': []
    }
    
    print(f"\n[INFO] Rozpoczynam {MEASUREMENTS} pomiarów...")
    
    # Główna pętla pomiarowa
    for i in range(MEASUREMENTS):
        print(f"\n[{i+1}/{MEASUREMENTS}] Pomiar...")
        
        # Wykonaj krok na ESP32
        if i > 0:  # nie rób kroku przy pierwszym pomiarze
            send_step_and_wait(sock)
            time.sleep(0.3)
        
        # Oblicz aktualną pozycję
        current_position = i * STEP_SIZE_M
        
        # Pobierz dane z radaru
        ch0, ch1 = acquire_data(sdr)
        # ch0, ch1 = [21.1706135741686 - 0.888916916651944j, 19.8632052111244 - 4.35960460501996j]
        
        # Podstawowe analizy
        # angle_basic, _, _ = analyze_angle_estimation(ch0, ch1, phaser.SignalFreq)
        # angle_mle = analyze_mle_method(ch0, ch1, phaser.SignalFreq, current_position)  # POPRAWIONA IMPLEMENTACJA

        # Zapisz dane do struktur
        measurements_data['ch0'].append(ch0)
        measurements_data['ch1'].append(ch1)
        measurements_data['positions'].append(current_position)
        # measurements_data['angles_basic'].append(angle_basic)
        # measurements_data['angles_mle'].append(angle_mle)
        
        print(f"[INFO] Pozycja: {current_position*100:.1f} cm")
        # print(f"[INFO] Kąt podstawowy: {angle_basic:.1f}°")
        # print(f"[INFO] Kąt MLE (poprawiony): {angle_mle:.1f}°")
        
        time.sleep(0.1)
    
    sock.close()
    print("\n[INFO] Pomiary zakończone, rozpoczynam analizę...")

    angle_mle = analyze_mle_method(measurements_data, phaser.SignalFreq)
    
    # # ANALIZA METODĄ 1: Estymacja współrzędnych
    # target_coords, refined_angle, refined_angles = analyze_coordinate_estimation_method(
    #     measurements_data, measurements_data['positions'], phaser.SignalFreq
    # )
    
    # # ANALIZA METODĄ 2: Synthetic Aperture
    # sa_angle, beamformer_output, test_angles = analyze_synthetic_aperture_method(
    #     measurements_data, measurements_data['positions'], phaser.SignalFreq
    # )
    
    # # ANALIZA METODĄ 3: MLE SAR
    # mle_sar_angle, mle_sar_pval, mle_sar_angles = analyze_mle_sar_method(
    #     measurements_data, measurements_data['positions'], phaser.SignalFreq
    # )
    
    # # Wyniki
    # print(f"\n===============================")
    # print(f"        WYNIKI ANALIZ          ")
    # print(f"===============================")
    # print(f"Średni kąt podstawowy: {np.mean(measurements_data['angles_basic']):.1f}°")
    # print(f"Średni kąt MLE (poprawiony): {np.mean(measurements_data['angles_mle']):.1f}°")
    # print(f"Estymowane współrzędne celu: ({target_coords[0]:.3f}, {target_coords[1]:.3f}) m")
    # print(f"Kąt z estymacji współrzędnych: {refined_angle:.1f}°")
    # print(f"Kąt z Synthetic Aperture: {sa_angle:.1f}°")
    # print(f"Kąt z MLE SAR: {mle_sar_angle:.1f}°")
    
    # # Wykresy
    # plt.figure(figsize=(20, 15))
    
    # # 1. Kąty w czasie
    # plt.subplot(2, 3, 1)
    # plt.plot(measurements_data['angles_basic'], 'b-o', label='Podstawowy', markersize=3)
    # plt.plot(measurements_data['angles_mle'], 'r-s', label='MLE', markersize=3)
    # plt.plot(refined_angles, 'g--', label='Estymacja współrzędnych')
    # plt.axhline(y=sa_angle, color='purple', linestyle=':', linewidth=2, label='Synthetic Aperture')
    # plt.title("Porównanie wszystkich metod")
    # plt.xlabel("Pomiar")
    # plt.ylabel("Kąt [°]")
    # plt.legend()
    # plt.grid(True)
    
    # # 2. Beamformer pattern
    # plt.subplot(2, 3, 2)
    # plt.plot(test_angles, 10*np.log10(beamformer_output + 1e-10), 'purple', linewidth=2)
    # plt.axvline(x=sa_angle, color='red', linestyle='--', label=f'Maksimum: {sa_angle:.1f}°')
    # plt.title("Wzorzec wiązki Synthetic Aperture")
    # plt.xlabel("Kąt [°]")
    # plt.ylabel("Moc [dB]")
    # plt.legend()
    # plt.grid(True)
    
    # # 3. Pozycja celu
    # plt.subplot(2, 3, 3)
    # plt.plot(measurements_data['positions'], np.zeros_like(measurements_data['positions']), 'bo-', label='Pozycje radaru')
    # plt.plot(target_coords[0], target_coords[1], 'r*', markersize=15, label=f'Cel ({target_coords[0]:.2f}, {target_coords[1]:.2f})')
    # plt.title("Estymowana pozycja celu")
    # plt.xlabel("X [m]")
    # plt.ylabel("Y [m]")
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    
    # # 4. Histogram wszystkich kątów
    # plt.subplot(2, 3, 4)
    # plt.hist(measurements_data['angles_basic'], bins=15, alpha=0.7, label='Podstawowy', density=True)
    # plt.hist(measurements_data['angles_mle'], bins=15, alpha=0.7, label='MLE', density=True)
    # plt.axvline(x=refined_angle, color='green', linestyle='--', label='Estym. współrzędne')
    # plt.axvline(x=sa_angle, color='purple', linestyle=':', label='Synthetic Aperture')
    # plt.title("Rozkład estymowanych kątów")
    # plt.xlabel("Kąt [°]")
    # plt.ylabel("Gęstość")
    # plt.legend()
    # plt.grid(True)
    
    # # 5. Błędy estymacji
    # plt.subplot(2, 3, 5)
    # basic_errors = np.array(measurements_data['angles_basic']) - refined_angle
    # mle_errors = np.array(measurements_data['angles_mle']) - refined_angle
    # plt.plot(basic_errors, 'b-', label='Błąd podstawowy', alpha=0.7)
    # plt.plot(mle_errors, 'r-', label='Błąd MLE', alpha=0.7)
    # plt.title("Błędy względem estymacji współrzędnych")
    # plt.xlabel("Pomiar")
    # plt.ylabel("Błąd [°]")
    # plt.legend()
    # plt.grid(True)
    
    # # 6. Stabilność metod
    # plt.subplot(2, 3, 6)
    # window_size = 20
    # if len(measurements_data['angles_basic']) >= window_size:
    #     basic_std = [np.std(measurements_data['angles_basic'][i:i+window_size]) 
    #                 for i in range(len(measurements_data['angles_basic'])-window_size)]
    #     mle_std = [np.std(measurements_data['angles_mle'][i:i+window_size]) 
    #               for i in range(len(measurements_data['angles_mle'])-window_size)]
    #     plt.plot(basic_std, 'b-', label='Stabilność podstawowa')
    #     plt.plot(mle_std, 'r-', label='Stabilność MLE')
    #     plt.title(f"Stabilność metod (okno {window_size})")
    #     plt.xlabel("Pomiar")
    #     plt.ylabel("Odchylenie std [°]")
    #     plt.legend()
    #     plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    
    # Zapisz wyniki
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # filename = f"radar_enhanced_{timestamp}.txt"
    
    # with open(filename, 'w') as f:
    #     f.write("# Pomiary radaru CN0566 - rozszerzone metody\n")
    #     f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    #     f.write(f"# Częstotliwość: {phaser.SignalFreq/1e9:.3f} GHz\n")
    #     f.write(f"# Rozmiar kroku: {STEP_SIZE_M*1000:.1f} mm\n")
    #     f.write(f"# Estymowane współrzędne celu: ({target_coords[0]:.3f}, {target_coords[1]:.3f}) m\n")
    #     f.write(f"# Kąt z estymacji współrzędnych: {refined_angle:.2f}°\n")
    #     f.write(f"# Kąt z Synthetic Aperture: {sa_angle:.2f}°\n")
    #     f.write("# Kolumny: Pomiar, Pozycja[m], Kąt_Podstawowy, Kąt_MLE, Kąt_Estym_Współrz, Kąt_SA\n")
        
    #     for i in range(len(measurements_data['angles_basic'])):
    #         f.write(f"{i+1:3d} {measurements_data['positions'][i]:8.5f} ")
    #         f.write(f"{measurements_data['angles_basic'][i]:7.2f} ")
    #         f.write(f"{measurements_data['angles_mle'][i]:7.2f} ")
    #         f.write(f"{refined_angles[i]:7.2f} {sa_angle:7.2f}\n")
    
    # print(f"\n[INFO] Wyniki zapisane do: {filename}")

if __name__ == "__main__":
    main()