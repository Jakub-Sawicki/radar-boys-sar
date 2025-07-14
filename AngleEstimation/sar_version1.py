import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound
from adi import ad9361
from adi.cn0566 import CN0566

ESP32_IP = "192.168.0.103"
ESP32_PORT = 3333
MEASUREMENTS = 100  # liczba pozycji (jak numSteps w MATLAB)
STEP_SIZE_M = 0.0005  # rozmiar kroku w metrach (0.5mm jak w MATLAB)

def connect_esp32():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ESP32_IP, ESP32_PORT))
    s.settimeout(5)
    print("[INFO] Połączono z ESP32")
    return s

def send_step_and_wait(sock):
    sock.sendall(b"STEP_CW\n")
    sock.sendall(b"STEP_CW\n")
    sock.sendall(b"STEP_CW\n")
    data = sock.recv(1024)
    if b"DONE_CW" in data:
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
    
    # Różnica fazowa między kanałami
    phase_diff = np.angle(correlation)
    
    # Estymacja kąta na podstawie różnicy fazowej
    sin_theta = phase_diff / (2 * np.pi * d_over_lambda)
    
    # Sprawdź czy wartość jest w zakresie [-1, 1]
    if abs(sin_theta) > 1:
        sin_theta = np.clip(sin_theta, -1, 1)
    
    estimated_angle_rad = np.arcsin(sin_theta)
    estimated_angle_deg = np.rad2deg(estimated_angle_rad)
    
    return estimated_angle_deg, phase_diff, correlation

def a_sar(phi_deg, positions_mm, lambda_mm):
    """
    Funkcja sterująca SAR - odpowiednik funkcji a_sar z MATLAB
    
    Args:
        phi_deg: kąt w stopniach
        positions_mm: pozycje elementów w milimetrach
        lambda_mm: długość fali w milimetrach
    
    Returns:
        steering: wektor sterujący
    """
    k = 2 * np.pi / lambda_mm  # liczba falowa
    positions_mm = np.array(positions_mm)
    steering = np.exp(1j * k * positions_mm * np.sin(np.deg2rad(phi_deg)))
    return steering.reshape(-1, 1)

def get_measure_sar(target_deg, ampl, element_positions_mm, lambda_mm, noise_level=0.1):
    """
    Symulacja pomiaru SAR - odpowiednik getMeasure_sar z MATLAB
    
    Args:
        target_deg: kąt celu w stopniach
        ampl: amplituda sygnału
        element_positions_mm: pozycje elementów w mm
        lambda_mm: długość fali w mm
        noise_level: poziom szumu
    
    Returns:
        out: pomiar z każdego elementu
    """
    steering = a_sar(target_deg, element_positions_mm, lambda_mm)
    out = steering * ampl
    
    # Dodaj szum (odpowiednik crandn w MATLAB)
    noise = (np.random.randn(len(element_positions_mm)) + 
             1j * np.random.randn(len(element_positions_mm))) / np.sqrt(2)
    noise = noise.reshape(-1, 1) * noise_level
    
    out = out + noise
    return out.flatten()

def mle_sar(Y, element_positions_mm, lambda_mm):
    """
    MLE dla SAR - odpowiednik MLE_sar z MATLAB
    
    Args:
        Y: macierz pomiarów [M x N] gdzie M=liczba elementów, N=liczba próbek
        element_positions_mm: pozycje wszystkich elementów w mm
        lambda_mm: długość fali w mm
    
    Returns:
        angle_estimated: estymowany kąt
        cost_values: wartości funkcji kosztu
        angle_vec: wektor kątów testowych
    """
    M_total, N = Y.shape # Zmieniono M na M_total, aby odzwierciedlić całkowitą liczbę wirtualnych elementów
    print(f"[INFO] MLE SAR: M_total={M_total} elementów, N={N} próbek")
    
    # Macierz korelacji
    R = np.dot(Y, Y.conj().T) / N
    
    def cost_function(angle_deg):
        """Funkcja kosztu MLE"""
        a_temp = a_sar(angle_deg, element_positions_mm, lambda_mm)
        
        # Oblicz projektor ortogonalny
        aH_a = np.dot(a_temp.conj().T, a_temp)
        
        if np.abs(aH_a) < 1e-12:
            return np.inf
        
        # P_perp = I - a*(a^H*a)^(-1)*a^H
        # Ważne: np.dot(a_temp, a_temp.conj().T) powinno być macierzą (M_total x M_total)
        # aH_a jest skalarem, więc dzielenie jest prawidłowe.
        P_perp = np.eye(M_total) - np.dot(a_temp, a_temp.conj().T) / aH_a
        
        # Funkcja kosztu
        J = np.abs(np.trace(np.dot(P_perp, R)))
        return J
    
    # Testuj kąty (jak w MATLAB)
    min_angle = -45
    max_angle = 45
    angle_vec = np.arange(min_angle, max_angle + 0.1, 0.1)
    cost_values = np.zeros(len(angle_vec))
    
    for i, angle in enumerate(angle_vec):
        cost_values[i] = cost_function(angle)
    
    # Znajdź minimum (jak w MATLAB)
    min_idx = np.argmin(cost_values)
    initial_guess = angle_vec[min_idx]
    
    # Dokładniejsza optymalizacja
    result = minimize_scalar(cost_function, bounds=(min_angle, max_angle), method='bounded')
    angle_estimated = result.x
    
    # Alternatywnie, użyj strategii z MATLAB
    angle_estimated_alt = initial_guess
    min_cost = cost_values[min_idx]
    
    for angle in range(min_angle + 10, max_angle + 1, 10):
        if cost_function(angle) < min_cost:
            try:
                result_local = minimize_scalar(cost_function, bounds=(angle-5, angle+5), method='bounded')
                if result_local.fun < min_cost:
                    angle_estimated_alt = result_local.x
                    min_cost = result_local.fun
            except:
                pass
    
    print(f"[INFO] Kąt estymowany (scipy): {angle_estimated:.2f}°")
    print(f"[INFO] Kąt estymowany (MATLAB style): {angle_estimated_alt:.2f}°")
    
    return angle_estimated, cost_values, angle_vec

def analyze_synthetic_aperture_mle(measurements_data, positions, freq_hz=10.3943359375e9):
    """
    Implementacja SAR MLE na podstawie kodu MATLAB
    
    Args:
        measurements_data: dane pomiarowe
        positions: pozycje radaru w metrach
        freq_hz: częstotliwość w Hz
    
    Returns:
        estimated_angle: estymowany kąt
        cost_values: wartości funkcji kosztu
        angle_vec: wektor testowych kątów
    """
    print("[INFO] Analiza SAR MLE (styl MATLAB)...")
    
    c = 3e8
    lambda_m = c / freq_hz
    lambda_mm = lambda_m * 1000  # w milimetrach
    d_mm = 0.014 * 1000  # rozstaw anten w mm (14mm) - używamy stałego rozstawu 14mm, nie lambda/2
    
    # Parametry jak w MATLAB
    # M_per_position: liczba fizycznych elementów w każdej pozycji pomiarowej (np. 2 dla CN0566)
    M_per_position = 2  
    
    # Buduj wirtualną antenę
    virtual_array_measurements = []
    element_positions = []
    
    for k, pos_m in enumerate(positions):
        shift_mm = pos_m * 1000  # konwertuj na mm
        
        # Pozycje anten dla tej iteracji (jak w MATLAB)
        # Zmiana: uwzględniamy fizyczny rozstaw anten d_mm
        pos_k = shift_mm + np.arange(M_per_position) * d_mm
        element_positions.extend(pos_k)
        
        # Pobierz pomiar dla tej pozycji
        ch0 = measurements_data['ch0'][k]
        ch1 = measurements_data['ch1'][k]
        
        # Stwórz "pomiar" z dwóch kanałów (średnie wartości)
        # To jest "próbka" dla tej pozycji, składająca się z danych z obu kanałów
        measure = np.array([np.mean(ch0), np.mean(ch1)])
        virtual_array_measurements.append(measure)
    
    # Konwertuj do macierzy [M_total x N]
    # Y w MLE_sar powinno mieć wymiary [liczba_wirtualnych_elementów x liczba_pomiarów_w_czasie]
    # Tutaj N to liczba pomiarów (MEASUREMENTS), a M_total to MEASUREMENTS * M_per_position
    
    # Transponujemy, aby mieć M_total w wierszach (liczba wirtualnych elementów)
    # i N w kolumnach (liczba pomiarów dla każdego wirtualnego elementu)
    # Ale 'measure' w pętli jest średnią, więc mamy [2] dla każdej pozycji.
    # Potrzebujemy struktury, gdzie każdy "wirtualny element" (czyli każda para kanałów dla każdej pozycji)
    # jest traktowany jako oddzielny element anteny.
    
    # Poprawka w konstrukcji virtual_array
    # Zamiast agregować średnie, musimy traktować każdą parę (ch0, ch1) z każdej pozycji
    # jako oddzielne "elementy" wirtualnej anteny.
    
    # Zmieniamy virtual_array_measurements w macierz, gdzie każdy wiersz to pomiar z konkretnego wirtualnego elementu,
    # a kolumny to "próbki" w czasie. Ponieważ bierzemy średnią w acquire_data, każda "próbka" jest pojedynczą wartością.
    # W kontekście SAR, każda fizyczna pozycja i każdy kanał na tej pozycji staje się wirtualnym elementem.
    
    # virtual_array powinno być [M_total x 1] jeśli każda kolumna Y reprezentuje pojedynczy pomiar
    # Jeśli Y jest macierzą korelacji, to powinno być [M_total x M_total]
    
    # Poprawka: Y powinien być macierzą [M_total x N_samples_per_measurement]
    # Ale w tym kodzie "N" jest liczbą pozycji pomiarowych (MEASUREMENTS).
    # Funkcja analyze_angle_estimation zwraca kąt na podstawie uśrednionych danych.
    # W przypadku MLE SAR, Y to surowe dane z każdego elementu/pozycji.
    
    # Zmiana: Zamiast uśredniać w `virtual_array.append(measure)`,
    # potrzebujemy macierzy danych, gdzie każda wirtualna antena (czyli każdy kanał z każdej pozycji)
    # ma swoje surowe dane.
    
    # Zatem Y będzie miało wymiary (MEASUREMENTS * M_per_position) x (sdr.rx_buffer_size)
    # Gdzie M_total = MEASUREMENTS * M_per_position
    
    # Rebuilding virtual_array_data
    virtual_array_data = []
    full_element_positions = [] # Pozycje dla każdego wirtualnego elementu
    
    # Oblicz fizyczny rozstaw anten w mm (14mm dla CN0566)
    # Ten rozstaw jest stały między CH0 a CH1 na tej samej pozycji
    fixed_antenna_spacing_mm = 14.0 # Zgodnie z datasheet CN0566
    
    for k, pos_m in enumerate(positions):
        shift_mm = pos_m * 1000  # konwertuj na mm
        
        # Pozycja pierwszego kanału (CH0) na obecnej pozycji pomiarowej
        ch0_pos = shift_mm
        # Pozycja drugiego kanału (CH1) na obecnej pozycji pomiarowej
        ch1_pos = shift_mm + fixed_antenna_spacing_mm
        
        full_element_positions.append(ch0_pos)
        full_element_positions.append(ch1_pos)
        
        # Dodaj surowe dane z kanału 0 i kanału 1 do listy
        virtual_array_data.append(measurements_data['ch0'][k])
        virtual_array_data.append(measurements_data['ch1'][k])
    
    # Konwertuj listę danych do macierzy numpy
    # virtual_array_data będzie listą tablic numpy (każda z rx_buffer_size próbek)
    # Musimy je spakować w jedną macierz Y, gdzie wiersze to wirtualne elementy, a kolumny to próbki
    Y_sar = np.array(virtual_array_data) # [M_total x rx_buffer_size]
    
    full_element_positions = np.array(full_element_positions)
    
    print(f"[INFO] Wirtualna antena: {Y_sar.shape[0]} elementów, {Y_sar.shape[1]} próbek na element")
    print(f"[INFO] Całkowita apertura: {(full_element_positions[-1] - full_element_positions[0]):.1f} mm")
    print(f"[INFO] Rozstaw elementów fizycznych (CH0-CH1): {fixed_antenna_spacing_mm:.2f} mm")
    
    # Wywołaj MLE SAR
    # Y_sar zawiera teraz surowe dane dla każdego wirtualnego elementu
    # full_element_positions zawiera dokładne pozycje każdego wirtualnego elementu
    estimated_angle, cost_values, angle_vec = mle_sar(Y_sar, full_element_positions, lambda_mm)
    
    return estimated_angle, cost_values, angle_vec, full_element_positions


def main():
    print("===============================")
    print("    RADAR CN0566 + ESP32       ")
    print("     SAR MLE (MATLAB style)    ")
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
    
    # Konfiguracja urządzeń
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
    lambda_mm = lambda_m * 1000
    # d_mm = lambda_mm / 2 # To nie jest rozstaw fizyczny anten CN0566!
    d_mm_physical = 14.0 # Fizyczny rozstaw anten w CN0566 to 14 mm
    
    print(f"[INFO] Częstotliwość: {phaser.SignalFreq/1e9:.3f} GHz")
    print(f"[INFO] Długość fali: {lambda_m*100:.2f} cm ({lambda_mm:.2f} mm)")
    print(f"[INFO] Rozstaw fizyczny anten (CH0-CH1): {d_mm_physical:.2f} mm")
    print(f"[INFO] Rozmiar kroku: {STEP_SIZE_M*1000:.1f} mm")
    
    time.sleep(0.5)
    
    # Połączenie z ESP32
    sock = connect_esp32()
    time.sleep(1)
    
    # Struktury danych
    measurements_data = {
        'ch0': [],
        'ch1': [],
        'positions': [],
        'angles_basic': []
    }
    
    print(f"\n[INFO] Rozpoczynam {MEASUREMENTS} pomiarów...")
    
    # Główna pętla pomiarowa (jak w MATLAB)
    for k in range(MEASUREMENTS):
        print(f"\n[{k+1}/{MEASUREMENTS}] Pomiar na pozycji {k}...")
        
        # Wykonaj krok na ESP32
        if k > 0:
            send_step_and_wait(sock)
            time.sleep(0.3)
        
        # Oblicz aktualną pozycję
        current_position = k * STEP_SIZE_M
        
        # Pobierz dane z radaru
        ch0, ch1 = acquire_data(sdr)
        
        # Podstawowa analiza (na podstawie jednej pary kanałów)
        angle_basic, _, _ = analyze_angle_estimation(ch0, ch1, phaser.SignalFreq)
        
        # Zapisz dane
        measurements_data['ch0'].append(ch0)
        measurements_data['ch1'].append(ch1)
        measurements_data['positions'].append(current_position)
        measurements_data['angles_basic'].append(angle_basic)
        
        print(f"[INFO] Pozycja: {current_position*1000:.1f} mm")
        print(f"[INFO] Kąt podstawowy: {angle_basic:.1f}°")
        
        time.sleep(0.1)
    
    sock.close()
    print("\n[INFO] Pomiary zakończone, rozpoczynam analizę SAR MLE...")
    
    # ANALIZA SAR MLE
    sar_angle, cost_values, angle_vec, element_positions = analyze_synthetic_aperture_mle(
        measurements_data, measurements_data['positions'], phaser.SignalFreq
    )
    
    # Wyniki
    print(f"\n===============================")
    print(f"        WYNIKI ANALIZ          ")
    print(f"===============================")
    print(f"Średni kąt podstawowy: {np.mean(measurements_data['angles_basic']):.1f}°")
    print(f"Kąt SAR MLE: {sar_angle:.1f}°")
    print(f"Całkowita apertura: {(element_positions[-1] - element_positions[0]):.1f} mm")
    print(f"Liczba elementów wirtualnych: {len(element_positions)}")
    
    # Wykresy
    plt.figure(figsize=(18, 12))
    
    # 1. Kąty w czasie
    plt.subplot(2, 3, 1)
    plt.plot(measurements_data['angles_basic'], 'b-o', label='Podstawowy', markersize=4)
    plt.axhline(y=sar_angle, color='red', linestyle='--', linewidth=2, label=f'SAR MLE: {sar_angle:.1f}°')
    plt.title("Porównanie metod estymacji kąta")
    plt.xlabel("Pozycja pomiaru")
    plt.ylabel("Kąt [°]")
    plt.legend()
    plt.grid(True)
    
    # 2. Funkcja kosztu MLE (jak w MATLAB)
    plt.subplot(2, 3, 2)
    plt.plot(angle_vec, cost_values, 'purple', linewidth=2)
    plt.axvline(x=sar_angle, color='red', linestyle='--', label=f'Minimum: {sar_angle:.1f}°')
    plt.title("Funkcja kosztu MLE SAR")
    plt.xlabel("Kąt [°]")
    plt.ylabel("Wartość funkcji kosztu")
    plt.legend()
    plt.grid(True)
    
    # 3. Pozycje elementów wirtualnej anteny
    plt.subplot(2, 3, 3)
    plt.plot(element_positions, np.zeros_like(element_positions), 'bo', markersize=4)
    plt.title("Pozycje elementów wirtualnej anteny")
    plt.xlabel("Pozycja [mm]")
    plt.ylabel("Położenie")
    plt.grid(True)
    
    # 4. Histogram kątów podstawowych
    plt.subplot(2, 3, 4)
    plt.hist(measurements_data['angles_basic'], bins=10, alpha=0.7, label='Podstawowy', density=True)
    plt.axvline(x=sar_angle, color='red', linestyle='--', label=f'SAR MLE: {sar_angle:.1f}°')
    plt.title("Rozkład estymowanych kątów")
    plt.xlabel("Kąt [°]")
    plt.ylabel("Gęstość")
    plt.legend()
    plt.grid(True)
    
    # 5. Stabilność estymacji podstawowej
    plt.subplot(2, 3, 5)
    deviations = np.abs(np.array(measurements_data['angles_basic']) - np.mean(measurements_data['angles_basic']))
    plt.plot(deviations, 'b-', alpha=0.7, label='Odchylenie od średniej')
    plt.title("Stabilność estymacji podstawowej")
    plt.xlabel("Pomiar")
    plt.ylabel("Odchylenie [°]")
    plt.legend()
    plt.grid(True)
    
    # 6. Parametry systemu
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f"Częstotliwość: {phaser.SignalFreq/1e9:.3f} GHz", transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Długość fali: {lambda_mm:.2f} mm", transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Rozstaw fizyczny anten: {d_mm_physical:.2f} mm", transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Rozmiar kroku: {STEP_SIZE_M*1000:.1f} mm", transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Liczba pozycji: {MEASUREMENTS}", transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f"Całkowita apertura: {(element_positions[-1] - element_positions[0]):.1f} mm", transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f"SAR MLE kąt: {sar_angle:.2f}°", transform=plt.gca().transAxes, fontweight='bold')
    plt.title("Parametry systemu")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Zapisz wyniki
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"radar_sar_mle_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("# Pomiary radaru CN0566 - SAR MLE (MATLAB style)\n")
        f.write(f"# Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Częstotliwość: {phaser.SignalFreq/1e9:.3f} GHz\n")
        f.write(f"# Długość fali: {lambda_mm:.2f} mm\n")
        f.write(f"# Rozmiar kroku: {STEP_SIZE_M*1000:.1f} mm\n")
        f.write(f"# Liczba pozycji: {MEASUREMENTS}\n")
        f.write(f"# Całkowita apertura: {(element_positions[-1] - element_positions[0]):.1f} mm\n")
        f.write(f"# SAR MLE kąt: {sar_angle:.2f}°\n")
        f.write("# Kolumny: Pomiar, Pozycja[mm], Kąt_Podstawowy, SAR_MLE_Kąt\n")
        
        for i in range(len(measurements_data['angles_basic'])):
            f.write(f"{i+1:3d} {measurements_data['positions'][i]*1000:8.2f} ")
            f.write(f"{measurements_data['angles_basic'][i]:7.2f} ")
            f.write(f"{sar_angle:7.2f}\n")
    
    print(f"\n[INFO] Wyniki zapisane do: {filename}")
    
    # Dodatkowo zapisz funkcję kosztu
    cost_filename = f"radar_sar_cost_{timestamp}.txt"
    with open(cost_filename, 'w') as f:
        f.write("# Funkcja kosztu SAR MLE\n")
        f.write("# Kolumny: Kąt[°], Wartość_funkcji_kosztu\n")
        for angle, cost in zip(angle_vec, cost_values):
            f.write(f"{angle:6.1f} {cost:12.6e}\n")
    
    print(f"[INFO] Funkcja kosztu zapisana do: {cost_filename}")

if __name__ == "__main__":
    main()