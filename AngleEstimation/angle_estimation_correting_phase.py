import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound
from adi import ad9361
from adi.cn0566 import CN0566

ESP32_IP = "192.168.0.103"
ESP32_PORT = 3333
MEASUREMENTS = 1  # ile kroków i pomiarów
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

def MLE_SAR(measurements_data):
    print("[INFO] Analiza metodą MLE SAR...")

    freq_hz = 10.3943359375e9
    c = 3e8
    lambda_mm = c / freq_hz * 1e3
    d_m = 24.25e-3  # rozstaw anten (nieużywany bezpośrednio tutaj)

    M = 2
    # N = measurements_data['virtualArray'].
    print(f"[INFO] Liczba elementów: {M}, próbki: {1}")

    # virtualArray = measurements_data['virtualArray']  # lista 3 arrayów

    virtualArray = np.array(measurements_data['virtualArray'])
    positions = np.array(measurements_data['positions'])  # lista 3 arrayów
    
    # Macierz korelacji
    R = np.dot(virtualArray, virtualArray.conj().T)

    def steering_vector_sar(angle_deg, positions_mm, lambda_mm):
        angle_rad = np.deg2rad(angle_deg)
        k = 2 * np.pi / lambda_mm
        return np.exp(1j * k * positions_mm * np.sin(angle_rad)).reshape(-1, 1)

    def cost_function_sar(angle_deg, positions_mm):
        a = steering_vector_sar(angle_deg, positions_mm, lambda_mm)
        aH = a.conj().T

        aH_a = np.dot(aH, a)
        if np.abs(aH_a) < 1e-12:
            return np.inf

        inv_aH_a = 1.0 / aH_a[0, 0]
        P_perp = np.eye(M) - np.dot(a, aH) * inv_aH_a

        J = np.abs(np.trace(np.dot(P_perp, R)))
        return J

    positions_mm = np.array(measurements_data['positions'])

    # Siatka przeszukiwania
    min_angle = -45
    max_angle = 45
    angle_vec = np.arange(min_angle, max_angle + 0.1, 0.1)
    pval = np.array([cost_function_sar(angle, positions_mm) for angle in angle_vec])

    # Rysowanie wykresu jak w MATLAB
    plt.figure(figsize=(10, 6))
    plt.plot(angle_vec, pval, label="MLE Cost Function")
    plt.title("MLE z syntetycznej anteny")
    plt.xlabel("Kąt [deg]")
    plt.ylabel("Funkcja kosztu")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Znajdź najlepszy punkt z siatki
    best_grid_idx = np.argmin(pval)
    best_grid_angle = angle_vec[best_grid_idx]

    # Optymalizacja lokalna wokół najlepszego kąta z siatki
    try:
        result = fminbound(lambda angle: cost_function_sar(angle, positions_mm),
                           max(min_angle, best_grid_angle - 5),
                           min(max_angle, best_grid_angle + 5),
                           xtol=1e-6)
        best_angle = result
        best_cost = cost_function_sar(result, positions_mm)
    except:
        best_angle = best_grid_angle
        best_cost = pval[best_grid_idx]

    # Dodatkowa optymalizacja z różnych punktów
    start_points = [min_angle, -20, 0, 20, max_angle]
    for start_angle in start_points:
        try:
            result = fminbound(lambda angle: cost_function_sar(angle, positions_mm),
                               min_angle, max_angle, xtol=1e-6)
            current_cost = cost_function_sar(result, positions_mm)
            if current_cost < best_cost:
                best_cost = current_cost
                best_angle = result
        except:
            continue

    print(f"[INFO] Oszacowany kąt: {best_angle:.2f}°")
    return best_angle

def main():
    print("===============================")
    print("    RADAR CN0566 + ESP32       ")
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
        'virtualArray': [],
        'positions': [],
        'angles_basic': [],
        'angles_mle': []
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
        current_position_ch0 = i * STEP_SIZE_M
        current_position_ch1 = i * STEP_SIZE_M + d_m
        
        # Pobierz dane z radaru
        # ch0, ch1 = acquire_data(sdr)
        ch0, ch1 = [21.1706135741686 - 0.888916916651944j, 19.8632052111244 - 4.35960460501996j]

        # Zapisz dane do struktur
        measurements_data['virtualArray'].append(ch0)
        measurements_data['virtualArray'].append(ch1)
        measurements_data['positions'].append(current_position_ch0)
        measurements_data['positions'].append(current_position_ch1)
        
        # print(f"[INFO] Pozycja: {current_position_ch0*100:.1f} cm")
        
        time.sleep(0.1)
    
    sock.close()
    print("\n[INFO] Pomiary zakończone, rozpoczynam analizę...")
    
    mle_sar_angle = MLE_SAR(measurements_data)

if __name__ == "__main__":
    main()
