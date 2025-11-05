import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound, minimize
from adi import ad9361
from adi.cn0566 import CN0566

ESP32_IP = "192.168.0.105"
ESP32_PORT = 3333
MEASUREMENTS = 1     # ile kroków i pomiarów
STEP_SIZE_M = 0.00018  # rozmiar kroku w metrach (0.18mm)

# ----------- ESP 32 handling -----------

def connect_esp32():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ESP32_IP, ESP32_PORT))
    s.settimeout(5)
    print("[INFO] Połączono z ESP32")
    return s

def send_step_and_wait(sock):
    sock.sendall(b"STEP_CW\n")
    data = sock.recv(1024)
    if b"DONE_CW" in data:
        print("[INFO] ESP32 wykonał krok")
    else:
        print("[WARN] Otrzymano:", data)

# ----------- ESP 32 handling -----------

def acquire_data(sdr):
    samples = sdr.rx()
    ch0 = samples[0]
    ch1 = samples[1]
    return ch0, ch1

def analyze_angle_estimation(data_ch0, data_ch1, freq_hz=10.3943359375e9, sample_rate=30e6):
    """
    Podstawowa estymacja kąta przylotu sygnału używając różnicy fazowej
    """
    c = 3e8  # prędkość światła
    lambda_m = c / freq_hz  # długość fali w metrach
    d_m = 0.02424  # rozstaw anten w metrach (14mm)                 // poprawka z 0.014, uwzgledniamy to ze jest duzo anten na kanal
    d_over_lambda = d_m / lambda_m  # normalizowany rozstaw
    
    # Oblicz korelację między kanałami
    correlation = np.mean(data_ch0 * np.conj(data_ch1))

    calibrated_correlation = correlation * np.exp(-1j * np.deg2rad(90))
    
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

def a_sar(angle_deg, element_positions_mm, lambda_mm):              # zgadza sie
    """
    Steering vector dla SAR (odpowiednik a_sar z MATLAB)
    """
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi / lambda_mm
    # Steering vector: exp(j * k * d * sin(theta))
    return np.exp(1j * k * element_positions_mm * np.sin(angle_rad)).reshape(-1, 1)

def plot_measurement_analysis(measurements_data, freq_hz=10.3943359375e9):
    """
    Generuje wykresy analizy wszystkich zebranych pomiarów
    """
    print("[INFO] Generowanie wykresów analizy...")
    
    # Połącz wszystkie dane z pomiarów
    all_ch0 = np.concatenate(measurements_data['ch0'])
    all_ch1 = np.concatenate(measurements_data['ch1'])
    sample_rate = measurements_data['fs']
    
    # Oblicz średni kąt z wszystkich pomiarów
    angles = []
    for ch0, ch1 in zip(measurements_data['ch0'], measurements_data['ch1']):
        angle, _, _ = analyze_angle_estimation(ch0, ch1, freq_hz, sample_rate)
        angles.append(angle)
    
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    
    # Oblicz oś czasu
    dt = 1 / sample_rate
    t_single = np.arange(len(measurements_data['ch0'][0])) * dt * 1e6  # czas pojedynczego pomiaru w μs
    t_all = np.arange(len(all_ch0)) * dt * 1e6  # czas wszystkich pomiarów w μs
    
    # Przygotuj dane do wyświetlenia
    ch0_real_all = np.real(all_ch0)
    ch0_imag_all = np.imag(all_ch0)
    ch1_real_all = np.real(all_ch1)
    ch1_imag_all = np.imag(all_ch1)
    ch0_mag_all = np.abs(all_ch0)
    ch1_mag_all = np.abs(all_ch1)
    ch0_phase_all = np.angle(all_ch0)
    ch1_phase_all = np.angle(all_ch1)
    
    # WYKRES 1: Przegląd wszystkich sygnałów
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(f'Analiza wszystkich {len(measurements_data["ch0"])} pomiarów (Średni kąt: {mean_angle:.1f}° ± {std_angle:.1f}°)', fontsize=16)
    
    # Części rzeczywiste i urojone
    axes[0, 0].plot(t_all, ch0_real_all, 'b-', label='Kanał 0 - Re', alpha=0.7, linewidth=0.8)
    axes[0, 0].plot(t_all, ch0_imag_all, 'b--', label='Kanał 0 - Im', alpha=0.7, linewidth=0.8)
    axes[0, 0].plot(t_all, ch1_real_all, 'r-', label='Kanał 1 - Re', alpha=0.7, linewidth=0.8)
    axes[0, 0].plot(t_all, ch1_imag_all, 'r--', label='Kanał 1 - Im', alpha=0.7, linewidth=0.8)
    axes[0, 0].set_xlabel('Czas [μs]')
    axes[0, 0].set_ylabel('Amplituda')
    axes[0, 0].set_title('Części rzeczywiste i urojone sygnałów')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Amplitudy
    axes[0, 1].plot(t_all, ch0_mag_all, 'b-', label='Kanał 0', alpha=0.8, linewidth=1)
    axes[0, 1].plot(t_all, ch1_mag_all, 'r-', label='Kanał 1', alpha=0.8, linewidth=1)
    axes[0, 1].set_xlabel('Czas [μs]')
    axes[0, 1].set_ylabel('Amplituda')
    axes[0, 1].set_title('Amplitudy sygnałów')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fazy
    axes[1, 0].plot(t_all, np.rad2deg(ch0_phase_all), 'b-', label='Kanał 0', alpha=0.8, linewidth=1)
    axes[1, 0].plot(t_all, np.rad2deg(ch1_phase_all), 'r-', label='Kanał 1', alpha=0.8, linewidth=1)
    axes[1, 0].set_xlabel('Czas [μs]')
    axes[1, 0].set_ylabel('Faza [°]')
    axes[1, 0].set_title('Fazy sygnałów')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Różnica faz
    phase_diff_all = np.angle(all_ch0 * np.conj(all_ch1))
    axes[1, 1].plot(t_all, np.rad2deg(phase_diff_all), 'g-', alpha=0.8, linewidth=1)
    axes[1, 1].axhline(y=np.rad2deg(np.mean(phase_diff_all)), color='red', linestyle='--', 
                      label=f'Średnia różnica faz: {np.rad2deg(np.mean(phase_diff_all)):.1f}°')
    axes[1, 1].set_xlabel('Czas [μs]')
    axes[1, 1].set_ylabel('Różnica faz [°]')
    axes[1, 1].set_title('Różnica faz między kanałami')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=False):
    """
    MLE estymacja kąta z danych ze wszystkich pozycji (syntetyczna apertura z dwóch kanałów)
    Y - macierz danych (M x N), gdzie M to liczba "wirtualnych anten" (2 kanały * liczba pozycji)
    element_positions_mm - pozycje w mm (M-elementowy wektor)
    """
    M, N = Y.shape
    R = (Y @ Y.conj().T) / N
    lambda_mm = 3e8 / freq_hz * 1e3

    ch0_idx = np.arange(0, M, 2)  # Indeksy lewych anten
    ch1_idx = np.arange(1, M, 2)  # Indeksy prawych anten

    # Podziel dane i pozycje na dwa kanały
    Y_left = Y[ch0_idx, :]
    Y_right = Y[ch1_idx, :]

    positions_left = element_positions_mm[ch0_idx]
    positions_right = element_positions_mm[ch1_idx]

    # Połącz lewy i prawy kanał jako syntetyczna linia antenowa
    Y_sorted = np.vstack((Y_left, Y_right))  # (M, N)
    element_positions_sorted = np.concatenate((positions_left, positions_right))  # (M,)

    # Używamy tych zreorganizowanych danych dalej
    Y = Y_sorted
    element_positions_mm = element_positions_sorted
    M = Y.shape[0]  # zaktualizowana liczba elementów

    def cost_function(angle_deg):
        a_temp = a_sar(angle_deg, element_positions_mm, lambda_mm)
        a_temp_h = a_temp.conj().T
        denominator = a_temp_h @ a_temp

        if np.abs(denominator) < 1e-12:         # ochrona przed dzieleniem przez 0
            return np.inf
        
        Pv = np.eye(M) - a_temp @ (1 / denominator) @ a_temp_h
        cost = np.abs(np.trace(Pv @ R))
        return cost
    
    # a_test = a_sar(30, element_positions_mm, lambda_mm)
    # print(f"[DEBUG] a_sar(30°): {np.round(a_test.ravel(), 2)}")
    # print(f"[DEBUG] R.shape = {R.shape}, R.diag.mean = {np.mean(np.diag(R)):.3f}, abs trace = {np.abs(np.trace(R)):.3f}")

    if verbose:
        angle_vec = np.arange(-60, 60.1, 0.1)
        pval = np.array([cost_function(a) for a in angle_vec])
        plt.figure()
        plt.plot(angle_vec, pval)
        plt.xlabel("Angle [deg]")
        plt.ylabel("Cost funciton")
        plt.title("MLE without synthetic aparature")
        plt.grid(True)
        plt.show()

    result = minimize_scalar(cost_function, bounds=(-45, 44), method='bounded')
    return result.x

def analyze_mle_with_aperture_dual(measurements_data, freq_hz, verbose=True):
    """
    Analiza MLE z wykorzystaniem danych z obu kanałów i wszystkich pozycji
    (syntetyczna apertura + 2 anteny)
    """
    try:
        M = len(measurements_data['ch0'])  # liczba pozycji
        N = len(measurements_data['ch0'][0])  # liczba próbek

        # Kalibracja fazowa kanału 1
        calibration_phase_deg = 90
        calibration_factor = np.exp(1j * np.deg2rad(calibration_phase_deg))

        # Przygotuj dane Y (2M x N)
        ch0_stack = []
        ch1_stack = []
        positions_mm = []

        for i in range(M):
            ch0 = np.array(measurements_data['ch0'][i])
            ch1 = np.array(measurements_data['ch1'][i]) * calibration_factor
            ch0_stack.append(ch0)
            ch1_stack.append(ch1)

            pos_mm = measurements_data['positions'][i] * 1000  # m → mm
            positions_mm.extend([pos_mm, pos_mm + 24.24])  # pozycja anteny 0 i 1

        # for i in range(M):
        #     print(f"ch0[{i}].shape = {ch0_stack[i].shape}, ch1[{i}].shape = {ch1_stack[i].shape}")

        Y = np.vstack(ch0_stack + ch1_stack)  # shape: (2M, N)
        element_positions_mm = np.array(positions_mm)  # shape: (2M,)

        # print(f"Y.shape = {Y.shape}") 
        # print(f"element_positions_mm.shape = {element_positions_mm.shape}")

        # dane synetetyczne
        # positions = np.array(element_positions_mm)  # Twoje aktualne pozycje
        # lambda_mm = 3e8 / 10.394e9 * 1e3  # częstotliwość 10.394 GHz

        # Y_synthetic = generate_synthetic_Y(positions, 25, lambda_mm)
        # estimated_angle = MLE_sar_full_aperture_dual(Y_synthetic, positions, 10.394e9, verbose=True)

        # print(f"True angle: 25°, Estimated: {estimated_angle:.2f}°")

        # Estymacja kąta
        estimated_angle = MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=verbose)
        print(f"[WYNIK] Kąt DOA z syntetycznej apertury (2 kanały): {estimated_angle:.2f}°")
        return estimated_angle
    except Exception as e:
        print(f"[ERROR] Błąd podczas analizy MLE z 2-kanałowej syntetycznej apertury: {e}")
        return None

def generate_chirp(fs, t_pulse, bandwidth):
    t = np.arange(0, t_pulse, 1/fs)
    k = bandwidth / t_pulse  # chirp rate
    chirp = np.exp(1j * np.pi * k * t**2)  # complex LFM
    return chirp

def generate_synthetic_Y(element_positions_mm, angle_deg, lambda_mm, N=4096, SNR_dB=30):
    k = 2 * np.pi / lambda_mm
    M = len(element_positions_mm)
    angle_rad = np.deg2rad(angle_deg)
    steering_vector = np.exp(1j * k * element_positions_mm * np.sin(angle_rad)).reshape(M, 1)
    signal = np.exp(1j * 2 * np.pi * np.random.rand(1, N))  # losowy sygnał (ciąg faz)
    noise_power = 10 ** (-SNR_dB / 10)
    noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * np.sqrt(noise_power / 2)
    Y = steering_vector @ signal + noise
    return Y


def main():
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
        'ch0': [],
        'ch1': [],
        'positions': [],
        'fs': sdr.sample_rate,
        'mle_angles': [],  # dodajemy listę na kąty MLE
        'basic_angles': []  # dodajemy listę na kąty podstawowe
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

        # Zapisz dane do struktur
        measurements_data['ch0'].append(ch0)
        measurements_data['ch1'].append(ch1)
        measurements_data['positions'].append(current_position)
        
        # Oblicz kąt podstawową metodą
        angle_basic, _, _ = analyze_angle_estimation(ch0, ch1, phaser.SignalFreq, sdr.sample_rate)
        measurements_data['basic_angles'].append(angle_basic)
        
        try:
            print(f"[INFO] Pozycja: {current_position*100:.1f} cm")
            print(f"[INFO] Kąt podstawowy: {angle_basic:.1f}°")
        except Exception as e:
            print(f"[INFO] Pozycja: {current_position*100:.1f} cm")
            print(f"[INFO] Kąt podstawowy: {angle_basic:.1f}°")
        
        time.sleep(0.05)
    
    sock.close()
    print("\n[INFO] Pomiary zakończone, rozpoczynam analizę...")

    # Saving measurment data to .npz file
    save_data = True
    if save_data:
        # np.savez_compressed(f"measurements_{timestamp}.npz", **measurements_data)
        np.savez_compressed(f"measurements/18_4_deg_single.npz", **measurements_data)
        print(f"Saved measurements")
    
    # Oblicz statystyki (ignorując NaN)
    # valid_mle_angles = [a for a in measurements_data['mle_angles'] if not np.isnan(a)]
    
    # mean_basic = np.mean(measurements_data['basic_angles'])
    # std_basic = np.std(measurements_data['basic_angles'])
    
    # if valid_mle_angles:
    #     mean_mle = np.mean(valid_mle_angles)
    #     std_mle = np.std(valid_mle_angles)
    #     print(f"\n[WYNIKI] Metoda podstawowa: {mean_basic:.1f}° ± {std_basic:.1f}°")
    #     print(f"[WYNIKI] Metoda MLE SAR: {mean_mle:.1f}° ± {std_mle:.1f}° (z {len(valid_mle_angles)} pomiarów)")
    # else:
    #     print(f"\n[WYNIKI] Metoda podstawowa: {mean_basic:.1f}° ± {std_basic:.1f}°")
    #     print("[WYNIKI] Metoda MLE SAR: Brak prawidłowych wyników")
    
    # # Pokaż wykresy ze wszystkimi danymi
    # plot_measurement_analysis(measurements_data, phaser.SignalFreq)

    print("\n[INFO] Analiza MLE SAR z syntetycznej apertury (kanały 0 i 1)...")
    angle_sar_aperture_dual = analyze_mle_with_aperture_dual(measurements_data, phaser.SignalFreq, verbose=True)
    if angle_sar_aperture_dual is not None:
        print(f"[WYNIKI] MLE SAR (syntetyczna apertura, 2 kanały): {angle_sar_aperture_dual:.2f}°")

    

if __name__ == "__main__":
    main()