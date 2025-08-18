import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound, minimize
from adi import ad9361
from adi.cn0566 import CN0566
import adi
import threading

ESP32_IP = "192.168.0.103"
ESP32_PORT = 3333
MEASUREMENTS = 50     # ile kroków i pomiarów
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
    UWAGA: Ta metoda zakłada stacjonarne anteny - nie uwzględnia ruchu platformy!
    """
    c = 3e8  # prędkość światła
    lambda_m = c / freq_hz  # długość fali w metrach
    d_m = 0.02424  # rozstaw anten w metrach (24.24mm)
    d_over_lambda = d_m / lambda_m  # normalizowany rozstaw
    
    # Oblicz korelację między kanałami
    correlation = np.mean(data_ch0 * np.conj(data_ch1))

    # Kalibracja fazowa - może wymagać dostrojenia dla ruchomej platformy
    calibrated_correlation = correlation * np.exp(-1j * np.deg2rad(90))
    
    # Różnica fazowa między kanałami
    phase_diff = np.angle(calibrated_correlation)
    
    # Estymacja kąta na podstawie różnicy fazowej (klasyczna formula DOA)
    sin_theta = phase_diff / (2 * np.pi * d_over_lambda)
    
    # Sprawdź czy wartość jest w zakresie [-1, 1]
    if abs(sin_theta) > 1:
        sin_theta = np.clip(sin_theta, -1, 1)
    
    estimated_angle_rad = np.arcsin(sin_theta)
    estimated_angle_deg = np.rad2deg(estimated_angle_rad)
    
    return estimated_angle_deg, phase_diff, calibrated_correlation

def a_sar(angle_deg, element_positions_mm, lambda_mm):
    """
    Steering vector dla SAR (odpowiednik a_sar z MATLAB)
    """
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi / lambda_mm
    # Steering vector: exp(j * k * d * sin(theta))
    return np.exp(1j * k * element_positions_mm * np.sin(angle_rad)).reshape(-1, 1)

def plot_measurement_analysis(measurements_data, freq_hz=10.3943359375e9):
    """
    Generuje wykresy analizy wszystkich zebranych pomiarów - ZOPTYMALIZOWANE
    """
    print("[INFO] Generowanie wykresów analizy...")
    
    # Ogranicz dane do analizy - weź co N-tą próbkę dla szybkości
    downsample_factor = max(1, len(measurements_data['ch0']) // 20)  # Max 20 pomiarów na wykresie
    sample_indices = range(0, len(measurements_data['ch0']), downsample_factor)
    
    # Połącz wybrane dane z pomiarów
    selected_ch0 = [measurements_data['ch0'][i] for i in sample_indices]
    selected_ch1 = [measurements_data['ch1'][i] for i in sample_indices]
    
    # Jeszcze bardziej ogranicz próbki w czasie - weź co 10-tą próbkę z każdego pomiaru
    time_downsample = 10
    all_ch0 = np.concatenate([ch0[::time_downsample] for ch0 in selected_ch0])
    all_ch1 = np.concatenate([ch1[::time_downsample] for ch1 in selected_ch1])
    
    sample_rate = measurements_data['fs']
    
    # Oblicz średni kąt z wszystkich pomiarów
    angles = []
    for ch0, ch1 in zip(measurements_data['ch0'], measurements_data['ch1']):
        angle, _, _ = analyze_angle_estimation(ch0, ch1, freq_hz, sample_rate)
        angles.append(angle)
    
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    
    # Oblicz oś czasu dla ograniczonych danych
    dt = 1 / sample_rate * time_downsample  # Uwzględnij downsampling
    t_all = np.arange(len(all_ch0)) * dt * 1e6  # czas w μs
    
    # Przygotuj dane do wyświetlenia
    ch0_real_all = np.real(all_ch0)
    ch0_imag_all = np.imag(all_ch0)
    ch1_real_all = np.real(all_ch1)
    ch1_imag_all = np.imag(all_ch1)
    ch0_mag_all = np.abs(all_ch0)
    ch1_mag_all = np.abs(all_ch1)
    ch0_phase_all = np.angle(all_ch0)
    ch1_phase_all = np.angle(all_ch1)
    
    # WYKRES 1: Przegląd wszystkich sygnałów - ZOPTYMALIZOWANY
    # fig1, axes = plt.subplots(2, 2, figsizaxes[0, 0].plot(t_all, ch0_real_all, 'b-', label='Kanał 0 - Re', alpha=0.7, linewidth=0.5)
    # axes[0, 0].plot(t_all, ch0_imag_all, 'b--', label='Kanał 0 - Im', alpha=0.7, linewidth=0.5)
    # axes[0, 0].plot(t_all, ch1_real_all, 'r-', label='Kanał 1 - Re', alpha=0.7, linewidth=0.5)
    # axes[0, 0].plot(t_all, ch1_imag_all, 'r--', label='Kanał 1 - Im', alpha=0.7, linewidth=0.5)
    # axes[0, 0].set_xlabel('Czas [μs]')
    # axes[0, 0].set_ylabel('Amplituda')
    # axes[0, 0].set_title('Części rzeczywiste i urojone sygnałów')
    # axes[0, 0].legend(loc='upper right')  # Konkretna lokalizacja zamiast 'best'
    # axes[0, 0].grid(True, alpha=0.3)
    
    # # Amplitudy
    # axes[0, 1].plot(t_all, ch0_mag_all, 'b-', label='Kanał 0', alpha=0.8, linewidth=0.8)
    # axes[0, 1].plot(t_all, ch1_mag_all, 'r-', label='Kanał 1', alpha=0.8, linewidth=0.8)
    # axes[0, 1].set_xlabel('Czas [μs]')
    # axes[0, 1].set_ylabel('Amplituda')
    # axes[0, 1].set_title('Amplitudy sygnałów')
    # axes[0, 1].legend(loc='upper right')
    # axes[0, 1].grid(True, alpha=0.3)
    
    # # Fazy
    # axes[1, 0].plot(t_all, np.rad2deg(ch0_phase_all), 'b-', label='Kanał 0', alpha=0.8, linewidth=0.8)
    # axes[1, 0].plot(t_all, np.rad2deg(ch1_phase_all), 'r-', label='Kanał 1', alpha=0.8, linewidth=0.8)
    # axes[1, 0].set_xlabel('Czas [μs]')
    # axes[1, 0].set_ylabel('Faza [°]')
    # axes[1, 0].set_title('Fazy sygnałów')
    # axes[1, 0].legend(loc='upper right')
    # axes[1, 0].grid(True, alpha=0.3)
    
    # # Różnica faz
    # phase_diff_all = np.angle(all_ch0 * np.conj(all_ch1))
    # axes[1, 1].plot(t_all, np.rad2deg(phase_diff_all), 'g-', alpha=0.8, linewidth=0.8)
    # axes[1, 1].axhline(y=np.rad2deg(np.mean(phase_diff_all)), color='red', linestyle='--', 
    #                   label=f'Średnia: {np.rad2deg(np.mean(phase_diff_all)):.1f}°')
    # axes[1, 1].set_xlabel('Czas [μs]')
    # axes[1, 1].set_ylabel('Różnica faz [°]')
    # axes[1, 1].set_title('Różnica faz między kanałami')
    # axes[1, 1].legend(loc='upper right')
    # axes[1, 1].grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # # plt.show()e=(15, 10))
    # fig1.suptitle(f'Analiza pomiarów SAR (Średni kąt: {mean_angle:.1f}° ± {std_angle:.1f}°)\n' + 
    #               f'Pokazano {len(selected_ch0)}/{len(measurements_data["ch0"])} pomiarów (downsampled)', fontsize=14)
    
    # Części rzeczywiste i urojone - z mniejszą ilością danych
    # axes[0, 0].plot(t_all, ch0_real_all, 'b-', label='Kanał 0 - Re', alpha=0.7, linewidth=0.5)
    # axes[0, 0].plot(t_all, ch0_imag_all, 'b--', label='Kanał 0 - Im', alpha=0.7, linewidth=0.5)
    # axes[0, 0].plot(t_all, ch1_real_all, 'r-', label='Kanał 1 - Re', alpha=0.7, linewidth=0.5)
    # axes[0, 0].plot(t_all, ch1_imag_all, 'r--', label='Kanał 1 - Im', alpha=0.7, linewidth=0.5)
    # axes[0, 0].set_xlabel('Czas [μs]')
    # axes[0, 0].set_ylabel('Amplituda')
    # axes[0, 0].set_title('Części rzeczywiste i urojone sygnałów')
    # axes[0, 0].legend(loc='upper right')  # Konkretna lokalizacja zamiast 'best'
    # axes[0, 0].grid(True, alpha=0.3)
    
    # # Amplitudy
    # axes[0, 1].plot(t_all, ch0_mag_all, 'b-', label='Kanał 0', alpha=0.8, linewidth=0.8)
    # axes[0, 1].plot(t_all, ch1_mag_all, 'r-', label='Kanał 1', alpha=0.8, linewidth=0.8)
    # axes[0, 1].set_xlabel('Czas [μs]')
    # axes[0, 1].set_ylabel('Amplituda')
    # axes[0, 1].set_title('Amplitudy sygnałów')
    # axes[0, 1].legend(loc='upper right')
    # axes[0, 1].grid(True, alpha=0.3)
    
    # # Fazy
    # axes[1, 0].plot(t_all, np.rad2deg(ch0_phase_all), 'b-', label='Kanał 0', alpha=0.8, linewidth=0.8)
    # axes[1, 0].plot(t_all, np.rad2deg(ch1_phase_all), 'r-', label='Kanał 1', alpha=0.8, linewidth=0.8)
    # axes[1, 0].set_xlabel('Czas [μs]')
    # axes[1, 0].set_ylabel('Faza [°]')
    # axes[1, 0].set_title('Fazy sygnałów')
    # axes[1, 0].legend(loc='upper right')
    # axes[1, 0].grid(True, alpha=0.3)
    
    # # Różnica faz
    # phase_diff_all = np.angle(all_ch0 * np.conj(all_ch1))
    # axes[1, 1].plot(t_all, np.rad2deg(phase_diff_all), 'g-', alpha=0.8, linewidth=0.8)
    # axes[1, 1].axhline(y=np.rad2deg(np.mean(phase_diff_all)), color='red', linestyle='--', 
    #                   label=f'Średnia: {np.rad2deg(np.mean(phase_diff_all)):.1f}°')
    # axes[1, 1].set_xlabel('Czas [μs]')
    # axes[1, 1].set_ylabel('Różnica faz [°]')
    # axes[1, 1].set_title('Różnica faz między kanałami')
    # axes[1, 1].legend(loc='upper right')
    # axes[1, 1].grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # # plt.show()
    
    print(f"[INFO] Wykresy wygenerowane (użyto {len(all_ch0)} próbek z {len(np.concatenate(measurements_data['ch0']))} dostępnych)")

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

    if verbose:
        angle_vec = np.arange(-45, 44.1, 0.1)
        pval = np.array([cost_function(a) for a in angle_vec])
        plt.figure()
        plt.plot(angle_vec, pval)
        plt.xlabel("Kąt [deg]")
        plt.ylabel("Funkcja kosztu")
        plt.title("MLE z syntetycznej apertury")
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

        Y = np.vstack(ch0_stack + ch1_stack)  # shape: (2M, N)
        element_positions_mm = np.array(positions_mm)  # shape: (2M,)

        # Estymacja kąta
        estimated_angle = MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=verbose)
        print(f"[WYNIK] Kąt DOA z syntetycznej apertury (2 kanały): {estimated_angle:.2f}°")
        return estimated_angle
    except Exception as e:
        print(f"[ERROR] Błąd podczas analizy MLE z 2-kanałowej syntetycznej apertury: {e}")
        return None

def generate_chirp_signal(sample_rate, duration_us=100, bandwidth_mhz=10, center_freq_mhz=1):
    """
    Generuje sygnał chirp dla radaru
    """
    duration_s = duration_us * 1e-6
    N = int(sample_rate * duration_s)
    
    # Zapewnij, że N jest potęgą 2 dla lepszej wydajności
    N = 2**int(np.log2(N))
    print(f"[DEBUG] Długość sygnału chirp: {N} próbek ({N/sample_rate*1e6:.1f} μs)")
    
    t = np.linspace(0, duration_s, N)
    bandwidth_hz = bandwidth_mhz * 1e6
    center_freq_hz = center_freq_mhz * 1e6
    
    # Chirp frequency sweep
    k = bandwidth_hz / duration_s  # chirp rate
    freq_inst = center_freq_hz + k * t
    
    # Generate I/Q chirp signal
    chirp_signal = np.exp(1j * 2 * np.pi * (center_freq_hz * t + 0.5 * k * t**2))
    
    # Scale to 14-bit range and ensure correct data type
    chirp_signal = chirp_signal * (2**13)
    chirp_signal = chirp_signal.astype(np.complex64)
    
    print(f"[DEBUG] Typ danych chirp: {chirp_signal.dtype}")
    print(f"[DEBUG] Kształt sygnału chirp: {chirp_signal.shape}")
    
    return chirp_signal, t

def setup_chirp_radar(sdr, phaser):
    """
    Konfiguruje radar do pracy z sygnałami chirp
    """
    print("[INFO] Konfigurowanie radaru chirp...")
    
    # Parametry chirp
    sample_rate = sdr.sample_rate
    chirp_duration_us = 100  # 100 μs chirp
    chirp_bandwidth_mhz = 5  # 5 MHz bandwidth
    chirp_center_freq_mhz = 1  # 1 MHz center frequency
    
    # Generuj sygnał chirp
    chirp_signal, t_chirp = generate_chirp_signal(
        sample_rate, chirp_duration_us, chirp_bandwidth_mhz, chirp_center_freq_mhz
    )
    
    # Konfiguracja nadajnika - UPROSZCZONA
    sdr.tx_enabled_channels = [0, 1]  # Tylko kanał 0
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = -88  # Dostosuj moc dla bezpiecznej pracy
    sdr.tx_hardwaregain_chan1 = 0  
    sdr.tx_lo = sdr.rx_lo  # Synchronizacja LO

    phaser.tx_trig_en = 0 
    phaser.enable = 0  
    
    # Konfiguracja phasera dla nadawania
    phaser._gpios.gpio_tx_sw = 0  # TX_OUT_2
    phaser._gpios.gpio_vctrl_1 = 1  # Użyj wbudowanego PLL
    phaser._gpios.gpio_vctrl_2 = 1  # Wyślij LO do obwodu Tx
    
    print(f"[INFO] Chirp radar skonfigurowany:")
    print(f"  - Czas trwania chirp: {chirp_duration_us} μs")
    print(f"  - Szerokość pasma: {chirp_bandwidth_mhz} MHz")
    print(f"  - Częstotliwość środkowa: {chirp_center_freq_mhz} MHz")
    print(f"  - Moc Tx: {sdr.tx_hardwaregain_chan0} dBm")
    print(f"  - Włączone kanały Tx: {sdr.tx_enabled_channels}")
    
    return chirp_signal

# def start_chirp_transmission(sdr, chirp_signal):
#     """
#     Rozpoczyna nadawanie chirp z dynamicznym zwiększaniem mocy co 10s
#     """
#     print("[INFO] Rozpoczynam nadawanie chirp z kontrolą mocy...")

#     tx_buffer = np.tile(chirp_signal, 10)
#     enabled_tx_channels = len(sdr.tx_enabled_channels)

#     sdr._ctx.set_timeout(30000)

#     # Konfiguracja - domyślnie słaby sygnał
#     normal_tx_gain_ch0 = -88
#     normal_tx_gain_ch1 = -20
#     strong_tx_gain_ch0 = -40
#     strong_tx_gain_ch1 = 0

#     sdr.tx_cyclic_buffer = True

#     try:
#         last_strong_signal_time = time.time()
#         strong_signal_interval = 10.0
#         strong_signal_duration = 2.0

#         while True:
#             now = time.time()
#             elapsed = now - last_strong_signal_time

#             if elapsed >= strong_signal_interval:
#                 print("[TX] Wysyłam MOCNY sygnał przez 2s...")
#                 sdr.tx_hardwaregain_chan0 = strong_tx_gain_ch0
#                 sdr.tx_hardwaregain_chan1 = strong_tx_gain_ch1
#                 sdr.tx([tx_buffer, np.zeros_like(tx_buffer)])
#                 time.sleep(strong_signal_duration)
#                 last_strong_signal_time = time.time()
#                 print("[TX] Powrót do normalnego nadawania...")
#                 sdr.tx_hardwaregain_chan0 = normal_tx_gain_ch0
#                 sdr.tx_hardwaregain_chan1 = normal_tx_gain_ch1
#             else:
#                 # Zwykłe nadawanie w tle (jeśli potrzeba)
#                 sdr.tx([tx_buffer, np.zeros_like(tx_buffer)])
#                 time.sleep(0.5)

#     except KeyboardInterrupt:
#         print("[INFO] Przerwano transmisję chirp.")
#     except Exception as e:
#         print(f"[ERROR] Błąd podczas transmisji: {e}")

def plot_sar_progress(measurements_data, positions, angles_basic, angles_mle=None):
    """
    Wykres postępu pomiarów SAR w czasie rzeczywistym
    """
    plt.figure(figsize=(15, 5))
    
    # Pozycje vs kąty
    plt.subplot(1, 3, 1)
    positions_cm = np.array(positions) * 100
    plt.plot(positions_cm, angles_basic, 'bo-', label='Kąt podstawowy', markersize=4)
    if angles_mle and len(angles_mle) > 0:
        plt.plot(positions_cm[:len(angles_mle)], angles_mle, 'ro-', label='Kąt MLE', markersize=4)
    plt.xlabel('Pozycja [cm]')
    plt.ylabel('Kąt DOA [°]')
    plt.title('Estymacja kąta vs pozycja')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moc sygnału
    plt.subplot(1, 3, 2)
    if len(measurements_data['ch0']) > 0:
        powers_ch0 = [np.mean(np.abs(ch0)**2) for ch0 in measurements_data['ch0']]
        powers_ch1 = [np.mean(np.abs(ch1)**2) for ch1 in measurements_data['ch1']]
        plt.plot(positions_cm, powers_ch0, 'b-', label='CH0', linewidth=2)
        plt.plot(positions_cm, powers_ch1, 'r-', label='CH1', linewidth=2)
        plt.xlabel('Pozycja [cm]')
        plt.ylabel('Średnia moc')
        plt.title('Moc sygnału odbieranego')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Różnica faz
    plt.subplot(1, 3, 3)
    if len(measurements_data['ch0']) > 0:
        phase_diffs = []
        for ch0, ch1 in zip(measurements_data['ch0'], measurements_data['ch1']):
            phase_diff = np.angle(np.mean(ch0 * np.conj(ch1)))
            phase_diffs.append(np.rad2deg(phase_diff))
        plt.plot(positions_cm, phase_diffs, 'g-', linewidth=2)
        plt.xlabel('Pozycja [cm]')
        plt.ylabel('Różnica faz [°]')
        plt.title('Różnica faz CH0-CH1')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

import threading

def start_chirp_transmission(sdr, chirp_signal):
    """
    Rozpoczyna ciągłe, cykliczne nadawanie chirp z niską mocą.
    """
    print("[INFO] Rozpoczynam ciągłe nadawanie chirp z niską mocą...")

    tx_buffer = np.tile(chirp_signal, 10)

    # Ustawienia mocy
    normal_tx_gain_ch0 = -88
    normal_tx_gain_ch1 = -20

    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = normal_tx_gain_ch0
    sdr.tx_hardwaregain_chan1 = normal_tx_gain_ch1

    # Start cyklicznego nadawania
    sdr.tx([tx_buffer, np.zeros_like(tx_buffer)])

# Funkcja `pulse_gain` jest już niepotrzebna, ponieważ
# sterowanie mocą będzie w głównej pętli.

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
    
    # Konfiguracja urządzeń (twoja oryginalna konfiguracja)
    time.sleep(0.5)
    phaser.configure(device_mode="rx")
    sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
    sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
    sdr._ctrl.debug_attrs["initialize"].value = "1"
    
    sdr.rx_enabled_channels = [0, 1]
    sdr._rxadc.set_kernel_buffers_count(1)
    rx = sdr._ctrl.find_channel("voltage0")
    rx.attrs["quadrature_tracking_en"].value = "1"
    
    # Parametry próbkowania (twoje oryginalne)
    sdr.sample_rate = int(30e6)
    sdr.rx_buffer_size = int(4 * 1024)
    sdr.rx_rf_bandwidth = int(10e6)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = 0
    sdr.rx_hardwaregain_chan1 = 0
    sdr.rx_lo = int(2.0e9)
    sdr.filter = "LTE20_MHz.ftr"
    
    # Konfiguracja phasera (twoja oryginalna)
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
    
    # NOWA KONFIGURACJA: Chirp radar
    print("\n" + "="*50)
    print("KONFIGURACJA RADARU CHIRP")
    print("="*50)
    
    chirp_signal = setup_chirp_radar(sdr, phaser)
    threading.Thread(target=start_chirp_transmission, args=(sdr, chirp_signal), daemon=True).start()

# reszta SAR / pomiary / przetwarzanie


    # start_chirp_transmission(sdr, chirp_signal)
    
    print("\n" + "="*50)
    print("ROZPOCZYNAM POMIARY SAR")
    print("="*50)
    
    # Uruchom pomiary SAR
    run_sar_measurements(phaser, sdr, chirp_signal)


def run_sar_measurements(phaser, sdr, chirp_signal):
    """
    Pomiary SAR z aktywnym nadawaniem chirp.
    """
    print("[INFO] Rozpoczynam pomiary SAR z aktywnym nadajnikiem...")
    
    # Połączenie z ESP32
    sock = connect_esp32()
    time.sleep(1)
    
    # Struktury danych dla nowych metod
    measurements_data = {
        'ch0': [],
        'ch1': [],
        'positions': [],
        'fs': sdr.sample_rate,
        'mle_angles': [],
        'basic_angles': []
    }
    
    print(f"\n[INFO] Rozpoczynam {MEASUREMENTS} pomiarów...")
    
    # Ustawienia mocy
    strong_tx_gain_ch0 = -40
    strong_tx_gain_ch1 = 0
    normal_tx_gain_ch0 = sdr.tx_hardwaregain_chan0
    normal_tx_gain_ch1 = sdr.tx_hardwaregain_chan1
    
    # Wyłącz tryb cykliczny, aby mieć pełną kontrolę nad nadawaniem
    # UWAGA: W wersji uproszczonej, nadawanie jest cykliczne. Zmieniamy tylko moc.
    # Ta linia nie jest już potrzebna, ponieważ start_chirp_transmission
    # już uruchomił cykliczne nadawanie.
    # sdr.tx_cyclic_buffer = False 

    # Główna pętla pomiarowa
    for i in range(MEASUREMENTS):
        print(f"\n[{i+1}/{MEASUREMENTS}] Pomiar...")
        
        # Wykonaj krok na ESP32 (jeśli nie jest to pierwszy pomiar)
        if i > 0:
            send_step_and_wait(sock)
            time.sleep(0.3)
        
        current_position = i * STEP_SIZE_M

        # Chwilowe zwiększenie mocy na czas pomiaru
        sdr.tx_hardwaregain_chan0 = strong_tx_gain_ch0
        sdr.tx_hardwaregain_chan1 = strong_tx_gain_ch1
        time.sleep(0.05)  # Krótki czas na ustabilizowanie się wzmocnienia

        # Pobierz dane z radaru
        ch0, ch1 = acquire_data(sdr)
        
        # Powrót do niskiej mocy
        sdr.tx_hardwaregain_chan0 = normal_tx_gain_ch0
        sdr.tx_hardwaregain_chan1 = normal_tx_gain_ch1
        
        # Zapisz dane do struktur
        measurements_data['ch0'].append(ch0)
        measurements_data['ch1'].append(ch1)
        measurements_data['positions'].append(current_position)
        
        # Oblicz kąt podstawową metodą
        angle_basic, phase_diff, correlation = analyze_angle_estimation(ch0, ch1, phaser.SignalFreq, sdr.sample_rate)
        measurements_data['basic_angles'].append(angle_basic)
        
        # Sprawdź moc sygnału
        power_ch0 = np.mean(np.abs(ch0)**2)
        power_ch1 = np.mean(np.abs(ch1)**2)
        
        print(f"[INFO] Pozycja: {current_position*100:.1f} cm")
        print(f"[INFO] Kąt podstawowy: {angle_basic:.1f}°")
        print(f"[INFO] Moc CH0: {10*np.log10(power_ch0+1e-12):.1f} dB")
        print(f"[INFO] Moc CH1: {10*np.log10(power_ch1+1e-12):.1f} dB")
        print(f"[INFO] Różnica faz: {np.rad2deg(phase_diff):.1f}°")
        
        time.sleep(0.05)
    
    sock.close()
    print("\n[INFO] Pomiary zakończone, rozpoczynam analizę...")
    
    # Pokaż wykresy ze wszystkimi danymi
    plot_measurement_analysis(measurements_data, phaser.SignalFreq)

    print("\n[INFO] Analiza MLE SAR z syntetycznej apertury (kanały 0 i 1)...")
    angle_sar_aperture_dual = analyze_mle_with_aperture_dual(measurements_data, phaser.SignalFreq, verbose=True)
    if angle_sar_aperture_dual is not None:
        print(f"[WYNIKI] MLE SAR (syntetyczna apertura, 2 kanały): {angle_sar_aperture_dual:.2f}°")
    
    # Podsumowanie wyników z uwzględnieniem geometrii SAR
    mean_basic_angle = np.mean(measurements_data['basic_angles'])
    std_basic_angle = np.std(measurements_data['basic_angles'])
    
    print("\n" + "="*50)
    print("PODSUMOWANIE WYNIKÓW SAR")
    print("="*50)
    print(f"Liczba pomiarów: {len(measurements_data['ch0'])}")
    print(f"Zakres pozycji: 0 - {(len(measurements_data['ch0'])-1) * STEP_SIZE_M * 100:.1f} cm")
    print(f"Rozmiar kroku: {STEP_SIZE_M * 1000:.1f} mm")
    print("")
    print("UWAGA O GEOMETRII SYSTEMU:")
    print("- Nadajnik i odbiornik poruszają się razem (monostatyczny SAR)")
    print("- Target: metalowa skrzynka w odległości ~50 cm")
    print("- Kąty są mierzone względem platformy, nie względem targetu!")
    print("")
    print("WYNIKI ESTYMACJI KĄTÓW:")
    print(f"Średni kąt (podstawowy): {mean_basic_angle:.1f}° ± {std_basic_angle:.1f}°")
    if angle_sar_aperture_dual is not None:
        print(f"Kąt MLE (syntetyczna apertura): {angle_sar_aperture_dual:.1f}°")
    print("")
    print("INTERPRETACJA:")
    if std_basic_angle > 10:
        print("⚠ Duża zmienność kątów może wynikać z:")
        print("  - Wielodrogowości sygnału (odbicia od różnych części targetu)")
        print("  - Ruchu platformy (efekt Dopplera)")
        print("  - Niedoskonałej kalibracji fazowej")
    else:
        print("✓ Stabilne wyniki kątowe - system działa poprawnie")
    print("="*50)

if __name__ == "__main__":
    main()