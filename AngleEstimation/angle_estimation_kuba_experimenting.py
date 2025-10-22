import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound, minimize
from adi import ad9361
from adi.cn0566 import CN0566
import adi
import threading
import datetime
from pathlib import Path

ESP32_IP = "192.168.0.108"
ESP32_PORT = 3333
MEASUREMENTS = 320    
STEP_MULT = 5
STEP_SIZE_M = 0.000996875 #0.00018 * STEP_MULT

PHASE_CORRECTION_DEG = 124.3
PHASE_CORRECTION_RAD = np.deg2rad(PHASE_CORRECTION_DEG)
PHASE_CORRECTION_FACTOR = np.exp(-1j * PHASE_CORRECTION_RAD)

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
    c = 3e8  
    lambda_m = c / freq_hz  
    d_m = 0.02424  
    d_over_lambda = d_m / lambda_m  
    
    correlation = np.mean(data_ch0 * np.conj(data_ch1))
    phase_diff = np.angle(correlation)
    
    sin_theta = phase_diff / (2 * np.pi * d_over_lambda)
    
    if abs(sin_theta) > 1:
        sin_theta = np.clip(sin_theta, -1, 1)
    
    estimated_angle_rad = np.arcsin(sin_theta)
    estimated_angle_deg = np.rad2deg(estimated_angle_rad)
    
    return estimated_angle_deg, phase_diff, correlation

def a_sar(angle_deg, element_positions_mm, lambda_mm):
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi / lambda_mm
    return np.exp(1j * k * element_positions_mm * np.sin(angle_rad)).reshape(-1, 1)

def plot_measurement_analysis(measurements_data, freq_hz=10.3943359375e9):
    print("[INFO] Generowanie wykresów analizy...")
    
    downsample_factor = max(1, len(measurements_data['ch0']) // 20)  
    sample_indices = range(0, len(measurements_data['ch0']), downsample_factor)
    
    selected_ch0 = [measurements_data['ch0'][i] for i in sample_indices]
    selected_ch1 = [measurements_data['ch1'][i] for i in sample_indices]
    
    time_downsample = 10
    all_ch0 = np.concatenate([ch0[::time_downsample] for ch0 in selected_ch0])
    all_ch1 = np.concatenate([ch1[::time_downsample] for ch1 in selected_ch1])
    
    sample_rate = measurements_data['fs']
    
    angles = []
    for ch0, ch1 in zip(measurements_data['ch0'], measurements_data['ch1']):
        angle, _, _ = analyze_angle_estimation(ch0, ch1, freq_hz, sample_rate)
        angles.append(angle)
    
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    
    dt = 1 / sample_rate * time_downsample
    t_all = np.arange(len(all_ch0)) * dt * 1e6
    
    ch0_real_all = np.real(all_ch0)
    ch0_imag_all = np.imag(all_ch0)
    ch1_real_all = np.real(all_ch1)
    ch1_imag_all = np.imag(all_ch1)
    
    ch0_mag_all = np.abs(all_ch0)
    ch1_mag_all = np.abs(all_ch1)
    ch0_phase_all = np.angle(all_ch0)
    ch1_phase_all = np.angle(all_ch1)
    
    print(f"[INFO] Wykresy wygenerowane (użyto {len(all_ch0)} próbek z {len(np.concatenate(measurements_data['ch0']))} dostępnych)")

def unwrap_signal_along_range(sig):
    """
    Unwrap fazę sygnału kompleksowego wzdłuż osi próbek (range/time).
    Zwraca sygnał zrekonstruowany jako mag * exp(1j * phase_unwrapped).
    """
    mag = np.abs(sig)
    phase = np.angle(sig)
    phase_unwrapped = np.unwrap(phase)  # unwrap wzdłuż osi (1D)
    return mag * np.exp(1j * phase_unwrapped)

def range_compression(received_signal, chirp_signal, fs):
    N = len(received_signal)
    
    rec_fft = np.fft.fft(received_signal, n=N)
    chirp_fft = np.fft.fft(chirp_signal, n=N)
    
    matched_filter = np.conj(chirp_fft)
    
    compressed_fft = rec_fft * matched_filter
    
    compressed_signal = np.fft.ifft(compressed_fft)
    
    return compressed_signal

def MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=False):
    M, N = Y.shape
    R = (Y @ Y.conj().T) / N
    lambda_mm = 3e8 / freq_hz * 1e3

    ch0_idx = np.arange(0, M, 2)  
    ch1_idx = np.arange(1, M, 2)  

    Y_left = Y[ch0_idx, :]
    Y_right = Y[ch1_idx, :]

    positions_left = element_positions_mm[ch0_idx]
    positions_right = element_positions_mm[ch1_idx]

    Y_sorted = np.vstack((Y_left, Y_right)) 
    element_positions_sorted = np.concatenate((positions_left, positions_right))  

    Y = Y_sorted
    element_positions_mm = element_positions_sorted
    M = Y.shape[0]

    def cost_function(angle_deg):
        a_temp = a_sar(angle_deg, element_positions_mm, lambda_mm)
        a_temp_h = a_temp.conj().T
        denominator = a_temp_h @ a_temp

        if np.abs(denominator) < 1e-12:         
            return np.inf
        
        Pv = np.eye(M) - a_temp @ (1 / denominator) @ a_temp_h
        cost = np.abs(np.trace(Pv @ R))
        return cost

    # Wykomentowany fragment odpowiedzialny za rysowanie wykresu
    # if verbose:
    #     angle_vec = np.arange(-45, 44.1, 0.1)
    #     pval = np.array([cost_function(a) for a in angle_vec])
    #     plt.figure()
    #     plt.plot(angle_vec, pval)
    #     plt.xlabel("Kąt [deg]")
    #     plt.ylabel("Funkcja kosztu")
    #     plt.title("MLE z syntetycznej apertury")
    #     plt.grid(True)
    #     plt.show()

    result = minimize_scalar(cost_function, bounds=(-45, 44), method='bounded')
    return result.x

def analyze_mle_with_aperture_dual(measurements_data, freq_hz, verbose=False): # Zmieniono `verbose=True` na `verbose=False`
    try:
        M = len(measurements_data['ch0'])  
        N = len(measurements_data['ch0'][0])  

        ch0_stack = []
        ch1_stack = []
        positions_mm = []

        for i in range(M):
            ch0 = np.array(measurements_data['ch0'][i])
            ch1 = np.array(measurements_data['ch1'][i])
            ch0_stack.append(ch0)
            ch1_stack.append(ch1)

            pos_mm = measurements_data['positions'][i] * 1000  
            positions_mm.extend([pos_mm, pos_mm + 24.24])  

        Y = np.vstack(ch0_stack + ch1_stack)  
        element_positions_mm = np.array(positions_mm)  

        estimated_angle = MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=verbose)
        print(f"[WYNIK] Kąt DOA z syntetycznej apertury (2 kanały): {estimated_angle:.2f}°")
        return estimated_angle
    except Exception as e:
        print(f"[ERROR] Błąd podczas analizy MLE z 2-kanałowej syntetycznej apertury: {e}")
        return None
    
def generate_chirp_signal(sample_rate, duration_us=100, bandwidth_mhz=200, center_freq_mhz=1):
    duration_s = duration_us * 1e-6
    N = int(sample_rate * duration_s)
    
    N = 2**int(np.log2(N))
    print(f"[DEBUG] Długość sygnału chirp: {N} próbek ({N/sample_rate*1e6:.1f} μs)")
    
    t = np.linspace(0, duration_s, N)
    bandwidth_hz = bandwidth_mhz * 1e6
    center_freq_hz = center_freq_mhz * 1e6
    
    k = bandwidth_hz / duration_s  
    freq_inst = center_freq_hz + k * t
    
    chirp_signal = np.exp(1j * 2 * np.pi * (center_freq_hz * t + 0.5 * k * t**2))
    
    chirp_signal = chirp_signal * (2**13)
    chirp_signal = chirp_signal.astype(np.complex64)
    
    print(f"[DEBUG] Typ danych chirp: {chirp_signal.dtype}")
    print(f"[DEBUG] Kształt sygnału chirp: {chirp_signal.shape}")
    
    return chirp_signal, t

def setup_chirp_radar(sdr, phaser):
    print("[INFO] Konfigurowanie radaru chirp...")
    
    sample_rate = sdr.sample_rate
    chirp_duration_us = 500 # 40
    chirp_bandwidth_mhz = 500 # 15
    chirp_center_freq_mhz = 12145 # 10394
    
    chirp_signal, t_chirp = generate_chirp_signal(
        sample_rate, chirp_duration_us, chirp_bandwidth_mhz, chirp_center_freq_mhz
    )
    
    sdr.tx_enabled_channels = [0, 1]
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = -88
    sdr.tx_hardwaregain_chan1 = -0  
    sdr.tx_lo = sdr.rx_lo

    phaser.tx_trig_en = 0 
    phaser.enable = 0  
    
    try:
        phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
        phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
        phaser._gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)
    except:
        phaser.gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
        phaser.gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
        phaser.gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)
    
    print(f"[INFO] Chirp radar skonfigurowany:")
    print(f"  - Czas trwania chirp: {chirp_duration_us} μs")
    print(f"  - Szerokość pasma: {chirp_bandwidth_mhz} MHz")
    print(f"  - Częstotliwość środkowa: {chirp_center_freq_mhz} MHz")
    print(f"  - Moc Tx: {sdr.tx_hardwaregain_chan0} dBm")
    print(f"  - Włączone kanały Tx: {sdr.tx_enabled_channels}")
    
    return chirp_signal

def plot_sar_progress(measurements_data, positions, angles_basic, angles_mle=None):
    plt.figure(figsize=(15, 5))
    
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
    
    plt.subplot(1, 3, 3)
    if len(measurements_data['ch0']) > 0:
        if 'unwrapped_phases' in measurements_data:
            plt.plot(positions_cm, measurements_data['unwrapped_phases'], 'g-', linewidth=2, label='Faza unwrapped')
        else:
            phase_diffs = []
            for ch0, ch1 in zip(measurements_data['ch0'], measurements_data['ch1']):
                phase_diff = np.angle(np.mean(ch0 * np.conj(ch1)))
                phase_diffs.append(np.rad2deg(phase_diff))
            phase_diffs_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phase_diffs)))
            plt.plot(positions_cm, phase_diffs_unwrapped, 'g-', linewidth=2, label='Faza unwrapped')

        plt.xlabel('Pozycja [cm]')
        plt.ylabel('Różnica faz [°]')
        plt.title('Różnica faz CH0-CH1')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def backprojection(measurements_data, freq_hz, image_size_m=5.0, resolution_m=0.01):
    print("[INFO] Rozpoczynam SAR imaging (zoptymalizowany Backprojection) z DWOMA KANAŁAMI...")
    
    c = 3e8  
    lambda_m = c / freq_hz  
    d_m = 0.02424  # Rozstaw anten z analizy MLE, używamy go tutaj
    
    x_coords = np.arange(-image_size_m/2, image_size_m/2, resolution_m)
    y_coords = np.arange(0.1, image_size_m, resolution_m)
    
    aperture_positions = np.array(measurements_data['positions'])
    measurements_ch0 = np.array(measurements_data['ch0'])
    measurements_ch1 = np.array(measurements_data['ch1'])
    sample_rate = measurements_data['fs']
    
    time_bins = np.arange(measurements_ch0.shape[1]) / sample_rate
    range_bins = time_bins * c / 2
    
    image_plane_ch0 = np.zeros((len(y_coords), len(x_coords)), dtype=np.complex128)
    image_plane_ch1 = np.zeros((len(y_coords), len(x_coords)), dtype=np.complex128)
    
    for i, x_pos_center in enumerate(aperture_positions):
        # Pozycja dla CH0
        x_pos_ch0 = x_pos_center # Zakładamy, że x_pos_center to pozycja dla CH0
        measurement_vector_ch0 = measurements_ch0[i, :]
        
        # Obliczenia dla CH0
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        dist_grid_ch0 = np.sqrt((X_grid - x_pos_ch0)**2 + Y_grid**2)
        round_trip_distance_ch0 = 2 * dist_grid_ch0
        
        mag_ch0 = np.abs(measurement_vector_ch0)
        phase_ch0 = np.angle(measurement_vector_ch0)  # powinna być już unwrapped
        interp_mag_ch0 = np.interp(round_trip_distance_ch0, range_bins, mag_ch0)
        interp_phase_ch0 = np.interp(round_trip_distance_ch0, range_bins, phase_ch0)
        pixel_contribution_interpolated_ch0 = interp_mag_ch0 * np.exp(1j * interp_phase_ch0)

        
        phase_correction_ch0 = np.exp(1j * 2 * np.pi * round_trip_distance_ch0 / lambda_m)
        image_plane_ch0 += pixel_contribution_interpolated_ch0 * phase_correction_ch0

        # Pozycja dla CH1 (przesunięta względem CH0)
        x_pos_ch1 = x_pos_center + d_m 
        measurement_vector_ch1 = measurements_ch1[i, :]

        # Obliczenia dla CH1
        dist_grid_ch1 = np.sqrt((X_grid - x_pos_ch1)**2 + Y_grid**2)
        round_trip_distance_ch1 = 2 * dist_grid_ch1

        mag_ch1 = np.abs(measurement_vector_ch1)
        phase_ch1 = np.angle(measurement_vector_ch1)  # powinna być już unwrapped
        interp_mag_ch1 = np.interp(round_trip_distance_ch1, range_bins, mag_ch1)
        interp_phase_ch1 = np.interp(round_trip_distance_ch1, range_bins, phase_ch1)
        pixel_contribution_interpolated_ch1 = interp_mag_ch1 * np.exp(1j * interp_phase_ch1)

        
        phase_correction_ch1 = np.exp(1j * 2 * np.pi * round_trip_distance_ch1 / lambda_m)
        image_plane_ch1 += pixel_contribution_interpolated_ch1 * phase_correction_ch1
        
    # Łączenie obrazów z obu kanałów
    # Możemy je po prostu zsumować (koherentnie) i wziąć moduł
    combined_image_plane = image_plane_ch0 + image_plane_ch1
    image = np.abs(combined_image_plane)

    eps = 1e-12
    image_normalized = 20 * np.log10((image + eps) / (np.max(image) + eps))
    
    return image_normalized, x_coords, y_coords

def start_chirp_transmission(sdr, chirp_signal, stop_event):
    print("[INFO] Rozpoczynam ciągłe nadawanie chirp z niską mocą...")

    tx_buffer = np.tile(chirp_signal, 10)

    normal_tx_gain_ch0 = -88
    normal_tx_gain_ch1 = -0

    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = normal_tx_gain_ch0
    sdr.tx_hardwaregain_chan1 = normal_tx_gain_ch1

    sdr.tx([tx_buffer, np.zeros_like(tx_buffer)])

def main():
    stop_event = threading.Event()  # event do zatrzymania wątku

    try:
        try:
            print("[INFO] Próba połączenia z CN0566...")
            phaser = CN0566(uri="ip:phaser.local")
            sdr = ad9361(uri="ip:192.168.2.1")
        except:
            print("[INFO] Próba połączenia po localhost")
            phaser = CN0566(uri="ip:localhost")
            sdr = ad9361(uri="ip:192.168.2.1")
        
        phaser.sdr = sdr
        
        phaser.configure(device_mode="rx")
        phaser.gcal = [1.0] * 8
        phaser.pcal = [0.0] * 8

        for i in range(0, 8):
            phaser.set_chan_phase(i, 0)

        gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
        for i in range(0, len(gain_list)):
            phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

        sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
        sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
        sdr._ctrl.debug_attrs["initialize"].value = "1"
        
        sdr.rx_enabled_channels = [0, 1]
        sdr._rxadc.set_kernel_buffers_count(1)
        rx = sdr._ctrl.find_channel("voltage0")
        rx.attrs["quadrature_tracking_en"].value = "1"
        
        # Configuring SDR Rx
        sdr.sample_rate = int(30e6) # try 0.6e6
        sdr.rx_buffer_size = int(4 * 1024)
        sdr.rx_rf_bandwidth = int(10e6)
        sdr.gain_control_mode_chan0 = "manual"
        sdr.gain_control_mode_chan1 = "manual"
        sdr.rx_hardwaregain_chan0 = int(30)
        sdr.rx_hardwaregain_chan1 = int(30)
        sdr.rx_lo = int(2.0e9)
        sdr.filter = "LTE20_MHz.ftr"
        
        phaser.SignalFreq = 10.3943359375e9
        phaser.lo = int(phaser.SignalFreq) + sdr.rx_lo
        
        
        
        phaser.set_beam_phase_diff(0.0)
        phaser.Averages = 16
        
        c = 3e8
        lambda_m = c / phaser.SignalFreq
        d_m = 24.25e-3
        d_over_lambda = d_m / lambda_m
        
        print(f"[INFO] Częstotliwość: {phaser.SignalFreq/1e9:.3f} GHz")
        print(f"[INFO] Długość fali: {lambda_m*100:.2f} cm")
        print(f"[INFO] Rozstaw anten: {d_m*100:.1f} cm")
        print(f"[INFO] Rozmiar kroku: {STEP_SIZE_M*1000:.1f} mm")
        print(f"[INFO] d/λ = {d_over_lambda:.3f}")
        
        time.sleep(0.1)
        
        print("\n" + "="*50)
        print("KONFIGURACJA RADARU CHIRP")
        print("="*50)
        
        chirp_signal = setup_chirp_radar(sdr, phaser)
        tx_thread = threading.Thread(target=start_chirp_transmission, args=(sdr, chirp_signal, stop_event))
        tx_thread.start()

        print("\n" + "="*50)
        print("ROZPOCZYNAM POMIARY SAR")
        print("="*50)
        
        run_sar_measurements(phaser, sdr, chirp_signal)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C wykryty, zatrzymywanie transmisji...")
    finally:
        # Zatrzymaj wątek nadawania
        stop_event.set()
        tx_thread.join()

        # Zamknij SDR
        try:
            sdr.tx_destroy_buffer()
            sdr.rx_destroy_buffer()
            sdr.rx_enabled_channels = []
            sdr.tx_enabled_channels = []
            sdr.close()
            print("[INFO] SDR zamknięty poprawnie.")
        except:
            print("[WARN] Problem przy zamykaniu SDR.")

        # Zamknij phasera
        try:
            phaser.close()
            print("[INFO] Phaser zamknięty poprawnie.")
        except:
            print("[WARN] Problem przy zamykaniu phasera.")

        print("[INFO] Program zakończony.")

def run_sar_measurements(phaser, sdr, chirp_signal):
    """
    Wykonuje pomiary SAR, analizuje dane i zapisuje wyniki do pliku NPZ.
    """
    print("[INFO] Rozpoczynam pomiary SAR z aktywnym nadajnikiem...")
    
    sock = connect_esp32()
    time.sleep(1)
    
    raw_measurements = {
        'ch0': [],
        'ch1': [],
        'positions': [],
        'fs': sdr.sample_rate
    }
    
    print(f"\n[INFO] Rozpoczynam {MEASUREMENTS} pomiarów...")
    
    strong_tx_gain_ch0 = -88
    strong_tx_gain_ch1 = 0
    normal_tx_gain_ch0 = sdr.tx_hardwaregain_chan0
    normal_tx_gain_ch1 = sdr.tx_hardwaregain_chan1
    
    for i in range(MEASUREMENTS):
        print(f"\n[{i+1}/{MEASUREMENTS}] Pomiar...")
        
        if i > 0:
            send_step_and_wait(sock)
            time.sleep(0.1)
        
        current_position = i * STEP_SIZE_M

        sdr.tx_hardwaregain_chan0 = strong_tx_gain_ch0
        sdr.tx_hardwaregain_chan1 = strong_tx_gain_ch1
        time.sleep(0.05)

        ch0, ch1 = acquire_data(sdr)

        raw_measurements['ch0'].append(ch0)
        raw_measurements['ch1'].append(ch1)
        raw_measurements['positions'].append(current_position)
        
        sdr.tx_hardwaregain_chan0 = normal_tx_gain_ch0
        sdr.tx_hardwaregain_chan1 = normal_tx_gain_ch1
        
        print(f"[INFO] Pozycja: {current_position*100:.1f} cm")
        
        time.sleep(0.05)
    
    sock.close()
    print("\n[INFO] Pomiary surowych danych zakończone, rozpoczynam kompresję i analizę...")
    
    measurements_data = {
        'ch0': [],
        'ch1': [],
        'positions': raw_measurements['positions'],
        'fs': raw_measurements['fs'],
        'mle_angles': [],
        'basic_angles': []
    }
    
    for i in range(len(raw_measurements['ch0'])):
        raw_ch0 = raw_measurements['ch0'][i]
        raw_ch1 = raw_measurements['ch1'][i]
        
        compressed_ch0 = range_compression(raw_ch0, chirp_signal, sdr.sample_rate)
        compressed_ch1 = range_compression(raw_ch1, chirp_signal, sdr.sample_rate)

        compressed_ch1 = compressed_ch1 * PHASE_CORRECTION_FACTOR

        compressed_ch0_unwrapped = unwrap_signal_along_range(compressed_ch0)
        compressed_ch1_unwrapped = unwrap_signal_along_range(compressed_ch1)
        
        measurements_data['ch0'].append(compressed_ch0_unwrapped)
        measurements_data['ch1'].append(compressed_ch1_unwrapped)

        angle_basic, phase_diff, correlation = analyze_angle_estimation(
            compressed_ch0_unwrapped, compressed_ch1_unwrapped, phaser.SignalFreq, sdr.sample_rate
        )

        # unwrap całej sekwencji faz (a nie pojedynczej wartości)
        if 'phase_diffs' not in measurements_data:
            measurements_data['phase_diffs'] = []

        measurements_data['phase_diffs'].append(phase_diff)
        measurements_data['basic_angles'].append(angle_basic)

        # unwrap na bieżąco
        unwrapped_phases = np.unwrap(measurements_data['phase_diffs'])
        measurements_data['unwrapped_phases'] = np.rad2deg(unwrapped_phases)


        power_ch0 = np.mean(np.abs(compressed_ch0)**2)
        power_ch1 = np.mean(np.abs(compressed_ch1)**2)

        print(f"[INFO] Pomiar {i+1}: Kąt podstawowy: {angle_basic:.1f}°, Moc CH0: {10*np.log10(power_ch0+1e-12):.1f} dB, Moc CH1: {10*np.log10(power_ch1+1e-12):.1f} dB")

    # Po zebraniu wszystkich phase_diff (radiany) — wykonaj unwrap wzdłuż apertury
    # phase_diffs_rad = measurements_data.get('phase_diffs_rad', [])
    # if len(phase_diffs_rad) > 0:
    #     phase_diffs_unwrapped_rad = np.unwrap(np.array(phase_diffs_rad))
    #     phase_diffs_unwrapped_deg = np.rad2deg(phase_diffs_unwrapped_rad)
    #     measurements_data['unwrapped_phases'] = phase_diffs_unwrapped_deg.tolist()
    # else:
    #     measurements_data['unwrapped_phases'] = []

    plot_measurement_analysis(measurements_data, phaser.SignalFreq)

    print("\n[INFO] Analiza MLE SAR z syntetycznej apertury (kanały 0 i 1)...")
    angle_sar_aperture_dual = analyze_mle_with_aperture_dual(measurements_data, phaser.SignalFreq, verbose=True)
    if angle_sar_aperture_dual is not None:
        print(f"[WYNIKI] MLE SAR (syntetyczna apertura, 2 kanały): {angle_sar_aperture_dual:.2f}°")
    
    if len(measurements_data['ch0']) >= 2:
        print("\n" + "="*50)
        print("SAR IMAGING")
        print("="*50)
        
        sar_image, x_coords, y_coords = backprojection(
            measurements_data, 
            phaser.SignalFreq,
            image_size_m=5,
            resolution_m=0.005
        )
        
        plt.figure(figsize=(8, 8))
        plt.imshow(sar_image, cmap='gray', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], origin='lower')
        plt.title('Obraz SAR (Backprojection)')
        plt.xlabel('Pozycja wzdłuż apertury [m]')
        plt.ylabel('Odległość od radaru [m]')
        plt.colorbar(label='Amplituda [dB]')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

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

    # --- POPRAWIONY BLOK KODU: ZAPISYWANIE DANYCH DO PLIKU NPZ ---
    print("\n[INFO] Zapisywanie surowych danych do pliku NPZ...")
    
    # Tworzenie nazwy pliku na podstawie daty i godziny, bez folderu
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_sar_measurements_{timestamp}.npz"
    
    # Słownik z danymi do zapisu
    # Zgodnie z prośbą, tylko surowe dane ch0 i ch1
    data_to_save = {
        'ch0': np.array(raw_measurements['ch0']),
        'ch1': np.array(raw_measurements['ch1']),
        'positions': np.array(raw_measurements['positions']),
        'fs': raw_measurements['fs']
    }
    
    try:
        np.savez_compressed(filename, **data_to_save)
        print(f"[SUKCES] Surowe dane pomiarowe zapisano do pliku: {filename}")
    except Exception as e:
        print(f"[BŁĄD] Nie udało się zapisać pliku: {e}")
    
    return

if __name__ == "__main__":
    main()