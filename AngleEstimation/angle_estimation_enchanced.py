import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound, minimize
from adi import ad9361
from adi.cn0566 import CN0566

ESP32_IP = "192.168.0.105"
ESP32_PORT = 3333
MEASUREMENTS = 330     # ile kroków i pomiarów
STEP_SIZE_M = 0.0001  # rozmiar kroku w metrach (0.1mm)

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
    d_m = 0.02424  # rozstaw anten w metrach
    d_over_lambda = d_m / lambda_m  # normalizowany rozstaw
    
    correlation = np.mean(data_ch0 * np.conj(data_ch1))

    calibrated_correlation = correlation * np.exp(-1j * np.deg2rad(90))
    
    # Różnica fazowa między kanałami
    phase_diff = np.angle(calibrated_correlation)
    
    # Estymacja kąta na podstawie różnicy fazowej
    sin_theta = phase_diff / (2 * np.pi * d_over_lambda)
    
    if abs(sin_theta) > 1:
        sin_theta = np.clip(sin_theta, -1, 1)
    
    estimated_angle_rad = np.arcsin(sin_theta)
    estimated_angle_deg = np.rad2deg(estimated_angle_rad)
    
    return estimated_angle_deg, phase_diff, calibrated_correlation

def a_sar(angle_deg, element_positions_mm, lambda_mm):           
    """
    Steering vector dla SAR
    """
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi / lambda_mm
    return np.exp(1j * k * element_positions_mm * np.sin(angle_rad)).reshape(-1, 1)

def MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=False):
    """
    MLE estymacja kąta z danych ze wszystkich pozycji
    """
    M, N = Y.shape
    R = (Y @ Y.conj().T) / N
    lambda_mm = 3e8 / freq_hz * 1e3

    ch0_idx = np.arange(0, M, 2)  # Indeksy lewych anten
    ch1_idx = np.arange(1, M, 2)  # Indeksy prawych anten

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
    """
    try:
        M = len(measurements_data['ch0'])  # liczba pozycji
        N = len(measurements_data['ch0'][0])  # liczba próbek

        calibration_phase_deg = 90
        calibration_factor = np.exp(1j * np.deg2rad(calibration_phase_deg))

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

        Y = np.vstack(ch0_stack + ch1_stack)  
        element_positions_mm = np.array(positions_mm)  

        # Estymacja kąta
        estimated_angle = MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=verbose)
        print(f"Kąt DOA z syntetycznej apertury: {estimated_angle:.2f}°")
        return estimated_angle
    except Exception as e:
        print(f"[ERROR] Błąd podczas analizy MLE z 2-kanałowej syntetycznej apertury: {e}")
        return None

def generate_chirp(fs, t_pulse, bandwidth):
    t = np.arange(0, t_pulse, 1/fs)
    k = bandwidth / t_pulse  # chirp rate
    chirp = np.exp(1j * np.pi * k * t**2)  # complex LFM
    return chirp

def main():
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
    
    c = 3e8
    lambda_m = c / phaser.SignalFreq
    d_m = 24.25e-3
    d_over_lambda = d_m / lambda_m
    
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
        'mle_angles': [], 
        'basic_angles': [] 
    }

    print(f"\nRozpoczynam {MEASUREMENTS} pomiarów...")
    
    # Główna pętla pomiarowa
    for i in range(MEASUREMENTS):
        print(f"\n[{i+1}/{MEASUREMENTS}] Pomiar...")
        
        if i > 0: 
            send_step_and_wait(sock)
            time.sleep(0.05)
        
        current_position = i * STEP_SIZE_M
        
        ch0, ch1 = acquire_data(sdr)

        measurements_data['ch0'].append(ch0)
        measurements_data['ch1'].append(ch1)
        measurements_data['positions'].append(current_position)
        
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
    print("\nPomiary zakończone, rozpoczynam analizę...")

    # Saving measurment data to .npz file
    save_data = True
    if save_data:
        # np.savez_compressed(f"measurements_{timestamp}.npz", **measurements_data)
        np.savez_compressed(f"measurements/30_deg_sar.npz", **measurements_data)
        print(f"Saved measurements")

    print("\n[INFO] Analiza MLE SAR z syntetycznej apertury (kanały 0 i 1)...")
    angle_sar_aperture_dual = analyze_mle_with_aperture_dual(measurements_data, phaser.SignalFreq, verbose=True)
    if angle_sar_aperture_dual is not None:
        print(f"[WYNIKI] MLE SAR (syntetyczna apertura, 2 kanały): {angle_sar_aperture_dual:.2f}°")

    

if __name__ == "__main__":
    main()