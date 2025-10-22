import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fminbound, minimize
import threading

MEASUREMENTS = 320
STEP_MULT = 5
STEP_SIZE_M = 0.000996875 
SIGNAL_FREQ = 10.3943359375e9

PHASE_CORRECTION_DEG = 30
PHASE_CORRECTION_RAD = np.deg2rad(PHASE_CORRECTION_DEG)
PHASE_CORRECTION_FACTOR = np.exp(-1j * PHASE_CORRECTION_RAD)

# ----------- ESP 32 handling -----------

def send_step_and_wait():
    print("[INFO] ESP32 wykonał krok")

# ----------- ESP 32 handling -----------

def acquire_data_from_file(filename="raw_sar_measurements_20250917_153648.npz", measurement_idx=0):
    """
    Odczytuje dane z pliku numpy zamiast z SDR
    
    Args:
        filename (str): Nazwa pliku z danymi
        measurement_idx (int): Indeks pomiaru do pobrania (0 to pierwszy pomiar)
    
    Returns:
        tuple: (ch0, ch1) - dane dla wybranego pomiaru
    """
    try:
        # Wczytanie danych z pliku
        data = np.load(filename)
        ch0_all = data['ch0']
        ch1_all = data['ch1']
        
        # Pobranie konkretnego pomiaru
        if measurement_idx >= len(ch0_all):
            raise IndexError(f"Indeks pomiaru {measurement_idx} przekracza dostępne pomiary (0-{len(ch0_all)-1})")
        
        ch0 = ch0_all[measurement_idx]
        ch1 = ch1_all[measurement_idx]
        
        print(f"Wczytano pomiar {measurement_idx} z pliku {filename}")
        print(f"Kształt ch0: {ch0.shape}, ch1: {ch1.shape}")
        
        return ch0, ch1
        
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {filename}")
        return None, None
    except Exception as e:
        print(f"Błąd podczas wczytywania danych: {e}")
        return None, None

def acquire_data_from_text(filename_ch0="ch0_data.txt", filename_ch1="ch1_data.txt", 
                          num_measurements=15, num_samples=512):
    """
    Alternatywna metoda odczytu z plików tekstowych
    
    Args:
        filename_ch0, filename_ch1 (str): Nazwy plików z danymi
        num_measurements (int): Liczba pomiarów
        num_samples (int): Liczba próbek na pomiar
    
    Returns:
        tuple: (ch0_all, ch1_all) - wszystkie dane jako listy
    """
    try:
        # Odczyt ch0
        ch0_data = []
        with open(filename_ch0, 'r') as f:
            lines = f.readlines()
            
        # Rekonstrukcja danych ch0
        ch0_all = []
        for m in range(num_measurements):
            measurement = []
            for s in range(num_samples):
                line_idx = m * num_samples + s
                if line_idx < len(lines):
                    parts = lines[line_idx].strip().split()
                    real_part = float(parts[0])
                    imag_part = float(parts[1])
                    measurement.append(complex(real_part, imag_part))
            ch0_all.append(np.array(measurement, dtype=np.complex64))
        
        # Odczyt ch1 (analogicznie)
        with open(filename_ch1, 'r') as f:
            lines = f.readlines()
            
        ch1_all = []
        for m in range(num_measurements):
            measurement = []
            for s in range(num_samples):
                line_idx = m * num_samples + s
                if line_idx < len(lines):
                    parts = lines[line_idx].strip().split()
                    real_part = float(parts[0])
                    imag_part = float(parts[1])
                    measurement.append(complex(real_part, imag_part))
            ch1_all.append(np.array(measurement, dtype=np.complex64))
        
        print(f"Wczytano {len(ch0_all)} pomiarów z plików tekstowych")
        return ch0_all, ch1_all
        
    except FileNotFoundError as e:
        print(f"Nie znaleziono pliku: {e}")
        return None, None
    except Exception as e:
        print(f"Błąd podczas wczytywania danych tekstowych: {e}")
        return None, None

def acquire_all_data_from_file(filename="raw_sar_measurements_20250912_145938.npz"):
    """
    Odczytuje WSZYSTKIE dane z pliku
    
    Returns:
        tuple: (ch0_all, ch1_all) - listy wszystkich pomiarów
    """
    try:
        # Wczytanie danych z pliku
        data = np.load(filename)
        ch0_all = data['ch0']  # Array shape: (15, 512)
        ch1_all = data['ch1']  # Array shape: (15, 512)
        
        print(f"Wczytano wszystkie dane z pliku {filename}")
        print(f"Kształt ch0_all: {ch0_all.shape}, ch1_all: {ch1_all.shape}")
        print(f"Liczba pomiarów: {len(ch0_all)}")
        
        # Konwersja do list (jeśli potrzebujesz kompatybilności z oryginalnym kodem)
        ch0_list = [ch0_all[i] for i in range(len(ch0_all))]
        ch1_list = [ch1_all[i] for i in range(len(ch1_all))]
        
        return ch0_list, ch1_list
        
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {filename}")
        return None, None
    except Exception as e:
        print(f"Błąd podczas wczytywania danych: {e}")
        return None, None
    except Exception as e:
        print(f"Błąd podczas wczytywania danych: {e}")
        return None, None

# Funkcja kompatybilna z oryginalnym API
def acquire_data():
    return acquire_all_data_from_file("raw_sar_measurements_20250917_153648.npz")

# ----------- Acquire simulated data END -------------

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
    
    # --- Wykres 1: Skompresowany sygnał (główna opcja) ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.abs(compressed_signal))
    # plt.title('Sygnał po kompresji impulsów')
    # plt.xlabel('Próbki')
    # plt.ylabel('Amplituda')
    # plt.grid(True)
    # plt.show()

    # --- Opcja 2 (wykomentowana): Widma sygnałów przed mnożeniem ---
    # Freq = np.fft.fftfreq(N, d=1/fs)
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(Freq / 1e6, np.abs(rec_fft))
    # plt.title('Widmo sygnału odebranego (przed kompresją)')
    # plt.xlabel('Częstotliwość [MHz]')
    # plt.ylabel('Amplituda')
    # plt.grid(True)
    
    # plt.subplot(2, 1, 2)
    # plt.plot(Freq / 1e6, np.abs(matched_filter))
    # plt.title('Widmo filtra dopasowanego (sprzężone widmo chirpa)')
    # plt.xlabel('Częstotliwość [MHz]')
    # plt.ylabel('Amplituda')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    return compressed_signal

def analyze_mle_with_aperture_dual(measurements_data, freq_hz, verbose=False):
    try:
        M = len(measurements_data['ch0'])  # liczba pomiarów (15)
        N = len(measurements_data['ch0'][0])  # liczba próbek na pomiar (512)

        # POPRAWNA struktura macierzy Y - przeplatane kanały dla każdego pomiaru
        Y_interleaved = []
        element_positions_mm = []

        for i in range(M):
            ch0 = np.array(measurements_data['ch0'][i])
            ch1 = np.array(measurements_data['ch1'][i])
            
            # Dodaj oba kanały dla pomiaru i
            Y_interleaved.append(ch0)  # wiersz 2*i
            Y_interleaved.append(ch1)  # wiersz 2*i+1
            
            # Pozycje elementów dla pomiaru i
            pos_mm = measurements_data['positions'][i]  # TODO sprawdzic czy tutaj * 1000 czy nie
            element_positions_mm.extend([pos_mm, pos_mm + 0.02424])  # CH0, potem CH1

        Y = np.vstack(Y_interleaved)  # Shape: (30, 512)
        element_positions_mm = np.array(element_positions_mm)

        print(f"[DEBUG] Struktura macierzy Y:")
        print(f"  Shape: {Y.shape}")
        print(f"  Pozycje elementów (mm): {element_positions_mm[:6]}...")  # pierwsze 6 pozycji
        
        estimated_angle = MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=verbose)
        print(f"[WYNIK] Kąt DOA z syntetycznej apertury (2 kanały): {estimated_angle:.2f}°")
        return estimated_angle
        
    except Exception as e:
        print(f"[ERROR] Błąd podczas analizy MLE z 2-kanałowej syntetycznej apertury: {e}")
        return None

def MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=False):
    total_elements, N = Y.shape  # total_elements = 2*M
    R = (Y @ Y.conj().T) / N
    lambda_mm = 3e8 / freq_hz * 1e3

    def cost_function(angle_deg):
        a_temp = a_sar(angle_deg, element_positions_mm, lambda_mm)
        a_temp_h = a_temp.conj().T
        denominator = a_temp_h @ a_temp

        if np.abs(denominator) < 1e-12:         
            return np.inf
        
        Pv = np.eye(total_elements) - a_temp @ (1 / denominator) @ a_temp_h
        cost = np.abs(np.trace(Pv @ R))
        return cost

    # Opcjonalny wykres funkcji kosztu
    if verbose:
        print(f"[DEBUG] Analiza MLE dla {total_elements} elementów syntetycznej apertury")
        print(f"[DEBUG] Zakres pozycji: {element_positions_mm.min():.1f} - {element_positions_mm.max():.1f} mm")
        
        # Możesz odkomentować poniższy kod, aby zobaczyć wykres funkcji kosztu
        # angle_vec = np.arange(-45, 44.1, 0.1)
        # pval = np.array([cost_function(a) for a in angle_vec])
        # plt.figure()
        # plt.plot(angle_vec, pval)
        # plt.xlabel("Kąt [deg]")
        # plt.ylabel("Funkcja kosztu")
        # plt.title("MLE z syntetycznej apertury")
        # plt.grid(True)
        # plt.show()

    result = minimize_scalar(cost_function, bounds=(-45, 44), method='bounded')
    return result.x  

def generate_chirp_signal(sample_rate, duration_us=100, bandwidth_mhz=10, center_freq_mhz=1):
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

    # plt.figure(figsize=(10, 6))
    # plt.plot(t * 1e6, chirp_signal.real, label='Część rzeczywista (I)')
    # plt.plot(t * 1e6, chirp_signal.imag, label='Część urojona (Q)')
    # plt.title('Sygnał Chirp w dziedzinie czasu')
    # plt.xlabel('Czas [μs]')
    # plt.ylabel('Amplituda')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    print(f"[DEBUG] Typ danych chirp: {chirp_signal.dtype}")
    print(f"[DEBUG] Kształt sygnału chirp: {chirp_signal.shape}")

    
    return chirp_signal, t

def setup_chirp_radar():
    print("[INFO] Konfigurowanie radaru chirp...")
    
    sample_rate = int(30e6)
    chirp_duration_us = 40
    chirp_bandwidth_mhz = 15
    chirp_center_freq_mhz = 10394
    
    chirp_signal, t_chirp = generate_chirp_signal(
        sample_rate, chirp_duration_us, chirp_bandwidth_mhz, chirp_center_freq_mhz
    )
    
    # sdr.tx_enabled_channels = [0, 1]
    # sdr.tx_cyclic_buffer = True
    # sdr.tx_hardwaregain_chan0 = -88
    # sdr.tx_hardwaregain_chan1 = -10  
    # sdr.tx_lo = sdr.rx_lo

    # phaser.tx_trig_en = 0 
    # phaser.enable = 0  
    
    # phaser._gpios.gpio_tx_sw = 0
    # phaser._gpios.gpio_vctrl_1 = 1
    # phaser._gpios.gpio_vctrl_2 = 1
    
    print(f"[INFO] Chirp radar skonfigurowany:")
    print(f"  - Czas trwania chirp: {chirp_duration_us} μs")
    print(f"  - Szerokość pasma: {chirp_bandwidth_mhz} MHz")
    print(f"  - Częstotliwość środkowa: {chirp_center_freq_mhz} MHz")
    print(f"  - Moc Tx: -88 dBm")
    print(f"  - Włączone kanały Tx: 0, 1")
    
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

def backprojection(measurements_data, freq_hz, image_size_m=1.0, resolution_m=0.01):
    """
    Poprawiony algorytm backprojection dla SAR imaging.
    """
    print("[INFO] Rozpoczynam poprawiony SAR imaging (Backprojection)...")
    
    c = 3e8
    lambda_m = c / freq_hz

    x_coords = np.arange(-image_size_m/2, image_size_m/2, resolution_m)
    y_coords = np.arange(0.1, 1.75, resolution_m)
    
    aperture_positions = np.array(measurements_data['positions'])
    measurements_ch0 = np.array(measurements_data['ch0'])
    sample_rate = measurements_data['fs']
    
    image_plane = np.zeros((len(y_coords), len(x_coords)), dtype=np.complex128)
    
    min_range = 0.05
    max_range = 2.0
    
    # Utworzenie osi odległości dla danych pomiarowych
    range_bins = np.linspace(min_range, max_range, measurements_ch0.shape[1])
    
    # Iteracja przez każdy piksel obrazu
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # Iteracja przez każdy pomiar (pozycję apertury)
            total_sum = 0 + 0j
            for k, x_pos_aperture in enumerate(aperture_positions):
                # Obliczenie odległości od anteny do piksela
                distance_to_pixel = np.sqrt((x - x_pos_aperture)**2 + y**2)
                
                # Interpolacja wartości sygnału dla danej odległości
                if distance_to_pixel >= min_range and distance_to_pixel <= max_range:
                    # Szukamy wartości pomiarowej odpowiadającej tej odległości
                    measured_value = np.interp(distance_to_pixel, range_bins, measurements_ch0[k, :])
                    
                    # Korekcja fazowa (faza powrotna)
                    # Wypadało by tutaj skorygować fazę o 2pi*R/lambda
                    phase_correction = np.exp(-1j * 2 * np.pi / lambda_m * 2 * distance_to_pixel)
                    
                    # Sumowanie skorygowanej wartości do piksela
                    total_sum += measured_value * phase_correction
            
            image_plane[i, j] = total_sum
            
    # Konwersja do dB
    image_magnitude = np.abs(image_plane)
    image_db = 20 * np.log10(image_magnitude / np.max(image_magnitude))
    image_db = np.maximum(image_db, -50)
    
    return image_db, x_coords, y_coords

def apply_additional_processing(measurements_data, freq_hz):
    """
    Dodatkowe przetwarzanie sygnału przed backprojection
    """
    print("[INFO] Zastosowanie dodatkowego przetwarzania sygnału...")
    
    processed_data = {
        'ch0': [],
        'ch1': [],
        'positions': measurements_data['positions'],
        'fs': measurements_data['fs']
    }
    
    for i in range(len(measurements_data['ch0'])):
        ch0 = measurements_data['ch0'][i]
        ch1 = measurements_data['ch1'][i]
        
        # 1. Filtrowanie pasmowe (opcjonalne)
        # Można zastosować filtr dla konkretnego zakresu częstotliwości
        
        # 2. Korekcja amplitudy (normalizacja)
        ch0_norm = ch0 / (np.max(np.abs(ch0)) + 1e-12)
        ch1_norm = ch1 / (np.max(np.abs(ch1)) + 1e-12)
        
        # 3. Usunięcie składowej stałej
        ch0_ac = ch0_norm - np.mean(ch0_norm)
        ch1_ac = ch1_norm - np.mean(ch1_norm)
        
        # 4. Okno czasowe dla redukcji efektów bocznych
        window = np.hanning(len(ch0_ac))
        ch0_windowed = ch0_ac * window
        ch1_windowed = ch1_ac * window
        
        processed_data['ch0'].append(ch0_windowed)
        processed_data['ch1'].append(ch1_windowed)
    
    return processed_data

def enhanced_range_compression(received_signal, chirp_signal, fs):
    """
    Ulepszona kompresja impulsów z lepszą obsługą fazą
    """
    N = len(received_signal)
    
    # Dopasowanie długości sygnałów
    if len(chirp_signal) != N:
        if len(chirp_signal) < N:
            chirp_signal = np.pad(chirp_signal, (0, N - len(chirp_signal)), 'constant')
        else:
            chirp_signal = chirp_signal[:N]
    
    # FFT obu sygnałów
    rec_fft = np.fft.fft(received_signal, n=N)
    chirp_fft = np.fft.fft(chirp_signal, n=N)
    
    # Filtr dopasowany ze stabilizacją numeryczną
    chirp_power = np.abs(chirp_fft)**2
    matched_filter = np.conj(chirp_fft) / (chirp_power + np.max(chirp_power) * 1e-6)
    
    # Kompresja
    compressed_fft = rec_fft * matched_filter
    compressed_signal = np.fft.ifft(compressed_fft)
    
    return compressed_signal

def plot_sar_results(image_db, x_coords, y_coords, measurements_data):
    """
    Kompleksowa wizualizacja wyników SAR
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Obraz SAR
    im1 = axes[0,0].imshow(image_db, cmap='gray', 
                          extent=[x_coords.min(), x_coords.max(), 
                                 y_coords.min(), y_coords.max()], 
                          origin='lower', aspect='equal')
    axes[0,0].set_title('Obraz SAR (Poprawiony Backprojection)')
    axes[0,0].set_xlabel('Pozycja wzdłuż apertury [m]')
    axes[0,0].set_ylabel('Odległość od radaru [m]')
    axes[0,0].grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(im1, ax=axes[0,0], label='Amplituda [dB]')
    
    # 2. Profil w kierunku range (Y)
    center_x_idx = len(x_coords) // 2
    range_profile = image_db[:, center_x_idx]
    axes[0,1].plot(y_coords, range_profile)
    axes[0,1].set_title('Profil Range (środek apertury)')
    axes[0,1].set_xlabel('Odległość [m]')
    axes[0,1].set_ylabel('Amplituda [dB]')
    axes[0,1].grid(True)
    
    # 3. Profil w kierunku azymut (X)
    # Znajdź maksimum w range i pokaż profil azymutowy
    max_range_idx = np.argmax(np.max(image_db, axis=1))
    azimuth_profile = image_db[max_range_idx, :]
    axes[1,0].plot(x_coords, azimuth_profile)
    axes[1,0].set_title(f'Profil Azymut (range = {y_coords[max_range_idx]:.2f}m)')
    axes[1,0].set_xlabel('Pozycja [m]')
    axes[1,0].set_ylabel('Amplituda [dB]')
    axes[1,0].grid(True)
    
    # 4. Mapa mocy sygnału wzdłuż apertury
    if 'ch0' in measurements_data and len(measurements_data['ch0']) > 0:
        positions_m = np.array(measurements_data['positions'])
        powers_ch0 = [20*np.log10(np.mean(np.abs(ch0))+1e-12) for ch0 in measurements_data['ch0']]
        powers_ch1 = [20*np.log10(np.mean(np.abs(ch1))+1e-12) for ch1 in measurements_data['ch1']]
        
        axes[1,1].plot(positions_m*100, powers_ch0, 'b-', label='CH0', linewidth=2)
        axes[1,1].plot(positions_m*100, powers_ch1, 'r-', label='CH1', linewidth=2)
        axes[1,1].set_title('Moc sygnału wzdłuż apertury')
        axes[1,1].set_xlabel('Pozycja [cm]')
        axes[1,1].set_ylabel('Moc [dB]')
        axes[1,1].legend()
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
def start_chirp_transmission(sdr, chirp_signal):
    print("[INFO] Rozpoczynam ciągłe nadawanie chirp z niską mocą...")

    tx_buffer = np.tile(chirp_signal, 10)

    normal_tx_gain_ch0 = -88
    normal_tx_gain_ch1 = -20

    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = normal_tx_gain_ch0
    sdr.tx_hardwaregain_chan1 = normal_tx_gain_ch1

    sdr.tx([tx_buffer, np.zeros_like(tx_buffer)])

def main():
    c = 3e8
    lambda_m = c / SIGNAL_FREQ
    d_m = 24.25e-3
    d_over_lambda = d_m / lambda_m
    
    print(f"[INFO] Częstotliwość: {SIGNAL_FREQ/1e9:.3f} GHz")
    print(f"[INFO] Długość fali: {lambda_m*100:.2f} cm")
    print(f"[INFO] Rozstaw anten: {d_m*100:.1f} cm")
    print(f"[INFO] Rozmiar kroku: {STEP_SIZE_M*1000:.1f} mm")
    print(f"[INFO] d/λ = {d_over_lambda:.3f}")
    
    # time.sleep(0.5)
    
    print("\n" + "="*50)
    print("KONFIGURACJA RADARU CHIRP")
    print("="*50)
    
    chirp_signal = setup_chirp_radar()

    # threading.Thread(target=start_chirp_transmission, args=(sdr, chirp_signal), daemon=True).start()

    print("\n" + "="*50)
    print("ROZPOCZYNAM POMIARY SAR")
    print("="*50)
    
    run_sar_measurements(chirp_signal)

def run_sar_measurements(chirp_signal):
    print("[INFO] Rozpoczynam pomiary SAR z aktywnym nadajnikiem...")
    
    raw_measurements = {
        'ch0': [],
        'ch1': [],
        'positions': [],
        'fs': int(30e6)
    }
    
    print(f"\n[INFO] Rozpoczynam {MEASUREMENTS} pomiarów...")
    
    ch0, ch1 = acquire_data()

    for i in range(MEASUREMENTS):
        print(f"\n[{i+1}/{MEASUREMENTS}] Pomiar...")
        
        current_position = i * STEP_SIZE_M

        raw_measurements['ch0'].append(ch0[i])
        raw_measurements['ch1'].append(ch1[i])
        raw_measurements['positions'].append(current_position)
        
        print(f"[INFO] Pozycja: {current_position*100:.1f} cm")
        
        # time.sleep(0.05)
    
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
        
        compressed_ch0 = range_compression(raw_ch0, chirp_signal, int(30e6))
        compressed_ch1 = range_compression(raw_ch1, chirp_signal, int(30e6))

        compressed_ch1 = compressed_ch1 * PHASE_CORRECTION_FACTOR

        compressed_ch0_unwrapped = unwrap_signal_along_range(compressed_ch0)
        compressed_ch1_unwrapped = unwrap_signal_along_range(compressed_ch1)
        
        measurements_data['ch0'].append(compressed_ch0_unwrapped)
        measurements_data['ch1'].append(compressed_ch1_unwrapped)

        angle_basic, phase_diff, correlation = analyze_angle_estimation(
            compressed_ch0_unwrapped, compressed_ch1_unwrapped, SIGNAL_FREQ, int(30e6)
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
    phase_diffs_rad = measurements_data.get('phase_diffs_rad', [])
    if len(phase_diffs_rad) > 0:
        phase_diffs_unwrapped_rad = np.unwrap(np.array(phase_diffs_rad))
        phase_diffs_unwrapped_deg = np.rad2deg(phase_diffs_unwrapped_rad)
        measurements_data['unwrapped_phases'] = phase_diffs_unwrapped_deg.tolist()
    else:
        measurements_data['unwrapped_phases'] = []

    # plot_measurement_analysis(measurements_data, SIGNAL_FREQ)

    print("\n[INFO] Analiza MLE SAR z syntetycznej apertury (kanały 0 i 1)...")
    angle_sar_aperture_dual = analyze_mle_with_aperture_dual(measurements_data, SIGNAL_FREQ, verbose=True)

    if angle_sar_aperture_dual is not None:                     # kąt dla MLE
        print(f"[WYNIKI] MLE SAR (syntetyczna apertura, 2 kanały): {angle_sar_aperture_dual:.2f}°")
    
    if len(measurements_data['ch0']) >= 2:
        print("\n" + "="*50)
        print("SAR IMAGING")
        print("="*50)
        
        sar_image, x_coords, y_coords = backprojection(
            measurements_data, 
            SIGNAL_FREQ,
            image_size_m=0.5,
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

    # mean_basic_angle = np.mean(measurements_data['basic_angles'])
    # std_basic_angle = np.std(measurements_data['basic_angles'])
    
    # print("\n" + "="*50)
    # print("PODSUMOWANIE WYNIKÓW SAR")
    # print("="*50)
    # print(f"Liczba pomiarów: {len(measurements_data['ch0'])}")
    # print(f"Zakres pozycji: 0 - {(len(measurements_data['ch0'])-1) * STEP_SIZE_M * 100:.1f} cm")
    # print(f"Rozmiar kroku: {STEP_SIZE_M * 1000:.1f} mm")
    # print("")
    # print("UWAGA O GEOMETRII SYSTEMU:")
    # print("- Nadajnik i odbiornik poruszają się razem (monostatyczny SAR)")
    # print("- Target: metalowa skrzynka w odległości ~50 cm")
    # print("- Kąty są mierzone względem platformy, nie względem targetu!")
    # print("")
    # print("WYNIKI ESTYMACJI KĄTÓW:")
    # print(f"Średni kąt (podstawowy): {mean_basic_angle:.1f}° ± {std_basic_angle:.1f}°")
    # if angle_sar_aperture_dual is not None:
    #     print(f"Kąt MLE (syntetyczna apertura): {angle_sar_aperture_dual:.1f}°")
    # print("")
    # print("INTERPRETACJA:")
    # if std_basic_angle > 10:
    #     print("⚠ Duża zmienność kątów może wynikać z:")
    #     print("  - Wielodrogowości sygnału (odbicia od różnych części targetu)")
    #     print("  - Ruchu platformy (efekt Dopplera)")
    #     print("  - Niedoskonałej kalibracji fazowej")
    # else:
    #     print("✓ Stabilne wyniki kątowe - system działa poprawnie")
    # print("="*50)

if __name__ == "__main__":
    main()