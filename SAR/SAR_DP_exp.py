
#!/usr/bin/env python3
"""
Offline SAR Backprojection Processing
Autor: Jakub Sawicki 2025
Oparty na kodzie pomiarowym SAR z CN0566
Wersja z obwiednią Hilberta i zachowaniem fazy
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import hilbert, medfilt
from scipy.ndimage import gaussian_filter1d

# ==========================================================
# ---- KONFIGURACJA ----
# ==========================================================
DATA_FILE = "./saved_measurments/300_20251029_115436.npz"
BW = 500e6           # Bandwidth [Hz]
ramp_time_s = 0.5e-3 # Ramp time [s] (0.5e3 us)
signal_freq = 100e3  # Intermediate Frequency [Hz]
fs = 0.6e6           # Sample rate [Hz]
c = 3e8              # Speed of light [m/s]
output_freq = 12.145e9  # Output frequency [Hz] - potrzebne do kompensacji fazy
STEP_SIZE_M = 0.0009844

# Parametry filtrowania
FILTER_METHOD = "median"  # "median", "gaussian", "hilbert", lub "none"
MEDIAN_KERNEL = 9         # 5, 7, 9, 11, 15 (musi być nieparzyste)
GAUSSIAN_SIGMA = 3        # 2-5 dla filtra Gaussa

# ==========================================================
# ---- FUNKCJE ----
# ==========================================================

def load_measurements(file_path):
    """Wczytaj zapisane dane pomiarowe z pliku .npz"""
    data = np.load(file_path, allow_pickle=True)
    print(f"\n✅ Wczytano plik: {file_path}")
    print(f"Zawiera {len(data['data_fft'])} pomiarów\n")
    return data


def apply_envelope_filter(fft_data, method="median", kernel_size=9, sigma=3):
    """
    Aplikuj filtr na amplitudę zachowując fazę
    
    Parametry:
    - method: "median", "gaussian", "hilbert", "none"
    - kernel_size: rozmiar kernela dla filtru medianowego (nieparzyste)
    - sigma: parametr dla filtru Gaussa
    """
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    
    if method == "median":
        magnitude_filtered = medfilt(magnitude, kernel_size=kernel_size)
        print(f"   ✓ Zastosowano filtr medianowy (kernel={kernel_size})")
    
    elif method == "gaussian":
        magnitude_filtered = gaussian_filter1d(magnitude, sigma=sigma)
        print(f"   ✓ Zastosowano filtr Gaussa (sigma={sigma})")
    
    elif method == "hilbert":
        # Obwiednia Hilberta z oryginalną fazą
        envelope = np.abs(hilbert(magnitude))
        magnitude_filtered = envelope
        print(f"   ✓ Zastosowano obwiednię Hilberta")
    
    elif method == "none":
        magnitude_filtered = magnitude
        print(f"   ✓ Brak filtrowania")
    
    else:
        raise ValueError(f"Nieznana metoda filtrowania: {method}")
    
    # Rekonstrukcja sygnału zespolonego z wygładzoną amplitudą i oryginalną fazą
    filtered_fft = magnitude_filtered * np.exp(1j * phase)
    
    return filtered_fft, magnitude, magnitude_filtered


def process_measurements_with_filter(measurements_data, filter_method, kernel_size, sigma):
    """Przetwarza wszystkie pomiary, aplikując filtr na amplitudę"""
    print(f"\n🔧 Przetwarzanie {len(measurements_data['data_fft'])} pomiarów z filtrowaniem...")
    
    processed_data = {
        'data_fft': [],
        'positions': measurements_data['positions']
    }
    
    for i, fft_raw in enumerate(measurements_data['data_fft']):
        filtered_fft, mag_orig, mag_filt = apply_envelope_filter(
            fft_raw, method=filter_method, kernel_size=kernel_size, sigma=sigma
        )
        processed_data['data_fft'].append(filtered_fft)
        
        if i == 0:  # Tylko raz wyświetl info
            pass  # Info jest już w apply_envelope_filter
    
    print(f"✅ Przetwarzanie zakończone\n")
    return processed_data


def smooth_fft(power_db, sigma=2):
    """Wygładzenie wykresu FFT dla wizualizacji"""
    return gaussian_filter1d(power_db, sigma=sigma)


def backprojection(measurements_data, BW, ramp_time_s, signal_freq, fs, output_freq,
                   azimuth_length_m=5, range_length_m=8,
                   resolution_azimuth_m=0.15, resolution_range_m=0.20,
                   use_phase_compensation=True):
    """
    Algorytm backprojection z opcjonalną kompensacją fazy
    
    Parametry:
    - use_phase_compensation: czy dodawać kompensację fazy (zalecane True)
    """
    print("🔹 Uruchamianie backprojection (offline)...")
    print(f"   Kompensacja fazy: {'TAK ✓' if use_phase_compensation else 'NIE'}")

    slope = BW / ramp_time_s
    
    azimuth_axis = np.arange(-azimuth_length_m/2, azimuth_length_m/2, resolution_azimuth_m)
    range_axis = np.arange(1, range_length_m, resolution_range_m)
    image = np.zeros((len(azimuth_axis), len(range_axis)), dtype=complex)
    
    antenna_positions = np.array(measurements_data['positions'])
    antenna_positions -= np.mean(antenna_positions)
    fft_data = measurements_data['data_fft']

    freq = np.linspace(-fs / 2, fs / 2, len(fft_data[0]))

    print(f"🧮 Siatka obrazu: {len(azimuth_axis)} x {len(range_axis)} pikseli")
    print(f"📡 Pozycje anten: {len(antenna_positions)}\n")

    for i, azim in enumerate(azimuth_axis):
        if i % 10 == 0:
            print(f" → Wiersz {i}/{len(azimuth_axis)}")

        for j, range_dist in enumerate(range_axis):
            pixel_value = 0 + 0j
            
            for k, ant_pos in enumerate(antenna_positions):
                # 1. Oblicz odległość geometryczną
                distance = np.sqrt((azim - ant_pos)**2 + range_dist**2)
                
                # 2. Kompensacja fazy (round-trip)
                if use_phase_compensation:
                    phase_shift = np.exp(-1j * 4 * np.pi * distance * output_freq / c)
                else:
                    phase_shift = 1.0
                
                # 3. Mapuj odległość na częstotliwość
                freq_value = (distance * 2 * slope / c) + signal_freq
                freq_index = int((freq_value + fs/2) / fs * len(freq))
                
                # 4. Sumowanie koherentne z kompensacją fazy
                if 0 <= freq_index < len(freq):
                    pixel_value += fft_data[k][freq_index] * phase_shift
            
            image[i, j] = pixel_value

    image_db = 20 * np.log10(np.abs(image) + 1e-15)

    plt.figure(figsize=(10, 8))
    plt.imshow(image_db.T,
               extent=[azimuth_axis[0], azimuth_axis[-1], range_axis[0], range_axis[-1]],
               aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Azimuth [m]')
    plt.ylabel('Range [m]')
    title = 'SAR Image - Backprojection (Offline)'
    if use_phase_compensation:
        title += ' + Phase Compensation'
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return image, image_db


def plot_first_fft_comparison(measurements_data, processed_data, BW, ramp_time_s, 
                               signal_freq, fs, range_length_m=8):
    """
    Wizualizacja porównania: oryginalny vs przefiltrowany pierwszy pomiar
    """
    first_fft_orig = measurements_data['data_fft'][0]
    first_fft_proc = processed_data['data_fft'][0]
    
    freq_axis = np.linspace(-fs/2, fs/2, len(first_fft_orig))
    slope = BW / ramp_time_s
    dist_axis = (freq_axis - signal_freq) * c / (2 * slope)
    mask = (dist_axis >= 0) & (dist_axis <= range_length_m)
    dist_plot = dist_axis[mask]
    
    # Amplitudy
    mag_orig = np.abs(first_fft_orig[mask])
    mag_proc = np.abs(first_fft_proc[mask])
    
    # Konwersja do dB
    power_db_orig = 20 * np.log10(mag_orig + 1e-15)
    power_db_proc = 20 * np.log10(mag_proc + 1e-15)
    
    # Fazy
    phase_orig = np.angle(first_fft_orig[mask])
    phase_proc = np.angle(first_fft_proc[mask])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Wykres 1: Amplituda
    ax1.plot(dist_plot, power_db_orig, alpha=0.4, label="Oryginalny", linewidth=1)
    ax1.plot(dist_plot, power_db_proc, linewidth=2, label="Przefiltrowany")
    ax1.set_xlabel("Range [m]")
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_title("Porównanie amplitudy - pierwszy pomiar")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wykres 2: Faza
    ax2.plot(dist_plot, phase_orig, alpha=0.5, label="Faza oryginalna", linewidth=1)
    ax2.plot(dist_plot, phase_proc, label="Faza po filtrowaniu", linewidth=1.5)
    ax2.set_xlabel("Range [m]")
    ax2.set_ylabel("Phase [rad]")
    ax2.set_title("Porównanie fazy (powinna być identyczna)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==========================================================
# ---- GŁÓWNY PROGRAM ----
# ==========================================================

if __name__ == "__main__":
    print("="*60)
    print("SAR BACKPROJECTION - Offline Processing")
    print("="*60)
    
    # Wczytaj dane
    measurements_data = load_measurements(DATA_FILE)
    
    # Przetwórz dane z filtrowaniem
    processed_data = process_measurements_with_filter(
        measurements_data, 
        filter_method=FILTER_METHOD,
        kernel_size=MEDIAN_KERNEL,
        sigma=GAUSSIAN_SIGMA
    )
    
    # Porównanie przed i po filtrowaniu
    plot_first_fft_comparison(measurements_data, processed_data, 
                              BW, ramp_time_s, signal_freq, fs)
    
    # Backprojection z kompensacją fazy
    image, image_db = backprojection(
        processed_data, BW, ramp_time_s, signal_freq, fs, output_freq,
        use_phase_compensation=True  # Zmień na False żeby zobaczyć różnicę
    )
    
    print("\n" + "="*60)
    print("✅ Przetwarzanie offline zakończone.")
    print("="*60)
