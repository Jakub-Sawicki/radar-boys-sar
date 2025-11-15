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
from scipy.signal import hilbert, detrend, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
import sys # Dodano do obsługi błędów

# ==========================================================
# ---- KONFIGURACJA ----
# ==========================================================

BW = 500e6           # Bandwidth [Hz]
ramp_time_s = 0.5e-3 # Ramp time [s] (0.5e3 us)
signal_freq = 100e3  # Intermediate Frequency [Hz]
fs = 0.6e6           # Sample rate [Hz]
c = 3e8              # Speed of light [m/s]
output_freq = 12.145e9  # Output frequency [Hz] - potrzebne do kompensacji fazy
STEP_SIZE_M = 0.0009844

fft_size = 1024 * 4
N_frame = fft_size
freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
slope = BW / ramp_time_s
dist = (freq - signal_freq) * c / (2 * slope)


# Ustawienia generacji obrazu

# --- NOWE USTAWIENIA DLA ODEJMOWANIA TŁA ---
DATA_FILE_SCENE = "measurments/330_7m_v1.npz"
DATA_FILE_BACKGROUND = "measurments/330_bez_obiektu.npz"
# --- KONIEC NOWYCH USTAWIEŃ ---

azimuth_length_m=2
range_length_m=14
resolution_azimuth_m=0.3
resolution_range_m=0.4

calibration_factor=1 # 3.8/5.5
vmin_val=None
vmax_val=None
vmin_val=80
# vmax_val=105

range_axis_start = 4

# ==========================================================
# ---- FUNKCJE ----
# ==========================================================

def load_measurements(file_path):
    """Wczytaj zapisane dane pomiarowe z pliku .npz"""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"\nWczytano plik: {file_path}")
        print(f"Zawiera {len(data['data_fft'])} pomiarów")
        return data
    except FileNotFoundError:
        print(f"\nBŁĄD: Nie znaleziono pliku: {file_path}")
        sys.exit(1)


def process_measurements(measurements_data):
    """Przetwarza wszystkie pomiary"""
    print(f"Przetwarzanie {len(measurements_data['data_fft'])} pomiarów z filtrowaniem HPF...")
    
    processed_data = {
        'data_fft': [],
        'positions': measurements_data['positions']
    }
    
    # Definicja filtra
    fs_nyquist = fs / 2
    cutoff_freq = 80e3 
    wn = cutoff_freq / fs_nyquist # Normalizowana częstotliwość (ok. 0.267)
    
    # --- POPRAWKA BŁĘDU ---
    # Używamy obliczonej wartości `wn`, a nie stałej `0.4`
    print(f"Używam filtra HPF Butterwortha, odcięcie: {cutoff_freq/1e3} kHz (Wn={wn:.3f})")
    b, a = butter(4, wn, btype='high') 
    # --- KONIEC POPRAWKI ---

    for i, data in enumerate(measurements_data['data_fft']):
        # Implementacja filtra
        y = filtfilt(b, a, data)

        # Okienkowanie
        win_funct = tukey(len(y), alpha=0.3)
        y = y * win_funct

        sp = np.fft.fftshift(np.fft.fft(y))
        processed_data['data_fft'].append(sp)
    
    print(f"Przetwarzanie zakończone\n")
    return processed_data


def backprojection(measurements_data, azimuth_length_m=2.4, range_length_m=11, resolution_azimuth_m=0.15, resolution_range_m=0.40):
    # Ta funkcja jest już poprawnie zaimplementowana dla bistatyki!
    # Nie wymaga zmian.
    
    print("Starting backprojection (BISTATIC MODE)")
    
    c = 3e8
    slope = BW / ramp_time_s
    
    # Pozycja TX względem RX (stacjonarny nadajnik)
    TX_pos = np.array([0.0, -0.2, 0.8])
    print(f"Pozycja stacjonarnego TX (X,Y,Z): {TX_pos}")

    azimuth_axis = np.arange(-azimuth_length_m/2, azimuth_length_m/2, resolution_azimuth_m)
    range_axis = np.arange(range_axis_start, range_length_m, resolution_range_m)
    
    image = np.zeros((len(azimuth_axis), len(range_axis)), dtype=complex)
    
    antenna_positions = np.array(measurements_data['positions'])
    antenna_positions -= np.mean(antenna_positions)
    
    fft_data = measurements_data['data_fft']
    
    print(f"Image grid: {len(azimuth_axis)} x {len(range_axis)} pixels")
    print(f"Processing {len(antenna_positions)} antenna positions...")

    freq_axis = np.linspace(-fs/2, fs/2, len(fft_data[0]))
    
    # GŁÓNA PĘTLA BACKPROJECTION
    for i, azim in enumerate(azimuth_axis):
        if i % 10 == 0:
            print(f"Processing azimuth row {i}/{len(azimuth_axis)}")
        
        for j, range_dist in enumerate(range_axis):
            pixel_value = 0 + 0j
            pixel_pos = np.array([azim, range_dist, 0.0]) # [X, Y, Z]

            # Dystans TX -> Pixel (stały dla tego piksela)
            dist_tx = np.linalg.norm(pixel_pos - TX_pos)
            
            for k, ant_pos_x in enumerate(antenna_positions):
                # Dystans Pixel -> RX (zmienny z pozycją anteny)
                rx_pos = np.array([ant_pos_x, 0.0, 0.0])
                dist_rx = np.linalg.norm(pixel_pos - rx_pos)

                # Całkowita droga bistatyczna
                total_distance = dist_tx + dist_rx

                # Kompensacja fazy (2*pi dla bistatyki)
                phase_shift = np.exp(1j * 2 * np.pi * total_distance * output_freq / c)
                
                # Mapowanie częstotliwości (oparte na total_distance)
                freq_value = (total_distance * slope / c) + signal_freq

                # Interpolacja
                if freq_value >= freq_axis[0] and freq_value <= freq_axis[-1]:
                    real_interp = np.interp(freq_value, freq_axis, np.real(fft_data[k]))
                    imag_interp = np.interp(freq_value, freq_axis, np.imag(fft_data[k]))
                    sample = real_interp + 1j * imag_interp
                    pixel_value += sample * phase_shift
            
            image[i, j] = pixel_value
    
    image_db = 20 * np.log10(np.abs(image) + 1e-15)
    return image_db, azimuth_axis, range_axis

def image_plot(image_db, azimuth_axis, range_axis, calibration_factor, vmin_val, vmax_val):
    # Ta funkcja nie wymaga zmian
    scaled_range_axis = range_axis * calibration_factor
    plot_kwargs = {
        'extent': [azimuth_axis[0], azimuth_axis[-1], 
                   scaled_range_axis[0], scaled_range_axis[-1]],
        'aspect': 'auto',
        'cmap': 'plasma',
        'origin': 'lower'
    }
    if vmin_val is not None:
        plot_kwargs['vmin'] = vmin_val
    if vmax_val is not None:
        plot_kwargs['vmax'] = vmax_val
    plt.figure(figsize=(10, 8))
    plt.imshow(image_db.T, **plot_kwargs)
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Azimuth [m]')
    plt.ylabel('Range [m]')
    plt.title('SAR Image - Backprojection (Coherent Background Subtraction)')
    plt.show()

def data_plot(processed_data):
    # Ta funkcja nie wymaga zmian
    global freq, dist
    spectrum = processed_data['data_fft'][0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot(dist, 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum))))
    ax1.set_title('Dystans vs Amplituda [dB]')
    ax1.set_xlabel('Dystans [m]')
    ax1.set_ylabel('Amplituda [dB]')
    ax1.grid(True)
    ax2.plot(freq/1e6, 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum))))
    ax2.set_title('Częstotliwość vs Amplituda [dB]')
    ax2.set_xlabel('Częstotliwość [MHz]')
    ax2.set_ylabel('Amplituda [dB]')
    ax2.grid(True)
    ax3.plot(freq/1e6, np.angle(spectrum, deg=True))
    ax3.set_title('Częstotliwość vs Faza [°]')
    ax3.set_xlabel('Częstotliwość [MHz]')
    ax3.set_ylabel('Faza [stopnie]')
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================================================
# ---- GŁÓWNY PROGRAM (ZMODYFIKOWANY O ODEJMOWANIE TŁA) ----
# ==========================================================

if __name__ == "__main__":
    print("="*60)
    print("SAR BACKPROJECTION - Offline Processing (z Odejmowaniem Tła)")
    print("="*60)
    
    # Krok 1: Wczytaj oba pomiary
    measurements_scene = load_measurements(DATA_FILE_SCENE)
    measurements_background = load_measurements(DATA_FILE_BACKGROUND)

    # Krok 2: Przetwórz oba pomiary (zastosuj filtr HPF itp.)
    processed_scene = process_measurements(measurements_scene)
    processed_background = process_measurements(measurements_background)

    # Krok 3: Sprawdź, czy pomiary są zgodne
    if len(processed_scene['data_fft']) != len(processed_background['data_fft']):
        print("\nBŁĄD: Pomiary sceny i tła mają różną liczbę pozycji!")
        print(f"Scena: {len(processed_scene['data_fft'])}, Tło: {len(processed_background['data_fft'])}")
        sys.exit(1)

    print("\nWykonuję koherentne odejmowanie tła...")
    
    # Krok 4: Odejmij koherentnie (zespolone) dane FFT
    subtracted_fft_data = []
    for i in range(len(processed_scene['data_fft'])):
        s_scene = processed_scene['data_fft'][i]
        s_bg = processed_background['data_fft'][i]
        subtracted_fft_data.append(s_scene - s_bg)

    # Krok 5: Przygotuj końcowy zestaw danych do backprojection
    processed_data_final = {
        'data_fft': subtracted_fft_data,
        'positions': processed_scene['positions'] # Pozycje są te same
    }
    print("Odejmowanie zakończone.")

    # Krok 6: Uruchom backprojection na *odjętych* danych
    image_db, azimuth_axis, range_axis = backprojection(
        processed_data_final, 
        azimuth_length_m, 
        range_length_m, 
        resolution_azimuth_m, 
        resolution_range_m
    )

    # Krok 7: Wyświetl wynik
    image_plot(image_db, azimuth_axis, range_axis, calibration_factor, vmin_val, vmax_val)
    
    print("\n" + "="*60)
    print("Przetwarzanie offline zakończone.")
    print("="*60)