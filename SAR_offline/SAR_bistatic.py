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

DATA_FILE = "measurments/330_3_80m_v2.npz"
# DATA_FILE = "measurments/330_bez_obiektu.npz"

azimuth_length_m=2
range_length_m=9
resolution_azimuth_m=0.3
resolution_range_m=0.4

calibration_factor=3.8/5.5
vmin_val=None
vmax_val=None
vmin_val=75
# vmax_val=105

# Dla 3.8m: azimuth_length_m=2.4, range_length_m=11, resolution_azimuth_m=0.15, resolution_range_m=0.40
# Dla 7m: 

range_axis_start = 1

# ==========================================================
# ---- FUNKCJE ----
# ==========================================================

def load_measurements(file_path):
    """Wczytaj zapisane dane pomiarowe z pliku .npz"""
    data = np.load(file_path, allow_pickle=True)
    print(f"\nWczytano plik: {file_path}")
    print(f"Zawiera {len(data['data_fft'])} pomiarów")

    return data

def process_measurements(measurements_data):
    """Przetwarza wszystkie pomiary"""
    print(f"\nPrzetwarzanie {len(measurements_data['data_fft'])} pomiarów z filtrowaniem...")
    
    processed_data = {
        'data_fft': [],
        'positions': measurements_data['positions']
    }
    
    # Definicja filtra
    fs_nyquist = fs / 2
    cutoff_freq = 80e3 
    wn = cutoff_freq / fs_nyquist # Normalizowana częstotliwość (ok. 0.267)
    b, a = butter(4, 0.4, btype='high') 

    for i, data in enumerate(measurements_data['data_fft']):
        # Implementacja filtra
        y = filtfilt(b, a, data)

        # Okienkowanie
        # win_funct = np.blackman(len(data))
        win_funct = tukey(len(y), alpha=0.3)
        y = y * win_funct

        sp = np.fft.fftshift(np.fft.fft(y))
        processed_data['data_fft'].append(sp)
    
    print(f"Przetwarzanie zakończone\n")
    return processed_data


def backprojection(measurements_data, azimuth_length_m=2.4, range_length_m=11, resolution_azimuth_m=0.15, resolution_range_m=0.40):
    print("Starting backprojection")
    
    # Parametry radaru (potrzebne do mapowania częstotliwość->odległość)
    c = 3e8
    slope = BW / ramp_time_s
    
    # Branie pod uwagę tego, że antena nadawcza i odbiorcza nie są w tym samym punkcie
    # Pozycja TX względem RX
    TX_pos = np.array([0.0, -0.2, 0.8])

    # Przygotowanie siatki obrazu
    azimuth_axis = np.arange(-azimuth_length_m/2, azimuth_length_m/2, resolution_azimuth_m)
    range_axis = np.arange(range_axis_start, range_length_m, resolution_range_m)
    
    # Inicjalizacja macierzy obrazu (zespolona - do koherentnego sumowania)
    image = np.zeros((len(azimuth_axis), len(range_axis)), dtype=complex)
    
    antenna_positions = np.array(measurements_data['positions'])
    antenna_positions -= np.mean(antenna_positions)
    
    fft_data = measurements_data['data_fft']
    
    print(f"Image grid: {len(azimuth_axis)} x {len(range_axis)} pixels")
    print(f"Processing {len(antenna_positions)} antenna positions...")

    # Ustal oś częstotliwości
    freq_axis = np.linspace(-fs/2, fs/2, len(fft_data[0]))
    
    # GŁÓWNA PĘTLA BACKPROJECTION

    for i, azim in enumerate(azimuth_axis):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing azimuth row {i}/{len(azimuth_axis)}")
        
        for j, range_dist in enumerate(range_axis):
            pixel_value = 0 + 0j
            pixel_pos = np.array([azim, range_dist, 0.0]) # [X, Y, Z]

            dist_tx = np.linalg.norm(pixel_pos - TX_pos)
            
            for k, ant_pos_x in enumerate(antenna_positions):
                # Calculating the distance between antenna position and pixel
                # distance = np.sqrt((azim - ant_pos)**2 + range_dist**2)
                rx_pos = np.array([ant_pos_x, 0.0, 0.0])
                dist_rx = np.linalg.norm(pixel_pos - rx_pos)

                total_distance = dist_tx + dist_rx

                # Adding phase shift
                # phase_shift = np.exp(1j * 4 * np.pi * distance * output_freq / c)
                phase_shift = np.exp(1j * 2 * np.pi * total_distance * output_freq / c)
                
                # Mapping distance to frequency, dist = (freq - signal_freq) * c / (2 * slope) => freq = (dist * 2 * slope / c) + signal_freq
                freq_value = (total_distance * slope / c) + signal_freq

                # Interpolacja liniowa wartości zespolonych
                if freq_value >= freq_axis[0] and freq_value <= freq_axis[-1]:
                    real_interp = np.interp(freq_value, freq_axis, np.real(fft_data[k]))
                    imag_interp = np.interp(freq_value, freq_axis, np.imag(fft_data[k]))
                    sample = real_interp + 1j * imag_interp
                    pixel_value += sample * phase_shift
            
            image[i, j] = pixel_value
    
    # Konwersja do skali dB do wyświetlenia
    image_db = 20 * np.log10(np.abs(image) + 1e-15)  # +1e-15 aby uniknąć log(0)

    return image_db, azimuth_axis, range_axis

def image_plot(image_db, azimuth_axis, range_axis, calibration_factor, vmin_val, vmax_val):

    # Kalibracja odległości
    scaled_range_axis = range_axis * calibration_factor

    plot_kwargs = {
        'extent': [azimuth_axis[0], azimuth_axis[-1], 
                   scaled_range_axis[0], scaled_range_axis[-1]],
        'aspect': 'auto',
        'cmap': 'plasma', # jet, viridis, YlOrRd, Reds, plasma
        'origin': 'lower'
    }

    # Normalizacja amplitudy (skali kolorów) jeśli zostały podane wartości maksymalne i minimalne
    if vmin_val is not None:
        plot_kwargs['vmin'] = vmin_val
    if vmax_val is not None:
        plot_kwargs['vmax'] = vmax_val

    # Wyświetlenie wyniku
    plt.figure(figsize=(10, 8))
    plt.imshow(image_db.T, **plot_kwargs)
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Azimuth [m]')
    plt.ylabel('Range [m]')
    plt.title('SAR Image - Backprojection')
    plt.show()

def data_plot(processed_data):
    global freq, dist

    num_positions = len(processed_data['positions'])
    N = len(processed_data['data_fft'][0])
    spectrum = processed_data['data_fft'][0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Amplituda (moduł FFT)
    ax1.plot(dist, 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum))))
    ax1.set_title('Dystans vs Amplituda [dB]')
    ax1.set_xlabel('Dystans [m]')
    ax1.set_ylabel('Amplituda [dB]')
    ax1.grid(True)

    # Widmo amplitudowe
    ax2.plot(freq/1e6, 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum))))
    ax2.set_title('Częstotliwość vs Amplituda [dB]')
    ax2.set_xlabel('Częstotliwość [MHz]')
    ax2.set_ylabel('Amplituda [dB]')
    ax2.grid(True)

    # Wykres fazy
    ax3.plot(freq/1e6, np.angle(spectrum, deg=True))
    ax3.set_title('Częstotliwość vs Faza [°]')
    ax3.set_xlabel('Częstotliwość [MHz]')
    ax3.set_ylabel('Faza [stopnie]')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================================
# ---- GŁÓWNY PROGRAM ----
# ==========================================================

if __name__ == "__main__":
    print("="*60)
    print("SAR BACKPROJECTION - Offline Processing")
    print("="*60)
    
    measurements_data = load_measurements(DATA_FILE)
    
    processed_data = process_measurements(measurements_data)

    # Showing data plots if measurment data is just a single measurment
    if len(measurements_data['positions']) == 1:
        data_plot(processed_data)   
    
    image_db, azimuth_axis, range_axis = backprojection(processed_data, azimuth_length_m, range_length_m, resolution_azimuth_m, resolution_range_m)

    image_plot(image_db, azimuth_axis, range_axis, calibration_factor, vmin_val, vmax_val)
    
    print("\n" + "="*60)
    print("Przetwarzanie offline zakończone.")
    print("="*60)