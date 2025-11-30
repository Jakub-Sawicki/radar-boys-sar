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
from scipy.signal import hilbert, detrend, butter, filtfilt, freqz
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
import sys

# ==========================================================
# ---- KONFIGURACJA ----
# ==========================================================

BW = 500e6           # Bandwidth [Hz]
ramp_time_s = 0.5e-3 # Ramp time [s] (0.5e3 us - czas JEDNEGO ZBOCZA)
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

DATA_FILE = "SAR_offline/measurements/330_3_80m_lewy_v1.npz"
# DATA_FILE = "SAR_offline/measurements/330_bez_obiektu.npz"

azimuth_length_m=2.5
range_length_m=10
resolution_azimuth_m=0.3
resolution_range_m=0.40

calibration_factor=3.8/5.5
vmin_val=None
vmax_val=None
vmin_val=85
# vmax_val=100

# Dla 3.8m: azimuth_length_m=2.4, range_length_m=11, resolution_azimuth_m=0.15, resolution_range_m=0.40
# Dla 7m: 

range_axis_start = 1

# ==========================================================
# ---- FUNKCJE ----
# ==========================================================

def load_measurements(file_path):
    """Wczytaj zapisane dane pomiarowe z pliku .npz"""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"\nWczytano plik: {file_path}")
        print(f"Zawiera {len(data['data_fft'])} pomiarów")
    except FileNotFoundError:
        print(f"\nBŁĄD: Nie znaleziono pliku: {file_path}")
        sys.exit(1)

    return data


# --- NOWA FUNKCJA DO KONWERSJI TRÓJKĄTA NA PIŁOKSZTAŁTNY ---
def extract_up_ramps(data_raw, fs, ramp_time_s):
    """
    Wyodrębnia tylko segmenty UP-chirp (sawtooth) z danych trójkątnych.
    Całkowity okres to 2 * ramp_time_s.
    """
    # 300 próbek
    samples_per_ramp_up = int(fs * ramp_time_s) 
    # 600 próbek
    samples_per_full_cycle = samples_per_ramp_up * 2 
    N_total = len(data_raw) 
    
    # Liczba pełnych okresów trójkątnych (Up + Down)
    N_full_cycles = N_total // samples_per_full_cycle
    
    up_ramps = []
    
    for i in range(N_full_cycles):
        # Ekstrakcja tylko pierwszych 300 próbek (UP-chirp) z każdego 600-próbkowego cyklu
        start_idx = i * samples_per_full_cycle
        end_idx = start_idx + samples_per_ramp_up
        up_ramps.append(data_raw[start_idx:end_idx])
        
    # Sprawdzamy, czy w pozostałej części jest jeszcze jeden pełny UP-chirp (300 próbek)
    remainder = N_total - (N_full_cycles * samples_per_full_cycle)
    if remainder >= samples_per_ramp_up:
        start_idx = N_full_cycles * samples_per_full_cycle
        end_idx = start_idx + samples_per_ramp_up
        up_ramps.append(data_raw[start_idx:end_idx])

    if not up_ramps:
        # Fallback - powinien się zdarzyć tylko, jeśli dane są za krótkie
        print("[WARNING] Nie udało się wydzielić segmentów UP-chirp. Zwracam oryginalne dane.")
        return data_raw

    # Nowy wektor danych zawiera tylko segmenty UP-chirp
    return np.concatenate(up_ramps)
# -----------------------------------------------------------


# --- NOWA FUNKCJA DO DOKUMENTACJI PROCESU PRZETWARZANIA SYGNAŁU ---
def plot_signal_processing_steps(raw_data, y_filtered, win_funct, sp_filtered, sp_windowed, fs, b, a):
    """Generuje wykresy dokumentujące etapy przetwarzania sygnału dla pierwszego pomiaru."""
    
    N = len(raw_data)
    time_vec = np.arange(N) / fs
    time_vec = np.arange(N) / fs

    N_fft = len(sp_filtered) 
    freq_vec = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/fs))

    # fig = plt.figure(figsize=(15, 12))
    # fig.suptitle('Dokumentacja Przetwarzania Sygnału (Pierwszy Pomiar)', fontsize=16)

    # # --- 1. Czasowa: Sygnał surowy vs po filtracji ---
    # # NOTE: Ta oś czasu jest przeskalowana, ponieważ len(raw_data) to teraz 2100 zamiast 4096
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.plot(time_vec * 1000, np.abs(raw_data), label='Sygnał po konwersji na piłokształtny (Amplituda)')
    # ax1.plot(time_vec * 1000, np.abs(y_filtered), label='Po filtracji Butterwortha')
    # ax1.set_title('1. Sygnał w dziedzinie czasu (Po konwersji na Sawtooth vs Filtrowany)')
    # ax1.set_xlabel('Czas [ms]')
    # ax1.set_ylabel('Amplituda [j.a.]')
    # ax1.legend()
    # ax1.grid(True)

    # # --- 2. Charakterystyka filtru Butterwortha ---
    # # Obliczanie charakterystyki filtru
    # w, h = freqz(b, a, worN=2000)
    # w_hz = w * fs / (2 * np.pi)
    
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax2.plot(w_hz / 1e3, 20 * np.log10(np.abs(h)))
    # # Częstotliwość odcięcia 80 kHz
    # ax2.axvline(x=80, color='r', linestyle='--', label='Projektowana f_odcięcia (80 kHz)')
    # # Normalizowana częstotliwość w kodzie to 0.4
    # ax2.axvline(x=0.4 * fs/2 / 1e3, color='g', linestyle='-.', label='Użyta f_norm (0.4 * fs/2)') 
    
    # ax2.set_title('2. Charakterystyka Amplitudowa Filtru Butterwortha (Rząd 4)')
    # ax2.set_xlabel('Częstotliwość [kHz]')
    # ax2.set_ylabel('Amplituda [dB]')
    # ax2.legend()
    # ax2.grid(True, which='both')
    # ax2.set_xlim(0, fs/2/1e3)

    # # --- 3. Czasowa: Okno Tukey ---
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.plot(time_vec * 1000, np.abs(y_filtered), label='Sygnał Filtrowany (Amplituda)')
    # ax3_twin = ax3.twinx()
    # ax3_twin.plot(time_vec * 1000, win_funct, 'r--', alpha=0.6, label='Okno Tukey (alpha=0.3)')
    # ax3.set_title('3. Aplikacja Okna Tukey w dziedzinie czasu')
    # ax3.set_xlabel('Czas [ms]')
    # ax3.set_ylabel('Amplituda sygnału [j.a.]')
    # ax3_twin.set_ylabel('Wartość Okna')
    # ax3.legend(loc='upper left')
    # ax3_twin.legend(loc='upper right')
    # ax3.grid(True)

    # # --- 4. Widmo: Przed oknem vs Po oknie ---
    # # Normalizacja dla czytelności
    # sp_filt_db = 20 * np.log10(np.abs(sp_filtered) / np.max(np.abs(sp_filtered)))
    # sp_win_db = 20 * np.log10(np.abs(sp_windowed) / np.max(np.abs(sp_windowed)))

    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.plot(freq_vec / 1e6, sp_filt_db, label='FFT po filtracji (przed oknem)')
    # ax4.plot(freq_vec / 1e6, sp_win_db, label='FFT po oknie Tukey')
    # ax4.set_title('4. Widmo: Efekt Okna Tukey (znormalizowane)')
    # ax4.set_xlabel('Częstotliwość [MHz]')
    # ax4.set_ylabel('Amplituda [dB]')
    # ax4.set_xlim(-0.3, 0.3)
    # ax4.legend()
    # ax4.grid(True)

    # plt.tight_layout()
    # plt.show()
    
# ------------------------------------------------------------------


def process_measurements(measurements_data):
    """Przetwarza wszystkie pomiary"""
    print(f"\nPrzetwarzanie {len(measurements_data['data_fft'])} pomiarów z filtrowaniem...")
    
    processed_data = {
        'data_fft': [],
        'positions': measurements_data['positions']
    }
    
    # Definicja filtra (przeniesiona poza pętlę)
    fs_nyquist = fs / 2
    b, a = butter(4, 0.4, btype='high') 

    for i, data_raw in enumerate(measurements_data['data_fft']):
        
        # --- KROK 1: EKSTRAKCJA RAMP UP (Konwersja Triangular -> Sawtooth) ---
        data = extract_up_ramps(data_raw, fs, ramp_time_s)
        
        # --- KROK 2: ZERO-PADDING do pierwotnego rozmiaru FFT (4096) ---
        # Wymagane, aby zachować rozdzielczość i kompatybilność z backprojection
        if len(data) < N_frame:
            data_padded = np.pad(data, (0, N_frame - len(data)), 'constant')
        else:
            data_padded = data[:N_frame] # Przycinanie w razie pomyłki

        # 1. Filtracja (na dopełnionych danych)
        y_filtered = filtfilt(b, a, data_padded)
        
        # 2. Okienkowanie
        win_funct = tukey(len(y_filtered), alpha=0.3)
        
        # --- ZBIERANIE DANYCH DO DOKUMENTACJI (Tylko dla i=0) ---
        if i == 0:
            # FFT filrowanego sygnału (Przed oknem)
            sp_filtered = np.fft.fftshift(np.fft.fft(y_filtered))
            # Używamy danych po ekstrakcji ramp do wykresu 1, aby pokazać, co wchodzi do filtracji
            raw_data_plot = data 
            
        y_windowed = y_filtered * win_funct

        # 3. FFT (po oknie)
        sp_windowed = np.fft.fftshift(np.fft.fft(y_windowed))
        
        # --- GENEROWANIE WYKRESÓW DOKUMENTUJĄCYCH ---
        if i == 0:
            # Przekazujemy dane przed paddingiem do wykresu 1, aby uniknąć wykresu z dużą ilością zer
            plot_signal_processing_steps(raw_data_plot, y_filtered[:len(raw_data_plot)], win_funct[:len(raw_data_plot)], sp_filtered, sp_windowed, fs, b, a)


        processed_data['data_fft'].append(sp_windowed)
    
    print(f"Przetwarzanie zakończone\n")
    return processed_data


def backprojection(measurements_data, azimuth_length_m=2.4, range_length_m=11, resolution_azimuth_m=0.15, resolution_range_m=0.40):
    print("Starting backprojection")
    
    # Parametry radaru (potrzebne do mapowania częstotliwość->odległość)
    c = 3e8
    slope = BW / ramp_time_s
    
    # Przygotowanie siatki obrazu
    azimuth_axis = np.arange(-azimuth_length_m/2, azimuth_length_m/2, resolution_azimuth_m)
    range_axis = np.arange(range_axis_start, range_length_m, resolution_range_m)
    
    # Inicjalizacja macierzy obrazu (zespolona - do koherentnego sumowania)
    image = np.zeros((len(azimuth_axis), len(range_axis)), dtype=complex)
    # image = np.zeros((len(range_axis), len(azimuth_axis)), dtype=complex)
    
    antenna_positions = np.array(measurements_data['positions'])
    antenna_positions -= np.mean(antenna_positions)
    
    fft_data = measurements_data['data_fft']
    
    print(f"Image grid: {len(azimuth_axis)} x {len(range_axis)} pixels")
    print(f"Processing {len(antenna_positions)} antenna positions...")
    
    # GŁÓWNA PĘTLA BACKPROJECTION

    for i, azim in enumerate(azimuth_axis):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing azimuth row {i}/{len(azimuth_axis)}")
        
        for j, range_dist in enumerate(range_axis):
            pixel_value = 0 + 0j
            
            for k, ant_pos in enumerate(antenna_positions):
                # Calculating the distance between antenna position and pixel
                distance = np.sqrt((azim - ant_pos)**2 + range_dist**2)

                # Adding phase shift
                phase_shift = np.exp(1j * 4 * np.pi * distance * output_freq / c)
                
                # Mapping distance to frequency, dist = (freq - signal_freq) * c / (2 * slope) => freq = (dist * 2 * slope / c) + signal_freq
                freq_value = (distance * 2 * slope / c) + signal_freq
                # freq_value = (distance * 4 * slope / c) + signal_freq

                # Ustal oś częstotliwości
                freq_axis = np.linspace(-fs/2, fs/2, len(fft_data[k]))

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
    # plt.imshow(image_db.T, **plot_kwargs)
    plt.imshow(np.flip(image_db.T, axis=1), **plot_kwargs)
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Azimuth [m]')
    plt.ylabel('Range [m]')
    plt.title('SAR Image - Backprojection')
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
    
    image_db, azimuth_axis, range_axis = backprojection(processed_data, azimuth_length_m, range_length_m, resolution_azimuth_m, resolution_range_m)

    image_plot(image_db, azimuth_axis, range_axis, calibration_factor, vmin_val, vmax_val)
    
    print("\n" + "="*60)
    print("Przetwarzanie offline zakończone.")
    print("="*60)