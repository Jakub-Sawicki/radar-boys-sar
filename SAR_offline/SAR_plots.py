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
# Dodano 'freqz' do analizy charakterystyki filtru
from scipy.signal import hilbert, detrend, butter, filtfilt, freqz, cheby1, cheby2, ellip
from scipy.signal.windows import tukey, hamming, hann, blackman, boxcar
from scipy.ndimage import gaussian_filter1d
import sys

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

DATA_FILE = "SAR_offline/measurements/330_3_80m_v2.npz"
# DATA_FILE = "SAR_offline/measurements/330_7m_lewo_v1.npz"
# DATA_FILE = "SAR_offline/measurements/330_bez_obiektu.npz"

save = False

azimuth_length_m=2.5
range_length_m=10
resolution_azimuth_m=0.3
resolution_range_m=0.40

calibration_factor=1 #3.8/5.5
vmin_val=None
vmax_val=None
vmin_val=70
vmax_val=115

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


def plot_windowing_effect(signal_before, signal_after, win_funct, fs, save=False):
    """
    Porównuje sygnał przed i po zastosowaniu okna (Tukey).
    Wykres 1: Sygnały w czasie + nałożony kształt okna.
              Zera obu osi Y są wyrównane.
    Wykres 2: Widma (FFT).
    """
    N = len(signal_before)
    time_vec = np.arange(N) / fs * 1000 # Czas w ms
    
    # Obliczenie FFT
    fft_size = max(N * 8, 16384) 
    freq_vec = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/fs))
    
    fft_before = np.fft.fftshift(np.fft.fft(signal_before, fft_size))
    fft_before_db = 20 * np.log10(np.abs(fft_before) / np.max(np.abs(fft_before)) + 1e-12)
    
    fft_after = np.fft.fftshift(np.fft.fft(signal_after, fft_size))
    fft_after_db = 20 * np.log10(np.abs(fft_after) / np.max(np.abs(fft_after)) + 1e-12)

    # --- WYKRES 1: DZIEDZINA CZASU ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Oś lewa: Sygnały
    ax1.plot(time_vec, np.abs(signal_before), label='Sygnał przed oknem', color='silver', linewidth=2)
    ax1.plot(time_vec, np.abs(signal_after), label='Sygnał po oknie Tukey', color='C0', linewidth=1.5)
    ax1.set_xlabel('Czas [ms]')
    ax1.set_ylabel('Amplituda sygnału')
    ax1.grid(True, alpha=0.5)
    ax1.set_xlim(time_vec[0], time_vec[-1])
    
    # WYMUSZENIE ZERA NA DOLE DLA OSI LEWEJ
    # Obliczamy max żeby ustawić ładny górny limit, ale dół sztywno na 0
    max_sig = np.max(np.abs(signal_before))
    ax1.set_ylim(bottom=0, top=max_sig * 1.1) 
    
    # Oś prawa: Kształt okna
    ax2 = ax1.twinx()
    ax2.plot(time_vec, win_funct, 'r--', alpha=0.6, linewidth=1.5, label='Kształt okna')
    ax2.set_ylabel('Waga okna', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # WYMUSZENIE ZERA NA DOLE DLA OSI PRAWEJ
    ax2.set_ylim(bottom=0, top=1.1) 
    
    # LEGENDA W PRAWYM GÓRNYM ROGU
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # loc='upper right' umieszcza legendę w rogu
    # framealpha=1 sprawia, że tło legendy jest nieprzezroczyste (żeby linie nie prześwitywały)
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', framealpha=1.0)
    
    # plt.title('1. Efekt w dziedzinie czasu: Tłumienie krawędzi przez okno', fontsize=12)
    plt.tight_layout()
    
    if save:
        plt.savefig('SAR_offline/plots/plot_windowing_time.png')
    else:
        plt.show()
    
    # --- WYKRES 2: DZIEDZINA CZĘSTOTLIWOŚCI ---
    plt.figure(figsize=(10, 6))
    plt.plot(freq_vec / 1e3, fft_before_db, label='Widmo przed oknem', color='silver', linewidth=2, alpha=0.8)
    plt.plot(freq_vec / 1e3, fft_after_db, label='Widmo po oknie', color='C0', linewidth=1.5)
    
    # plt.title('2. Efekt w dziedzinie częstotliwości (Redukcja przecieku widma)', fontsize=12)
    plt.xlabel('Częstotliwość [kHz]')
    plt.ylabel('Amplituda znormalizowana [dB]')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', alpha=0.5)
    
    plt.xlim(0, fs/2/1e3) 
    plt.ylim(-100, 5) 
    plt.tight_layout()
    
    if save:
        plt.savefig('SAR_offline/plots/plot_windowing_fft.png')
    else:
        plt.show()
        

def plot_window_comparison(N, fs, save=False):
    """
    Generuje zestawienie 4 okien: Prostokątne (odniesienie), Hamming, Hanning, Tukey.
    Zoom na centrum (DC) pokazuje kompromis między szerokością listka a tłumieniem.
    """
    # Zastąpiono Blackmana oknem Prostokątnym (boxcar)
    windows_data = [
        ('Prostokątne (Brak okna)', boxcar(N)),   # Najlepsza rozdzielczość, najgorsze listki boczne
        ('Hamming', hamming(N)),                  # Klasyk
        ('Hanning', hann(N)),                     # Szybki spadek listków bocznych
        ('Tukey (alpha=0.8)', tukey(N, alpha=0.8)) # Twój wybór (kompromis)
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    
    # Bardzo duże FFT dla gładkości wykresu
    fft_size = max(N * 32, 65536) 
    freq_vec = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/fs))
    
    for ax, (name, win) in zip(axs, windows_data):
        # 1. FFT z zero-paddingiem
        W = np.fft.fftshift(np.fft.fft(win, fft_size))
        
        # 2. Normalizacja do 0 dB
        W_mag = np.abs(W)
        W_db = 20 * np.log10(W_mag / np.max(W_mag) + 1e-12)
        
        # 3. Rysowanie
        ax.plot(freq_vec / 1e3, W_db, linewidth=2) # Nieco grubsza linia
        
        # 4. Miniaturka czasu
        ax_inset = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
        ax_inset.plot(win, color='orange', linewidth=1.5)
        # ax_inset.set_title("Dziedzina czasu", fontsize=8)
        ax_inset.text(0.5, 0.1, 'Dziedzina czasu', transform=ax_inset.transAxes, 
                      ha='center', va='center', fontsize=8, color='black')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_ylim(-0.1, 1.1) # Margines dla prostokąta
        
        # Stylizacja
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplituda [dB]')
        ax.set_xlabel('Częstotliwość [kHz]')
        ax.grid(True, which='both', alpha=0.4)
        ax.set_ylim(-80, 5)  # Skupiamy się na górnych 80dB
        
        # --- ZOOM ---
        # Ustawiamy sztywny, wąski zakres, np. +/- 3 kHz (jak na Twoim obrazku)
        # Możesz to zmienić na +/- 2 jeśli chcesz jeszcze bliżej
        ax.set_xlim(-3, 3) 

    # fig.suptitle(f'Analiza Okien: Rozdzielczość (szerokość) vs Przeciek (listki boczne)', fontsize=14)
    plt.tight_layout()
    
    if save:
        plt.savefig('SAR_offline/plots/plot_windows_comparison.png')
    else:
        plt.show()

def plot_fft_before_after(raw_data, filtered_data, fs, save=False):
    """
    Porównuje widmo (FFT) sygnału surowego i przefiltrowanego (przed okienkowaniem).
    """
    N = len(raw_data)
    # Oś częstotliwości
    freq_vec = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    
    # Obliczenie FFT
    fft_raw = np.fft.fftshift(np.fft.fft(raw_data))
    fft_filt = np.fft.fftshift(np.fft.fft(filtered_data))
    
    # Konwersja do dB (zabezpieczenie przed log(0))
    # Normalizujemy oba widma względem maksimum sygnału SUROWEGO, 
    # aby zobaczyć rzeczywiste tłumienie wprowadzone przez filtr.
    max_val = np.max(np.abs(fft_raw))
    
    fft_raw_db = 20 * np.log10(np.abs(fft_raw) / max_val + 1e-12)
    fft_filt_db = 20 * np.log10(np.abs(fft_filt) / max_val + 1e-12)
    
    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    
    plt.plot(freq_vec / 1e3, fft_raw_db, label='Sygnał surowy', color='silver', linewidth=1.5, alpha=0.8) #, color='lightgray', linewidth=1.5
    plt.plot(freq_vec / 1e3, fft_filt_db, label='Sygnał po filtracji', color='C0', linewidth=1.5)
    
    # Oznaczenie pasma przepustowego (zmienne low/high z Twojego kodu to ok. 120 i 200 kHz)
    plt.axvline(x=120, color='red', linestyle='--', alpha=0.5, label='f_low (120 kHz)')
    plt.axvline(x=200, color='red', linestyle='--', alpha=0.5, label='f_high (200 kHz)')
    
    # plt.title('Porównanie widma FFT: Przed vs Po filtracji (bez okna)')
    plt.xlabel('Częstotliwość [kHz]')
    plt.ylabel('Amplituda znormalizowana [dB]')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', alpha=0.6)
    plt.xlim(0, fs/2/1e3) # Pokaż tylko dodatnie częstotliwości (0 do Nyquista)
    plt.ylim(-100, 5)     # Zakres dB
    
    plt.tight_layout()
    
    if save:
        plt.savefig('SAR_offline/plots/plot_fft_compare_raw_vs_filt.png')
    else:
        plt.show()

def plot_signal_processing_steps(raw_data, y_filtered, win_funct, sp_filtered, sp_windowed, fs, b, a, b1, a1, b2, a2, b3, a3, save=False):
    """
    Generuje wykresy dokumentujące etapy przetwarzania sygnału.
    Sekcja 2 wyświetla porównanie 4 filtrów (Butterworth, Cheby1, Cheby2, Ellip).
    """
    
    N = len(raw_data)
    time_vec = np.arange(N) / fs
    freq_vec = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

    # --- 1. Czasowa: Sygnał surowy vs po filtracji (Butterworth) ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.plot(time_vec * 1000, np.abs(raw_data), label='Sygnał surowy (Amplituda)')
    ax1.plot(time_vec * 1000, np.abs(y_filtered), label='Po filtracji Butterwortha')
    ax1.set_title('1. Sygnał w dziedzinie czasu (Surowy vs Filtrowany)')
    ax1.set_xlabel('Czas [ms]')
    ax1.set_ylabel('Amplituda [j.a.]')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig('SAR_offline/plots/plot_1_czasowa.png')
    else:
        plt.show()
    
    # --- 2. Charakterystyka 4 filtrów (Butterworth, Cheby1, Cheby2, Ellip) ---
    # Tworzymy siatkę wykresów 2x2
    fig2, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten() # Spłaszczamy tablicę osi, aby łatwo po niej iterować
    
    # Lista konfiguracji filtrów do pętli
    filters_data = [
        (b, a, 'Butterworth'),
        (b1, a1, 'Chebyshev Type I'),
        (b2, a2, 'Chebyshev Type II'),
        (b3, a3, 'Elliptic')
    ]
    
    for ax, (bi, ai, name) in zip(axs, filters_data):
        # Obliczenie odpowiedzi częstotliwościowej dla danego filtru
        w, h = freqz(bi, ai, worN=2000)
        w_hz = w * fs / (2 * np.pi)
        
        # Rysowanie charakterystyki
        ax.plot(w_hz / 1e3, 20 * np.log10(np.abs(h)), label=f'Charakterystyka {name}')
        
        # Linie pomocnicze (identyczne jak w Twoim wzorcu)
        ax.axvline(x=120, color='r', linestyle='--', alpha=0.7, label='f_low (120 kHz)')
        ax.axvline(x=200, color='r', linestyle='--', alpha=0.7, label='f_high (200 kHz)')
        
        # Stylizacja wzorowana na Twoim kodzie
        ax.set_title(f'Charakterystyka: {name}')
        ax.set_xlabel('Częstotliwość [kHz]')
        ax.set_ylabel('Amplituda [dB]')
        ax.grid(True, which='both')
        
        # Ustawienie limitów osi (z Twojego wzorca)
        ax.set_xlim(0, fs/2/1e3)
        ax.set_ylim(-100, 5)
        
        # Legenda (opcjonalnie zmniejszona czcionka, żeby nie zasłaniała)
        ax.legend(loc='lower right', fontsize='small')

    # fig2.suptitle('2. Porównanie Charakterystyk Amplitudowych Filtrów', fontsize=14)
    plt.tight_layout()
    
    if save:
        plt.savefig('SAR_offline/plots/plot_2_porownanie_filtrow.png')
    else:
        plt.show()
    
    # --- 3. Czasowa: Okno Tukey ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    ax3.plot(time_vec * 1000, np.abs(y_filtered), label='Sygnał Filtrowany (Amplituda)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_vec * 1000, win_funct, 'r--', alpha=0.6, label='Okno Tukey')
    ax3.set_title('3. Aplikacja Okna Tukey w dziedzinie czasu')
    ax3.set_xlabel('Czas [ms]')
    ax3.set_ylabel('Amplituda sygnału [j.a.]')
    ax3_twin.set_ylabel('Wartość Okna')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig('SAR_offline/plots/plot_3_okno.png')
    else:
        plt.show()
    
    # --- 4. Widmo: Przed oknem vs Po oknie ---
    # Zabezpieczenie przed log(0) przy normalizacji
    sp_filt_abs = np.abs(sp_filtered)
    sp_win_abs = np.abs(sp_windowed)
    
    sp_filt_db = 20 * np.log10(sp_filt_abs / np.max(sp_filt_abs) + 1e-12)
    sp_win_db = 20 * np.log10(sp_win_abs / np.max(sp_win_abs) + 1e-12)

    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    ax4.plot(freq_vec / 1e6, sp_filt_db, label='FFT po filtracji (przed oknem)')
    ax4.plot(freq_vec / 1e6, sp_win_db, label='FFT po oknie Tukey')
    ax4.set_title('4. Widmo: Efekt Okna Tukey (znormalizowane)')
    ax4.set_xlabel('Częstotliwość [MHz]')
    ax4.set_ylabel('Amplituda [dB]')
    ax4.set_xlim(-0.3, 0.3) # Dostosuj zakres wg potrzeb
    ax4.legend()
    ax4.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig('SAR_offline/plots/plot_4_widmo.png')
    else:
        plt.show()


def process_measurements(measurements_data):
    """Przetwarza wszystkie pomiary"""
    print(f"\nPrzetwarzanie {len(measurements_data['data_fft'])} pomiarów z filtrowaniem...")
    
    processed_data = {
        'data_fft': [],
        'positions': measurements_data['positions']
    }
    
    # Definicja filtra (przeniesiona poza pętlę)
    fs_nyquist = fs / 2 # 300 kHz
    low = 0.40 # 105 kHz, około metr
    high = 0.666 # 200 kHz, 15 metrow
    # b, a = butter(4, 0.35, btype='high') 
    b,a = butter(6, [low, high], 'bandpass')
    b1, a1 = cheby1(6, 1, [low, high], 'bandpass')
    b2, a2 = cheby2(6, 40, [low, high], 'bandpass')
    b3, a3 = ellip(6, 1, 40, [low, high], 'bandpass')

    for i, data in enumerate(measurements_data['data_fft']):
        
        # 1. Filtracja
        y_filtered = filtfilt(b, a, data)
        # y_filtered = data
        
        # 2. Okienkowanie
        win_funct = tukey(len(y_filtered), alpha=0.8)
        
        # --- ZBIERANIE DANYCH DO DOKUMENTACJI (Tylko dla i=0) ---
        if i == 0:
            # FFT filrowanego sygnału (Przed oknem)
            sp_filtered = np.fft.fftshift(np.fft.fft(y_filtered))
            raw_data = measurements_data['data_fft'][i] # Sygnał surowy
            
        y_windowed = y_filtered * win_funct
        y_filtered_no_window = filtfilt(b, a, data)

        # 3. FFT (po oknie)
        sp_windowed = np.fft.fftshift(np.fft.fft(y_windowed))
        
        # --- GENEROWANIE WYKRESÓW DOKUMENTUJĄCYCH ---
        if i == 0:
            plot_windowing_effect(y_filtered, y_windowed, win_funct, fs, save=save)
            plot_window_comparison(len(data), fs, save=save)
            plot_fft_before_after(data, y_filtered_no_window, fs, save=save)
            plot_signal_processing_steps(raw_data, y_filtered, win_funct, sp_filtered, sp_windowed, fs, b, a, b1, a1, b2, a2, b3, a3)

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
    if save == True:
        plt.savefig('SAR_offline/plots/SAR_image.png')
    else:
        plt.show()


# Usuwamy starą, ogólną funkcję data_plot, zastępując ją nową, bardziej szczegółową.
# def data_plot(processed_data):
# ...

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