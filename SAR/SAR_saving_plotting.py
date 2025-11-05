#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

DATA_FILE = "saved_measurments/raw_data/330_3_80m_lewy_v1.npz"
BW = 500e6
ramp_time_s = 0.5e-3
signal_freq = 100e3
fs = 0.6e6
c = 3e8
STEP_SIZE_M = 0.0009844
f_c = 12.145e9  # częstotliwość nośna (GHz)

def load_measurements(file_path):
    data = np.load(file_path, allow_pickle=True)
    print(f"\n✅ Wczytano plik: {file_path}")
    print(f"Zawiera {len(data['data_fft'])} pomiarów\n")
    return data

def preprocess_fft(measurements_data):
    """Konwertuje surowe dane I/Q na FFT (zespolone, z fazą)"""
    fft_data = []
    slope = BW / ramp_time_s
    N = len(measurements_data["data_fft"][0])

    for i, raw in enumerate(measurements_data["data_fft"]):
        raw = np.array(raw)  # zespolone dane I/Q
        win = windows.blackman(len(raw))
        raw_win = raw * win
        sp = np.fft.fftshift(np.fft.fft(raw_win))
        sp /= np.sum(win)  # normalizacja amplitudy
        fft_data.append(sp)

    fft_data = np.array(fft_data, dtype=complex)
    print(f"✅ Wykonano FFT dla {len(fft_data)} pomiarów")
    return fft_data

# ==============================================================
def phase_correction(fft_data):
    """
    Korekcja fazowa między pomiarami — wyrównanie względem pierwszego pomiaru.
    """
    ref = fft_data[0]
    corrected = [ref]

    for k in range(1, len(fft_data)):
        cross_corr = np.vdot(ref, fft_data[k])  # iloczyn hermitowski
        phase_shift = np.angle(cross_corr)
        fft_data[k] *= np.exp(-1j * phase_shift)
        corrected.append(fft_data[k])

    print("✅ Zastosowano korekcję fazową między pomiarami")
    return np.array(corrected, dtype=complex)

# ==============================================================
def motion_compensation(fft_data, antenna_positions):
    """
    Kompensacja błędów ruchu na podstawie korelacji fazy
    """
    print("✅ Kompensacja błędów ruchu...")
    corrected = [fft_data[0]]
    
    for k in range(1, len(fft_data)):
        # Korelacja z poprzednim pomiarem
        cross_corr = np.correlate(np.abs(fft_data[k]), np.abs(fft_data[k-1]), mode='same')
        peak_idx = np.argmax(cross_corr)
        actual_shift_samples = peak_idx - len(cross_corr)//2
        
        # Korekcja pozycji
        if actual_shift_samples != 0:
            phase_correction_val = np.exp(-1j * 2 * np.pi * actual_shift_samples / len(fft_data[k]))
            corrected.append(fft_data[k] * phase_correction_val)
        else:
            corrected.append(fft_data[k])
    
    print("✅ Motion compensation zakończona")
    return np.array(corrected, dtype=complex)

# ==============================================================
def backprojection(measurements_data, fft_data, BW, ramp_time_s, signal_freq, fs,
                   azimuth_length_m=1.5, range_length_m=7,
                   resolution_azimuth_m=0.1, resolution_range_m=0.15):
    print("🔹 Uruchamianie backprojection...")

    slope = BW / ramp_time_s
    azimuth_axis = np.arange(-azimuth_length_m/2, azimuth_length_m/2, resolution_azimuth_m)
    range_axis = np.arange(0.25, range_length_m, resolution_range_m)
    image = np.zeros((len(azimuth_axis), len(range_axis)), dtype=complex)

    antenna_positions = np.array(measurements_data['positions'])

    freq = np.linspace(-fs / 2, fs / 2, len(fft_data[0]))

    print(f"🧮 Siatka obrazu: {len(azimuth_axis)} x {len(range_axis)} pikseli")
    print(f"📡 Pozycje anten: {len(antenna_positions)}\n")

    # Główna pętla backprojection
    for i, azim in enumerate(azimuth_axis):
        if i % 10 == 0:
            print(f" → Wiersz {i}/{len(azimuth_axis)}")

        for j, range_dist in enumerate(range_axis):
            pixel_value = 0 + 0j
            for k, ant_pos in enumerate(antenna_positions):
                distance = np.sqrt((azim - ant_pos)**2 + range_dist**2)

                # POPRAWKA: użyj 4*slope zamiast 2*slope (dwukierunkowy tor sygnału)
                freq_value = (distance * 4 * slope / c) + signal_freq
                freq_index = int((freq_value + fs/2) / fs * len(freq))

                if 0 <= freq_index < len(freq):
                    # Faza koherentna z korekcją odległości
                    phase = np.exp(1j * 4 * np.pi * f_c * distance / c)
                    pixel_value += fft_data[k][freq_index] * phase
            image[i, j] = pixel_value

    image_db = 20 * np.log10(np.abs(image) + 1e-15)

    plt.figure(figsize=(10, 8))
    plt.imshow(image_db.T,
               extent=[azimuth_axis[0], azimuth_axis[-1], range_axis[0], range_axis[-1]],
               aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Azimuth [m]')
    plt.ylabel('Range [m]')
    plt.title('SAR Image - Backprojection (Motion Compensated)')
    plt.tight_layout()
    plt.show()

    return image, image_db

# ==============================================================
def plot_fft_analysis(fft_data, BW, ramp_time_s, signal_freq, fs, range_length_m=8):
    first_fft = fft_data[0]
    freq_axis = np.linspace(-fs/2, fs/2, len(first_fft))
    slope = BW / ramp_time_s
    
    # POPRAWKA: użyj 4*slope
    dist_axis = (freq_axis - signal_freq) * c / (4 * slope)
    
    mask = (dist_axis >= 0) & (dist_axis <= range_length_m)
    dist_plot = dist_axis[mask]
    magnitude = np.abs(first_fft[mask])
    phase = np.angle(first_fft[mask])
    power_db = 20 * np.log10(magnitude + 1e-15)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(dist_plot, power_db)
    ax1.set_xlabel("Range [m]")
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_title("Amplituda - pierwszy pomiar")
    ax1.grid(True, alpha=0.3)
    ax2.plot(dist_plot, phase)
    ax2.set_xlabel("Range [m]")
    ax2.set_ylabel("Phase [rad]")
    ax2.set_title("Faza - pierwszy pomiar")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==============================================================
if __name__ == "__main__":
    print("="*60)
    print("SAR BACKPROJECTION - Motion Compensation Only")
    print("="*60)

    measurements_data = load_measurements(DATA_FILE)

    # 1️⃣ FFT z zachowaniem fazy
    fft_data = preprocess_fft(measurements_data)

    # 2️⃣ Korekcja fazowa między pomiarami
    fft_data = phase_correction(fft_data)

    # 3️⃣ Motion compensation
    fft_data = motion_compensation(fft_data, measurements_data['positions'])

    # 4️⃣ Analiza FFT
    # plot_fft_analysis(fft_data, BW, ramp_time_s, signal_freq, fs)

    # 5️⃣ Rekonstrukcja obrazu SAR
    image, image_db = backprojection(
        measurements_data, fft_data, BW, ramp_time_s, signal_freq, fs
    )

    print("\n" + "="*60)
    print("✅ Przetwarzanie offline zakończone.")
    print("="*60)