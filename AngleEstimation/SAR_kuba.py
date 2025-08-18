import time
import numpy as np
import socket
import matplotlib.pyplot as plt
from adi import ad9361
from adi.cn0566 import CN0566

# --- Configuration Parameters ---
ESP32_IP = "192.168.0.103"
ESP32_PORT = 3333
NUM_STEPS = 100               # liczba kroków (pozycji)
STEP_SIZE_M = 0.0005          # krok w metrach
FFT_SIZE = 4096               # rozmiar FFT (dopasować do pliku Waterfall)
SAMPLE_RATE = 0.6e6           # Hz
CENTER_FREQ = 2.1e9           # Hz (podobnie jak w Waterfall)
CHIRP_BW = 500e6              # Hz
RAMP_TIME_S = 0.0005          # s
C = 3e8                       # prędkość światła

# --- Helper Functions ---
def connect_esp32():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ESP32_IP, ESP32_PORT))
    sock.settimeout(5)
    print("[INFO] Połączono z ESP32")
    return sock

def step_forward(sock, steps=1):
    for _ in range(steps):
        sock.sendall(b"STEP_CW\n")
    data = sock.recv(1024)
    if b"DONE_CW" in data:
        print("[INFO] ESP32 wykonał krok")
    else:
        print("[WARN] Nieoczekiwana odpowiedź ESP32:", data)

# Acquire a single range profile (waterfall slice)
def acquire_range_profile(sdr):
    samples = sdr.rx()
    # Zsumuj oba kanały
    iq = samples[0] + samples[1]
    # Windowing i FFT
    window = np.blackman(len(iq))
    spec = np.fft.fftshift(np.fft.fft(iq * window, n=FFT_SIZE))
    mag = np.abs(spec)
    # Przeskaluj na odległość
    slope = CHIRP_BW / RAMP_TIME_S
    freq = np.linspace(-SAMPLE_RATE/2, SAMPLE_RATE/2, FFT_SIZE)
    dist = (freq) * C / (2 * slope)
    return dist, mag

# Simple backprojection SAR
# positions: array [M] z-coordinates w metrach
# profiles: list of arrays [M x N_range]
def backprojection_image(positions, dist, profiles, x_image, z_image):
    # x_image: cross-range positions (w metrach)
    # z_image: range positions (w metrach)
    img = np.zeros((len(z_image), len(x_image)), dtype=np.complex128)
    for m, x_pos in enumerate(positions):
        for ix, x in enumerate(x_image):
            for iz, z in enumerate(z_image):
                r = np.sqrt((x - x_pos)**2 + z**2)
                # find nearest dist index
                idx = np.argmin(np.abs(dist - r))
                img[iz, ix] += profiles[m][idx]
    return np.abs(img)

# --- Main Routine ---
def main():
    # Połączenie do radaru CN0566
    try:
        phaser = CN0566(uri="ip:phaser.local")
        sdr = ad9361(uri="ip:phaser.local:50901")
    except:
        phaser = CN0566(uri="ip:localhost")
        sdr = ad9361(uri="ip:192.168.2.1")
    phaser.sdr = sdr
    # Konfiguracja SDR
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_buffer_size = FFT_SIZE
    sdr.rx_enabled_channels = [0, 1]
    sdr.rx_lo = int(CENTER_FREQ)
    # Wyłącz TX
    sdr.tx_enabled_channels = []
    time.sleep(0.1)

    # Połączenie do ESP32
    sock = connect_esp32()
    time.sleep(0.5)

    # Tablice danych
    positions = []
    profiles = []

    # Pomiar w każdym kroku
    for k in range(NUM_STEPS):
        print(f"[STEP {k+1}/{NUM_STEPS}] Pozition: {k*STEP_SIZE_M*1000:.2f} mm")
        if k > 0:
            step_forward(sock)
            time.sleep(0.1)
        curr_pos = k * STEP_SIZE_M
        dist, mag = acquire_range_profile(sdr)
        positions.append(curr_pos)
        profiles.append(mag)
        print(f"  Zebrano profil, max amp: {mag.max():.1f}")

    sock.close()
    print("[INFO] Zakończono akwizycję")

    # Definicja siatki obrazu
    x_im = np.linspace(positions[0], positions[-1], len(positions))
    # ograniczamy zakres z do np. od 0.5 do 5 m
    z_im = np.linspace(0.5, 5.0, 512)

    # Rekonstrukcja SAR
    print("[INFO] Rekonstrukcja obrazu SAR...")
    image = backprojection_image(positions, dist, profiles, x_im, z_im)

    # Wyświetlenie wyniku
    plt.figure(figsize=(8,6))
    extent = [x_im[0], x_im[-1], z_im[-1], z_im[0]]
    plt.imshow(20*np.log10(image+1e-6), extent=extent, aspect='auto', cmap='gray')
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Range (m)')
    plt.title('SAR Image')
    plt.show()

if __name__ == '__main__':
    main()
