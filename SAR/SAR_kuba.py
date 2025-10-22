#!/usr/bin/env python3
'''SAR imaging with Phaser (CN0566)
   Modified code of Jon Kraft
   
   Jakub Sawicki 2025'''

# Imports
import adi

import sys
import socket
import time
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore, QtGui
from scipy.signal import hilbert

# Instantiate all the Devices
rpi_ip = "ip:phaser.local"  # IP address of the Raspberry Pi
sdr_ip = "ip:192.168.2.1"  # "192.168.2.1, or pluto.local"  # IP address of the Transceiver Block
my_sdr = adi.ad9361(uri=sdr_ip)
my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)

# Configure ESP32 connection
ESP32_IP = "192.168.0.105"
ESP32_PORT = 3333
MEASUREMENTS = 1     # How many steps/measurments
STEP_SIZE_M = 0.0009844  # Step size [m]   31.5 cm in 320 steps

def connect_esp32():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ESP32_IP, ESP32_PORT))
    s.settimeout(5)
    print("Connected to ESP32")
    return s

def send_step_and_wait(sock):
    sock.sendall(b"STEP_CW\n")
    data = sock.recv(1024)
    if b"DONE_CW" in data:
        print("ESP32 completed a step")
    else:
        print("Warning, received:", data)

# Initialize both ADAR1000s, set gains to max, and all phases to 0
my_phaser.configure(device_mode="rx")
my_phaser.load_gain_cal()
my_phaser.load_phase_cal()
for i in range(0, 8):
    my_phaser.set_chan_phase(i, 0)

gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
for i in range(0, len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

# Setup Raspberry Pi GPIO states
try:
    my_phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    my_phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    my_phaser._gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)
except:
    my_phaser.gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    my_phaser.gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    my_phaser.gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)

sample_rate = 0.6e6
center_freq = 2.1e9
signal_freq = 100e3
num_slices = 600     # this sets how much time will be displayed on the waterfall plot
fft_size = 1024 * 4
plot_freq = 100e3    # x-axis freq range to plot
img_array = np.ones((num_slices, fft_size))*(-100)

# Configure SDR Rx
my_sdr.sample_rate = int(sample_rate)
my_sdr.rx_lo = int(center_freq)  # set this to output_freq - (the freq of the HB100)
my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
my_sdr.rx_buffer_size = int(fft_size)
my_sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
my_sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
my_sdr.rx_hardwaregain_chan0 = int(30)  # must be between -3 and 70
my_sdr.rx_hardwaregain_chan1 = int(30)  # must be between -3 and 70
# Configure SDR Tx
my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True  # must set cyclic buffer to true for the tdd burst mode.  Otherwise Tx will turn on and off randomly
my_sdr.tx_hardwaregain_chan0 = -88  # must be between 0 and -88
my_sdr.tx_hardwaregain_chan1 = -0  # must be between 0 and -88

# Configure the ADF4159 Rampling PLL
output_freq = 12.145e9
BW = 500e6
num_steps = 500
ramp_time = 0.5e3  # us
my_phaser.frequency = int(output_freq / 4)  # Output frequency divided by 4
my_phaser.freq_dev_range = int(
    BW / 4
)  # frequency deviation range in Hz.  This is the total freq deviation of the complete freq ramp
my_phaser.freq_dev_step = int(
    (BW/4) / num_steps
)  # frequency deviation step in Hz.  This is fDEV, in Hz.  Can be positive or negative
my_phaser.freq_dev_time = int(
    ramp_time
)  # total time (in us) of the complete frequency ramp
print("requested freq dev time = ", ramp_time)
ramp_time = my_phaser.freq_dev_time
ramp_time_s = ramp_time / 1e6
print("actual freq dev time = ", ramp_time)
my_phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
my_phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
my_phaser.delay_start_en = 0  # delay start
my_phaser.ramp_delay_en = 0  # delay between ramps.
my_phaser.trig_delay_en = 0  # triangle delay
my_phaser.ramp_mode = "continuous_triangular"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
my_phaser.sing_ful_tri = (
    0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
)
my_phaser.tx_trig_en = 0  # start a ramp with TXdata
my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

# Print config
print(
    """
CONFIG:
Sample rate: {sample_rate}MHz
Num samples: 2^{Nlog2}
Bandwidth: {BW}MHz
Ramp time: {ramp_time}ms
Output frequency: {output_freq}MHz
IF: {signal_freq}kHz
""".format(
        sample_rate=sample_rate / 1e6,
        Nlog2=int(np.log2(fft_size)),
        BW=BW / 1e6,
        ramp_time=ramp_time / 1e3,
        output_freq=output_freq / 1e6,
        signal_freq=signal_freq / 1e3,
    )
)

# Create a sinewave waveform
fs = int(my_sdr.sample_rate)
N = int(my_sdr.rx_buffer_size)
fc = int(signal_freq / (fs / N)) * (fs / N)
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 1 * (i + 1j * q)

# Send data
my_sdr._ctx.set_timeout(0)
my_sdr.tx([iq * 0.5, iq])  # only send data to the 2nd channel (that's all we need)

c = 3e8
default_chirp_bw = 500e6
N_frame = fft_size
freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
slope = BW / ramp_time_s
dist = (freq - signal_freq) * c / (2 * slope)

def main():
    # Connecting to ESP32
    sock = connect_esp32()
    time.sleep(1)
    
    measurements_data = {
        'data_fft': [],
        'positions': [],
    }

    print(f"\nStarting {MEASUREMENTS} measurments")

    for i in range(MEASUREMENTS):
        print(f"\n[{i+1}/{MEASUREMENTS}] Measurment...")
        
        if i > 0:
            send_step_and_wait(sock)
            time.sleep(0.3)

        current_position = i * STEP_SIZE_M
        data_fft = measure()

        measurements_data['data_fft'].append(data_fft)
        measurements_data['positions'].append(current_position)

    sock.close()
    print("\nMeasurments completed")

    backprojection(measurements_data)

def measure():
    global freq, dist

    data = my_sdr.rx()
    data = data[0] + data[1]
    win_funct = np.blackman(len(data))
    y = data * win_funct
    sp = np.absolute(np.fft.fft(y))
    sp = np.fft.fftshift(sp)

    # absolute value, not used right now
    s_mag = np.abs(sp) / np.sum(win_funct) # fft magnitude without phase information
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))

    sp = sp / np.sum(win_funct)  # FFT amplitude normalization
    # usunięcie szybkich oscylacji (obwiednia Hilberta)
    envelope = np.abs(hilbert(sp))
    

    if MEASUREMENTS == 1:
        # plots showing the measured data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(dist, envelope)
        ax1.set_title('Dystans vs dBFS')
        ax1.set_xlabel('Dystans')
        ax1.set_ylabel('dBFS')

        ax2.plot(freq, envelope)
        ax2.set_title('Częstotliwość vs dBFS')
        ax2.set_xlabel('Częstotliwość')
        ax2.set_ylabel('dBFS')

        plt.tight_layout()
        plt.show()

    return envelope
    # return sp

def backprojection(measurements_data, azimuth_length_m=10, range_length_m=10, resolution_azimuth_m=0.05, resolution_range_m=0.15):
    print("Starting backprojection")
    
    # Parametry radaru (potrzebne do mapowania częstotliwość->odległość)
    c = 3e8
    slope = BW / ramp_time_s
    
    # Przygotowanie siatki obrazu
    azimuth_axis = np.arange(-azimuth_length_m/2, azimuth_length_m/2, resolution_azimuth_m)
    range_axis = np.arange(1, range_length_m, resolution_range_m)
    
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
                # 1. Oblicz odległość geometryczną między anteną a pikselem
                distance = np.sqrt((azim - ant_pos)**2 + range_dist**2)
                
                # 2. Mapuj odległość na częstotliwość (odwrotność procesu z dist)
                #    Korzystamy z: dist = (freq - signal_freq) * c / (2 * slope)
                #    więc: freq = (dist * 2 * slope / c) + signal_freq
                freq_value = (distance * 2 * slope / c) + signal_freq
                
                # 3. Mapuj częstotliwość na indeks w tablicy FFT
                #    freq jest w zakresie [-fs/2, fs/2] po fftshift
                freq_index = int((freq_value + fs/2) / fs * len(freq))
                
                # 4. Zabezpieczenie przed indeksami poza zakresem
                if 0 <= freq_index < len(fft_data[k]):
                    # 5. Pobierz wartość zespoloną z FFT i dodaj do piksela
                    pixel_value += fft_data[k][freq_index]
            
            # Zapisz skumulowaną wartość dla piksela
            image[i, j] = pixel_value
            # image[j, i] = pixel_value
    
    # Konwersja do skali dB do wyświetlenia
    image_db = 20 * np.log10(np.abs(image) + 1e-15)  # +1e-15 aby uniknąć log(0)
    
    # Wyświetlenie wyniku
    plt.figure(figsize=(10, 8))
    # plt.imshow(image_db.T, aspect='auto', cmap='jet', origin='lower')
    plt.imshow(image_db.T, 
               extent=[azimuth_axis[0], azimuth_axis[-1], range_axis[0], range_axis[-1]], 
               aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Azimuth [m]')
    plt.ylabel('Range [m]')
    plt.title('SAR Image - Backprojection')
    plt.show()

     # ----- DODATKOWY WYKRES: moc vs odległość dla pierwszego pomiaru -----
    try:
        # fft_data[0] w Twoim kodzie to już fftshift(np.abs(FFT))
        first_fft = fft_data[0]
        Nframe = len(first_fft)

        # Utwórz oś częstotliwości (zgodnie z użytym fftshift)
        freq_axis = np.linspace(-fs/2, fs/2, Nframe)  # [Hz]

        # Przekształcenie częstotliwości -> odległość: dist = (freq - IF) * c / (2 * slope)
        # Używamy tych samych parametrów co wcześniej w backprojection
        c = 3e8
        slope = BW / ramp_time_s
        dist_axis = (freq_axis - signal_freq) * c / (2 * slope)  # [m]

        # Filtrujemy tylko sensowny zakres (ujemne dystanse pomijamy)
        mask = (dist_axis >= 0) & (dist_axis <= range_length_m)
        dist_plot = dist_axis[mask]
        power_db = 20 * np.log10(np.abs(first_fft[mask]) + 1e-15)

        plt.figure(figsize=(8,4))
        plt.plot(dist_plot, power_db)
        plt.xlabel("Range [m]")
        plt.ylabel("Magnitude [dB]")
        plt.title("Moc vs odległość — pierwszy pomiar")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Nie udało się narysować wykresu mocy dla pierwszego pomiaru:", e)
    # --------------------------------------------------------------------
    
    return image, image_db

if __name__ == "__main__":
    main()