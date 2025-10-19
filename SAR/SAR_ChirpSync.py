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

# Instantiate all the Devices
rpi_ip = "ip:phaser.local"  # IP address of the Raspberry Pi
sdr_ip = "ip:192.168.2.1"  # "192.168.2.1, or pluto.local"  # IP address of the Transceiver Block
my_sdr = adi.ad9361(uri=sdr_ip)
my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)

# Configure ESP32 connection
ESP32_IP = "192.168.0.103"
ESP32_PORT = 3333
MEASUREMENTS = 200     # How many steps/measurments
STEP_SIZE_M = 0.00018  # Step size [m]

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
my_phaser.ramp_mode = "single_sawtooth_burst"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
my_phaser.sing_ful_tri = (
    0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
)
my_phaser.tx_trig_en = 1  # start a ramp with TXdata
my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

""" Synchronize chirps to the start of each Pluto receive buffer
"""
# Configure TDD controller
sdr_pins = adi.one_bit_adc_dac(sdr_ip)
sdr_pins.gpio_tdd_ext_sync = True # If set to True, this enables external capture triggering using the L24N GPIO on the Pluto.  When set to false, an internal trigger pulse will be generated every second
tdd = adi.tddn(sdr_ip)
sdr_pins.gpio_phaser_enable = True
tdd.enable = False         # disable TDD to configure the registers
tdd.sync_external = True
tdd.startup_delay_ms = 0
PRI_ms = ramp_time/1e3 + 1.0
tdd.frame_length_ms = PRI_ms    # each chirp is spaced this far apart
num_chirps = 1
tdd.burst_count = num_chirps       # number of chirps in one continuous receive buffer

tdd.channel[0].enable = True
tdd.channel[0].polarity = False
tdd.channel[0].on_raw = 0
tdd.channel[0].off_raw = 10
tdd.channel[1].enable = True
tdd.channel[1].polarity = False
tdd.channel[1].on_raw = 0
tdd.channel[1].off_raw = 10
tdd.channel[2].enable = True
tdd.channel[2].polarity = False
tdd.channel[2].on_raw = 0
tdd.channel[2].off_raw = 10
tdd.enable = True

# From start of each ramp, how many "good" points do we want?
# For best freq linearity, stay away from the start of the ramps
ramp_time = int(my_phaser.freq_dev_time)
ramp_time_s = ramp_time / 1e6
begin_offset_time = 0.10 * ramp_time_s   # time in seconds
print("actual freq dev time = ", ramp_time)
good_ramp_samples = int((ramp_time_s-begin_offset_time) * sample_rate)
start_offset_time = tdd.channel[0].on_ms/1e3 + begin_offset_time
start_offset_samples = int(start_offset_time * sample_rate)

# size the fft for the number of ramp data points
power=8
fft_size = int(2**power)
num_samples_frame = int(tdd.frame_length_ms/1000*sample_rate)
while num_samples_frame > fft_size:     
    power=power+1
    fft_size = int(2**power) 
    if power==18:
        break
print("fft_size =", fft_size)

# Pluto receive buffer size needs to be greater than total time for all chirps
total_time = tdd.frame_length_ms * num_chirps   # time in ms
print("Total Time for all Chirps:  ", total_time, "ms")
buffer_time = 0
power=12
while total_time > buffer_time:     
    power=power+1
    buffer_size = int(2**power) 
    buffer_time = buffer_size/my_sdr.sample_rate*1000   # buffer time in ms
    if power==23:
        break     # max pluto buffer size is 2**23, but for tdd burst mode, set to 2**22
print("buffer_size:", buffer_size)
my_sdr.rx_buffer_size = buffer_size
print("buffer_time:", buffer_time, " ms")


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

    # Toggling the burst pin from the raspberry pi to start measurments 
    my_phaser._gpios.gpio_burst = 0
    my_phaser._gpios.gpio_burst = 1
    my_phaser._gpios.gpio_burst = 0

    data = my_sdr.rx()
    sum_data = data[0] + data[1]

    # select just the linear portion of the last chirp
    rx_bursts = np.zeros((num_chirps, good_ramp_samples), dtype=complex)
    for burst in range(num_chirps):
        start_index = start_offset_samples + burst*num_samples_frame
        stop_index = start_index + good_ramp_samples
        rx_bursts[burst] = sum_data[start_index:stop_index]
        burst_data = np.ones(fft_size, dtype=complex)*1e-10
        #win_funct = np.blackman(len(rx_bursts[burst]))
        win_funct = np.ones(len(rx_bursts[burst]))
        burst_data[start_offset_samples:(start_offset_samples+good_ramp_samples)] = rx_bursts[burst]*win_funct

    # win_funct = np.blackman(len(data))
    # y = data * win_funct
    sp = np.absolute(np.fft.fft(burst_data))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct) # fft magnitude without phase information
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))

    return sp

def backprojection(measurements_data, azimuth_length_m=3, range_length_m=6, resolution_m=0.1):
    print("Starting backprojection")
    
    # Parametry radaru (potrzebne do mapowania częstotliwość->odległość)
    c = 3e8
    slope = BW / ramp_time_s
    
    # Przygotowanie siatki obrazu
    azimuth_axis = np.arange(-azimuth_length_m/2, azimuth_length_m/2, resolution_m)
    range_axis = np.arange(0.1, range_length_m, resolution_m)
    
    # Inicjalizacja macierzy obrazu (zespolona - do koherentnego sumowania)
    image = np.zeros((len(azimuth_axis), len(range_axis)), dtype=complex)
    
    # Pozycje anteny (zakładamy, że zaczynamy od pozycji 0 i poruszamy się liniowo)
    antenna_positions = np.array(measurements_data['positions'])
    
    # Dane FFT z pomiarów
    fft_data = measurements_data['data_fft']
    
    print(f"Image grid: {len(azimuth_axis)} x {len(range_axis)} pixels")
    print(f"Processing {len(antenna_positions)} antenna positions...")
    
    # GŁÓWNA PĘTLA BACKPROJECTION
    # Dla każdego piksela na obrazie...
    for i, azim in enumerate(azimuth_axis):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing azimuth row {i}/{len(azimuth_axis)}")
        
        for j, range_dist in enumerate(range_axis):
            pixel_value = 0 + 0j
            
            # Dla każdej pozycji anteny...
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
    
    # Konwersja do skali dB do wyświetlenia
    image_db = 20 * np.log10(np.abs(image) + 1e-15)  # +1e-15 aby uniknąć log(0)
    
    # Wyświetlenie wyniku
    plt.figure(figsize=(10, 8))
    plt.imshow(image_db, 
               extent=[range_axis[0], range_axis[-1], azimuth_axis[-1], azimuth_axis[0]], 
               aspect='auto', cmap='jet')
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Range [m]')
    plt.ylabel('Azimuth [m]')
    plt.title('SAR Image - Backprojection')
    plt.show()
    
    return image, image_db

if __name__ == "__main__":
    main()