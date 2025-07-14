import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from adi import ad9361
from adi.cn0566 import CN0566

# -----------------------------
# MLE single snapshot function
# -----------------------------
def analyze_single_snapshot(data_vector):
    Y = np.array(data_vector).reshape(-1, 1)
    M = len(data_vector)
    R = np.outer(Y.flatten(), Y.flatten().conj()) / np.linalg.norm(Y) ** 2

    def a(angleDeg, M):
        angle_rad = np.deg2rad(angleDeg)
        return np.exp(1j * np.pi * np.sin(angle_rad) * np.arange(M)).reshape(-1, 1)

    def CostFunction(angleDeg):
        aTemp = a(angleDeg, M)
        aH_a = np.dot(aTemp.conj().T, aTemp)
        if np.abs(aH_a) < 1e-12:
            return np.inf
        Pv = np.eye(M) - np.dot(aTemp, aTemp.conj().T) / aH_a
        J = np.abs(np.trace(np.dot(Pv, R)))
        return J

    result = minimize_scalar(CostFunction, bounds=(-90, 90), method="bounded")
    return result.x, CostFunction

def plot_angle_cost_function(CostFunction, estimated_angle):
    angleVec = np.linspace(-90, 90, 500)
    costVals = [CostFunction(a) for a in angleVec]

    # plt.figure(figsize=(10, 5))
    # plt.plot(angleVec, costVals, label='Cost Function')
    # plt.axvline(estimated_angle, color='r', linestyle='--', label=f'Est: {estimated_angle:.2f}°')
    # plt.xlabel('Angle (deg)')
    # plt.ylabel('Cost')
    # plt.title('MLE Cost Function vs Angle')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

def plot_complex_plane(data_vector):
    real_parts = np.real(data_vector)
    imag_parts = np.imag(data_vector)

    # plt.figure(figsize=(6, 6))
    # plt.scatter(real_parts, imag_parts, c='b', s=80)
    # for idx, (r, im) in enumerate(zip(real_parts, imag_parts)):
    #     plt.text(r, im, f'{idx}', fontsize=12)
    # plt.xlabel('Real')
    # plt.ylabel('Imag')
    # plt.title('Snapshot Complex Plane')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()

def process_received_data(data):
    """
    Przetwarza dane odebrane z dwóch kanałów.
    Zwraca reprezentatywne wartości zespolone zachowujące fazę.
    """
    # Opcja 1: Użyj pierwszej próbki (dla sygnałów CW)
    channel_data = [data[0][0], data[1][0]]
    
    # Opcja 2: Średnia z zachowaniem fazy (dla sygnałów modulowanych)
    # channel_data = [np.mean(data[0]), np.mean(data[1])]
    
    # Opcja 3: Korelacja ze znanym sygnałem (dla konkretnych modulacji)
    # reference_signal = np.exp(1j * 2 * np.pi * np.arange(len(data[0])))
    # channel_data = [np.sum(data[0] * reference_signal.conj()),
    #                 np.sum(data[1] * reference_signal.conj())]
    
    return channel_data

def calculate_phase_difference(ch0, ch1):
    """Oblicza różnicę faz między kanałami"""
    phase_diff = np.angle(ch1) - np.angle(ch0)
    # Normalizacja do zakresu [-π, π]
    # phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    return np.degrees(phase_diff)

# -----------------------------
# Connection to CN0566 + Pluto
# -----------------------------
try:
    print("Connecting to CN0566 via ip:localhost...")
    my_phaser = CN0566(uri="ip:localhost")
    print("CN0566 found. Connecting to PlutoSDR via ip:192.168.2.1...")
    my_sdr = ad9361(uri="ip:192.168.2.1")
except:
    print("Connecting via fallback ip:phaser.local...")
    my_phaser = CN0566(uri="ip:phaser.local")
    my_sdr = ad9361(uri="ip:phaser.local:50901")

my_phaser.sdr = my_sdr

# -----------------------------
# Configuration of devices
# -----------------------------
time.sleep(0.5)
my_phaser.configure(device_mode="rx")
my_sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
my_sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
my_sdr._ctrl.debug_attrs["initialize"].value = "1"

my_sdr.rx_enabled_channels = [0, 1]
my_sdr._rxadc.set_kernel_buffers_count(1)
rx = my_sdr._ctrl.find_channel("voltage0")
rx.attrs["quadrature_tracking_en"].value = "1"
my_sdr.sample_rate = int(30e6)
my_sdr.rx_buffer_size = int(4 * 256)
my_sdr.rx_rf_bandwidth = int(10e6)
my_sdr.gain_control_mode_chan0 = "manual"
my_sdr.gain_control_mode_chan1 = "manual"
my_sdr.rx_hardwaregain_chan0 = 0
my_sdr.rx_hardwaregain_chan1 = 0
my_sdr.rx_lo = int(2.0e9)
my_sdr.filter = "LTE20_MHz.ftr"

my_sdr.tx_hardwaregain_chan0 = -80
my_sdr.tx_hardwaregain_chan1 = -80

# UWAGA: Częstotliwość sygnału znaleziona przez sweep
my_phaser.SignalFreq = 10.3943359375e9
# LO frequency dla całego systemu
my_phaser.lo = int(my_phaser.SignalFreq) + my_sdr.rx_lo

# Set gain on all 8 antennas
gain_list = [64] * 8
for i in range(len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i], apply_cal=False)

# Set phaser to boresight
my_phaser.set_beam_phase_diff(0.0)
my_phaser.Averages = 8

print("\n===============================")
print("       ANGLE DETECTION         ")
print("===============================")
print(f"Signal frequency: {my_phaser.SignalFreq/1e9:.4f} GHz")
print(f"SDR RX LO: {my_sdr.rx_lo/1e9:.4f} GHz")
print(f"Phaser LO: {my_phaser.lo/1e9:.4f} GHz")

try:
    while True:
        print("\nCollecting 2-channel RX data for angle estimation...")
        data = my_sdr.rx()
        
        # POPRAWKA: Użyj funkcji zachowującej fazę
        channel_data = process_received_data(data)
        
        print(f"RX0 complex: {channel_data[0]}")
        print(f"RX1 complex: {channel_data[1]}")
        # channel_data[0] = complex(20.3801880803942, -1.59724593307763)
        # channel_data[1] = complex(10.8204036839501, -16.9772803145076)
        
        # Sprawdź moc sygnału
        power_ch0 = np.abs(channel_data[0])**2
        power_ch1 = np.abs(channel_data[1])**2
        print(f"Power CH0: {power_ch0:.2e}, Power CH1: {power_ch1:.2e}")
        
        # Różnica faz
        phase_diff = calculate_phase_difference(channel_data[0], channel_data[1])
        print(f"Phase difference: {phase_diff:.2f}°")
        
        # Sprawdź czy mamy wystarczającą moc sygnału
        if power_ch0 < 1e-10 or power_ch1 < 1e-10:
            print("⚠️  Bardzo słaby sygnał - sprawdź konfigurację!")
            user_in = input("\nContinue measurement? (y/n): ").strip().lower()
            if user_in != 'y':
                break
            continue

        print("Estimating angle...")
        estimated_angle, cost_func = analyze_single_snapshot(channel_data)

        print(f"\n🎯 Estimated Angle: {(estimated_angle+45):.2f}°")

        # Plot cost function
        plot_angle_cost_function(cost_func, estimated_angle)

        # Plot data in complex plane
        plot_complex_plane(channel_data)

        user_in = input("\nContinue measurement? (y/n): ").strip().lower()
        if user_in != 'y':
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    print("\nCleaning up resources...")
    try:
        del my_sdr
        del my_phaser
        print("Resources cleaned up.")
    except:
        pass

print("Measurement session completed.")