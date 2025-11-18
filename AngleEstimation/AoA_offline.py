import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import hilbert, detrend, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar, fminbound, minimize
import sys

# ==========================================================
# ---- KONFIGURACJA ----
# ==========================================================

SignalFreq = 10.3943359e9

DATA_FILE = "measurements/30_deg_sar.npz"
# DATA_FILE = "measurements/30_deg_single.npz"

# ==========================================================
# ---- FUNKCJE ----
# ==========================================================

def load_measurements(file_path):
    """Wczytaj zapisane dane pomiarowe z pliku .npz"""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"\nWczytano plik: {file_path}")
    except FileNotFoundError:
        print(f"\nBŁĄD: Nie znaleziono pliku: {file_path}")
        sys.exit(1)

    return data


def analyze_angle_estimation(data_ch0, data_ch1, freq_hz=10.3943359375e9, sample_rate=30e6):
    """
    Podstawowa estymacja kąta przylotu sygnału używając różnicy fazowej
    """
    c = 3e8  # prędkość światła
    lambda_m = c / freq_hz  # długość fali w metrach
    d_m = 0.02424  # rozstaw anten w metrach
    d_over_lambda = d_m / lambda_m  # normalizowany rozstaw
    
    correlation = np.mean(data_ch0 * np.conj(data_ch1))

    calibrated_correlation = correlation * np.exp(-1j * np.deg2rad(90))
    
    # Różnica fazowa między kanałami
    phase_diff = np.angle(calibrated_correlation)
    
    # Estymacja kąta na podstawie różnicy fazowej
    sin_theta = phase_diff / (2 * np.pi * d_over_lambda)
    
    if abs(sin_theta) > 1:
        sin_theta = np.clip(sin_theta, -1, 1)
    
    estimated_angle_rad = np.arcsin(sin_theta)
    estimated_angle_deg = np.rad2deg(estimated_angle_rad)
    
    return estimated_angle_deg, phase_diff, calibrated_correlation


def a_sar(angle_deg, element_positions_mm, lambda_mm):           
    """
    Steering vector dla SAR
    """
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi / lambda_mm
    return np.exp(1j * k * element_positions_mm * np.sin(angle_rad)).reshape(-1, 1)


def MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=False):
    """
    MLE estymacja kąta z danych ze wszystkich pozycji
    """
    M, N = Y.shape
    R = (Y @ Y.conj().T) / N
    lambda_mm = 3e8 / freq_hz * 1e3

    ch0_idx = np.arange(0, M, 2)  # Indeksy lewych anten
    ch1_idx = np.arange(1, M, 2)  # Indeksy prawych anten

    Y_left = Y[ch0_idx, :]
    Y_right = Y[ch1_idx, :]

    positions_left = element_positions_mm[ch0_idx]
    positions_right = element_positions_mm[ch1_idx]

    Y_sorted = np.vstack((Y_left, Y_right))
    element_positions_sorted = np.concatenate((positions_left, positions_right)) 

    Y = Y_sorted
    element_positions_mm = element_positions_sorted
    M = Y.shape[0] 

    def cost_function(angle_deg):
        a_temp = a_sar(angle_deg, element_positions_mm, lambda_mm)
        a_temp_h = a_temp.conj().T
        denominator = a_temp_h @ a_temp

        if np.abs(denominator) < 1e-12:         # ochrona przed dzieleniem przez 0
            return np.inf
        
        Pv = np.eye(M) - a_temp @ (1 / denominator) @ a_temp_h
        cost = np.abs(np.trace(Pv @ R))
        return cost

    if verbose:
        angle_vec = np.arange(-45, 44.1, 0.1)
        pval = np.array([cost_function(a) for a in angle_vec])
        plt.figure()
        plt.plot(angle_vec, pval)
        plt.xlabel("Kąt [deg]")
        plt.ylabel("Funkcja kosztu")
        plt.title("MLE z syntetycznej apertury")
        plt.grid(True)
        plt.show()

    result = minimize_scalar(cost_function, bounds=(-45, 44), method='bounded')
    return result.x


def analyze_mle_with_aperture_dual(measurements_data, freq_hz, verbose=True):
    """
    Analiza MLE z wykorzystaniem danych z obu kanałów i wszystkich pozycji
    """
    try:
        M = len(measurements_data['ch0'])  # liczba pozycji
        N = len(measurements_data['ch0'][0])  # liczba próbek

        calibration_phase_deg = 90
        calibration_factor = np.exp(1j * np.deg2rad(calibration_phase_deg))

        ch0_stack = []
        ch1_stack = []
        positions_mm = []

        for i in range(M):
            ch0 = np.array(measurements_data['ch0'][i])
            ch1 = np.array(measurements_data['ch1'][i]) * calibration_factor
            ch0_stack.append(ch0)
            ch1_stack.append(ch1)

            pos_mm = measurements_data['positions'][i] * 1000  # m → mm
            positions_mm.extend([pos_mm, pos_mm + 24.24])  # pozycja anteny 0 i 1

        Y = np.vstack(ch0_stack + ch1_stack)  
        element_positions_mm = np.array(positions_mm)  

        # Estymacja kąta
        estimated_angle = MLE_sar_full_aperture_dual(Y, element_positions_mm, freq_hz, verbose=verbose)
        print(f"Kąt DOA z syntetycznej apertury: {estimated_angle:.2f}°")
        return estimated_angle
    except Exception as e:
        print(f"[ERROR] Błąd podczas analizy MLE z 2-kanałowej syntetycznej apertury: {e}")
        return None

def fix_position_scaling(measurements_data):
    """
    Naprawia błędne skalowanie pozycji w danych pomiarowych.
    Błąd: STEP_SIZE_M był 0.0001 (0.1mm), a powinien być 0.001 (1mm).
    Korekta: Mnożymy pozycje przez 10.
    """
    print("[INFO] Stosowanie korekty skalowania pozycji (x10)...")
    
    # Sprawdźmy czy dane są w słowniku (z .npz po wczytaniu) czy w obiekcie NpzFile
    # np.load z allow_pickle=True zwraca obiekt NpzFile, który zachowuje się jak słownik,
    # ale jest tylko do odczytu. Musimy przekonwertować go na zwykły słownik.
    
    data_dict = dict(measurements_data)
    
    # Oryginalne pozycje (z błędnym krokiem 0.0001)
    positions = data_dict['positions']
    
    # Nowe pozycje (z poprawnym krokiem 0.001)
    # Mnożymy x10, bo 0.001 / 0.0001 = 10
    corrected_positions = positions * 10.0
    
    data_dict['positions'] = corrected_positions
    
    print(f"   Stary zakres: {positions[0]:.4f}m - {positions[-1]:.4f}m")
    print(f"   Nowy zakres:  {corrected_positions[0]:.4f}m - {corrected_positions[-1]:.4f}m")
    
    return data_dict


if __name__ == "__main__":
    print("="*60)
    print("Angle of arrival estimation - Offline Processing")
    print("="*60)
    
    measurements_data = load_measurements(DATA_FILE)
    measurements_data = fix_position_scaling(measurements_data)

    # Analiza konkretnego pomiaru
    sample = 0
    angle_basic, _, _ = analyze_angle_estimation(measurements_data['ch0'][sample], measurements_data['ch1'][sample])
    print(f"Wynik dla próbki numer {sample}: {angle_basic}")

    angle_sar_aperture_dual = analyze_mle_with_aperture_dual(measurements_data, SignalFreq, verbose=True)
    if angle_sar_aperture_dual is not None:
        print(f"[WYNIKI] MLE SAR (syntetyczna apertura, 2 kanały): {angle_sar_aperture_dual:.2f}°")


