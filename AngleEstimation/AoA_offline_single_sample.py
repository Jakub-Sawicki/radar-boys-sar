import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import sys

# ==========================================================
# ---- KONFIGURACJA ----
# ==========================================================

SignalFreq = 10.3943359e9
DATA_FILE = "measurements/30_deg_sar.npz"
# DATA_FILE = "measurements/30_deg_single.npz"

SAMPLE_INDEX = 165  # Numer próbki do analizy (np. 150)

# Stałe geometryczne anteny CN0566
ANTENNA_SPACING_MM = 24.24  # Rozstaw anten w mm

# ==========================================================
# ---- FUNKCJE ----
# ==========================================================

def load_measurements(file_path):
    """Wczytaj zapisane dane pomiarowe z pliku .npz"""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"\nWczytano plik: {file_path}")
        return data
    except FileNotFoundError:
        print(f"\nBŁĄD: Nie znaleziono pliku: {file_path}")
        sys.exit(1)

def a_vector(angle_deg, element_positions_mm, lambda_mm):          
    """
    Wektor sterujący (steering vector) dla danego kąta i pozycji anten
    """
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi / lambda_mm
    # reshape(-1, 1) zapewnia wektor kolumnowy
    return np.exp(1j * k * element_positions_mm * np.sin(angle_rad)).reshape(-1, 1)

def analyze_single_sample_mle(measurements_data, sample_idx, freq_hz, verbose=True):
    """
    Analiza MLE dla pojedynczej próbki (jednego 'strzału' radaru)
    z wykorzystaniem danych z obu kanałów (ch0 i ch1).
    """
    try:
        # Sprawdzenie czy indeks jest poprawny
        total_samples = len(measurements_data['ch0'])
        if sample_idx < 0 or sample_idx >= total_samples:
            print(f"[ERROR] Indeks próbki {sample_idx} poza zakresem (0-{total_samples-1})")
            return None

        print(f"Analiza próbki numer: {sample_idx} / {total_samples-1}")

        # Pobranie danych dla wybranej próbki
        # measurements_data['ch0'] to lista tablic, bierzemy element o indeksie sample_idx
        # Każdy element to wektor N próbek czasowych (np. z jednego chirpa)
        ch0_data = np.array(measurements_data['ch0'][sample_idx])
        ch1_data = np.array(measurements_data['ch1'][sample_idx])
        
        N_snapshots = len(ch0_data) # Liczba próbek w jednym pomiarze (snapshots)

        # Kalibracja fazy (zgodnie z Twoim oryginalnym kodem)
        calibration_phase_deg = 90
        calibration_factor = np.exp(1j * np.deg2rad(calibration_phase_deg))
        ch1_data = ch1_data * calibration_factor

        # Budowa macierzy sygnału Y [M x N]
        # M = 2 (liczba anten w tym jednym pomiarze)
        Y = np.vstack((ch0_data, ch1_data))
        M = Y.shape[0] # Powinno być 2

        # Macierz kowariancji R [M x M]
        # R = (Y * Y^H) / N
        R = (Y @ Y.conj().T) / N_snapshots

        # Parametry fali
        lambda_mm = 3e8 / freq_hz * 1e3
        
        # Pozycje anten w lokalnym układzie współrzędnych (dla pojedynczego pomiaru)
        # Pierwsza antena w 0, druga w odległości d
        element_positions_mm = np.array([0, ANTENNA_SPACING_MM])

        # Definicja funkcji kosztu MLE
        def cost_function(angle_deg):
            # Wektor sterujący a(theta)
            a_temp = a_vector(angle_deg, element_positions_mm, lambda_mm)
            a_temp_h = a_temp.conj().T
            
            # Mianownik normalizacyjny (a^H * a)
            denominator = a_temp_h @ a_temp
            if np.abs(denominator) < 1e-12: 
                return np.inf
            
            # Macierz projekcji na podprzestrzeń szumu: Pv = I - a * (a^H a)^-1 * a^H
            # Dla pojedynczego źródła a^H a to skalar, więc odwrotność to 1/skalar
            Pv = np.eye(M) - a_temp @ (1 / denominator) @ a_temp_h
            
            # Koszt MLE (deterministyczny): trace(Pv * R)
            # Szukamy minimum tej funkcji
            cost = np.abs(np.trace(Pv @ R))
            return cost

        # Rysowanie wykresu funkcji kosztu
        if verbose:
            angle_vec = np.arange(-45, 44.1, 0.1)
            pval = np.array([cost_function(a) for a in angle_vec])
            
            # Normalizacja Min-Max do zakresu [0, 1]
            if np.max(pval) != np.min(pval):
                pval_norm = (pval - np.min(pval)) / (np.max(pval) - np.min(pval))
            else:
                pval_norm = pval # Zabezpieczenie dla płaskiego wykresu
                
            plt.figure(figsize=(10, 6)) 
            plt.plot(-angle_vec, pval_norm, linewidth=2) # Dodane odbicie lustrzane żeby wynik zgadzał się z intuicyją kierunku
            plt.xlabel("Kąt [deg]")
            plt.ylabel("Znormalizowana funkcja kosztu")
            plt.title("MLE bez syntetycznej apertury")
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.minorticks_on()
            
            # Opcjonalnie: zaznacz minimum kropką
            min_idx = np.argmin(pval_norm)
            plt.scatter(-angle_vec[min_idx], pval_norm[min_idx], color='red', zorder=5)
            plt.text(-angle_vec[min_idx], pval_norm[min_idx] - 0.04, 
                        f" Min: {-angle_vec[min_idx]:.1f}°", ha='center', color='red', fontweight='bold')
            
            plt.show()

        # Optymalizacja - szukanie minimum
        result = minimize_scalar(cost_function, bounds=(-60, 60), method='bounded')
        return result.x

    except Exception as e:
        print(f"[ERROR] Błąd podczas analizy próbki {sample_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================================
# ---- MAIN ----
# ==========================================================

if __name__ == "__main__":
    print("="*60)
    print(f"MLE Single Sample Analysis - Sample #{SAMPLE_INDEX}")
    print("="*60)
    
    measurements_data = load_measurements(DATA_FILE)
    
    # Wywołanie analizy dla wybranej próbki
    estimated_angle = analyze_single_sample_mle(
        measurements_data, 
        SAMPLE_INDEX, 
        SignalFreq, 
        verbose=True
    )
    
    if estimated_angle is not None:
        print(f"\n[WYNIK] Estymowany kąt dla próbki {SAMPLE_INDEX}: {estimated_angle:.2f}°")
        
        # Opcjonalnie: Porównanie z prostą metodą fazową dla tej samej próbki
        from angle_estimation_enchanced import analyze_angle_estimation # Załóżmy, że masz to w innym pliku lub przekopiuj funkcję
        
        # Jeśli nie masz importu, użyj definicji lokalnej (zdefiniowałem ją wyżej w Twoim kodzie)
        # Tutaj używam Twojej funkcji analyze_angle_estimation z poprzedniego kodu
        ch0 = np.array(measurements_data['ch0'][SAMPLE_INDEX])
        ch1 = np.array(measurements_data['ch1'][SAMPLE_INDEX])
        
        angle_basic, phase_diff, _ = analyze_angle_estimation(ch0, ch1, SignalFreq)
        print(f"[PORÓWNANIE] Metoda interferometryczna (phase diff): {angle_basic:.2f}°")