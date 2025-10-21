import matplotlib.pyplot as plt
import numpy as np

def plot_sar_heatmap(sar_data, extent=None, cmap='jet', title='SAR Image'):
    """
    Heatmap - podstawowa wizualizacja SAR
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(20*np.log10(np.abs(sar_data)), 
               cmap=cmap, 
               extent=extent,
               aspect='auto',
               origin='lower')
    plt.colorbar(label='Amplitude [dB]')
    plt.title(title)
    plt.xlabel('Range [m]')
    plt.ylabel('Azimuth [m]')
    plt.show()

# Przykład użycia
sar_data = np.random.rand(1000, 1000) + 1j*np.random.rand(1000, 1000)
plot_sar_heatmap(sar_data)