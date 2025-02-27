import pyaudio as pa
import struct
import numpy as np
import matplotlib.pyplot as plt

# Constantes
CHUNK = 1024 * 2       # Número de muestras por cuadro
FORMAT = pa.paInt16    # Formato de audio de 16 bits
CHANNELS = 1           # Canal mono
RATE = 44100           # Tasa de muestreo en Hz
N_FFT = CHUNK          # Tamaño de FFT
N_TIME = 100           # Número de pasos de tiempo en el espectrograma

# Inicializar PyAudio y el flujo de audio
p = pa.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Configurar la gráfica
fig, (ax, ax1, ax2) = plt.subplots(3, figsize=(10, 8))

# Gráfica en el dominio del tiempo
x = np.arange(0, 2 * CHUNK, 2)  # Eje de tiempo
line, = ax.plot(x, np.random.rand(CHUNK), 'r')
ax.set_ylim(-32000, 32000)
ax.set_xlim(0, CHUNK)
ax.set_title("Señal de Audio en Tiempo Real (Dominio del Tiempo)")
ax.set_xlabel("Muestras")
ax.set_ylabel("Amplitud")

# Gráfica en el dominio de la frecuencia
freqs = np.fft.rfftfreq(N_FFT, 1 / RATE)
line_fft, = ax1.semilogx(freqs, np.random.rand(len(freqs)), 'b')
ax1.set_ylim(0, 1)
ax1.set_xlim(20, RATE / 2)
ax1.set_title("Espectro de Frecuencia (Dominio de la Frecuencia)")
ax1.set_xlabel("Frecuencia (Hz)")
ax1.set_ylabel("Magnitud")

# Configuración del espectrograma
spectrogram_data = np.zeros((len(freqs), N_TIME))  # Almacenamiento de Tiempo x Frecuencia
img = ax2.imshow(spectrogram_data, aspect='auto', origin='lower',
                 extent=[0, N_TIME, 20, RATE / 2], cmap='magma')
ax2.set_yscale('log')
ax2.set_xlim(0, N_TIME)
ax2.set_ylim(20, RATE / 2)
ax2.set_title("Espectrograma (Tiempo vs Frecuencia)")
ax2.set_xlabel("Pasos de Tiempo")
ax2.set_ylabel("Frecuencia (Hz)")
fig.colorbar(img, ax=ax2, label="Magnitud (dB)")

plt.tight_layout()
plt.show(block=False)

print("Captura de Audio en Tiempo Real: Hable en el micrófono y observe las gráficas.")

# Bucle de graficación en tiempo real
try:
    while plt.fignum_exists(fig.number):
        # Leer y desempaquetar datos de audio
        data = stream.read(CHUNK)
        dataInt = struct.unpack(str(CHUNK) + 'h', data)

        # Procesamiento FFT
        fft_magnitude = np.abs(np.fft.rfft(dataInt)) * 2 / (33000 * CHUNK)
        fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)  # Convertir a escala dB

        # Actualizar gráfica en el dominio del tiempo
        line.set_ydata(dataInt)
        
        # Actualizar gráfica en el dominio de la frecuencia
        line_fft.set_ydata(fft_magnitude)

        # Actualizar datos del espectrograma (desplazar a la izquierda, agregar nueva columna)
        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        spectrogram_data[:, -1] = fft_magnitude_db  # Agregar última FFT en escala dB
        
        # Actualizar imagen del espectrograma
        img.set_data(spectrogram_data)
        img.set_clim(vmin=-80, vmax=0)  # Ajustar rango dB

        # Refrescar gráfica
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # Pequeña pausa para permitir el procesamiento de eventos GUI

except KeyboardInterrupt:
    print("Captura de audio detenida.")
    
finally:
    # Limpiar y cerrar el flujo
    stream.stop_stream()
    stream.close()
    p.terminate()
