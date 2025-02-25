from matplotlib.animation import FuncAnimation
from constants import *
import numpy as np
import matplotlib.pyplot as plt

from mic import get_audio_stream

def plot_spectrogram():
    # Set up frequency axis
    freq_bins = np.fft.rfftfreq(CHUNK_SIZE, 1.0/SAMPLE_RATE)
    
    # Create the figure and axes for plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize empty spectrogram data
    spectrogram_data = np.zeros((len(freq_bins),NUM_CHUNKS))
    
    # Create the spectrogram image
    # -60 to 0 dB 
    img = ax.imshow(
        spectrogram_data,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=60,
        interpolation='nearest'
    )
    
    # Set up the colorbar
    cbar = fig.colorbar(img)
    cbar.set_label('Amplitude (dB)')
    
    # Set up plot labels
    ax.set_title('Espectrograma')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Frequency (Hz)')
    
    # Set custom y-tick labels based on frequency bins
    num_yticks = 10
    ytick_indices = np.linspace(0, len(freq_bins)-1, num_yticks, dtype=int)
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels([f"{freq_bins[i]:.0f}" for i in ytick_indices])

    stream = get_audio_stream()
    # Function to update the plot with new audio data
    def update_plot(frame):
        nonlocal spectrogram_data
        
        # Read audio data
        audio_data = np.frombuffer(stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)
        
        # Compute FFT and convert to dB scale
        fft_data = np.abs(np.fft.rfft(audio_data))
        
        # Convert to dB (with protection against log(0))
        epsilon = 1e-10
        fft_db = 20 * np.log10(fft_data + epsilon)
        
        # Shift the spectrogram data to the left and update the rightmost column
        spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
        spectrogram_data[:, -1] = fft_db
        
        # Update the image data
        img.set_array(spectrogram_data)
        
        return [img]
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, interval=30, blit=True, cache_frame_data=True) 
    
    # Set up function to clean up resources when the plot window is closed
    def on_close(event):
        stream.stop_stream()
        stream.close()
        plt.close(fig)
   
    fig.canvas.mpl_connect('close_event', on_close)

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Clean up resources (in case plt.show() returns)
    stream.stop_stream()
    stream.close()
