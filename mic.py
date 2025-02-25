import pyaudio
from constants import SAMPLE_RATE, FORMAT, CHUNK_SIZE

# Set up the PyAudio object
p = pyaudio.PyAudio()

def get_audio_stream():
    ### Returns a PyAudio stream object with the default microphone as input

    # Returns
    # Audio stream in pyaudio.Stream
    return p.open(format=FORMAT, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)