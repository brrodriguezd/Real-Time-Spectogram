import pyaudio

# Set up the PyAudio object
sample_rate = 44100
format = pyaudio.paInt16
chunk = 1024
p = pyaudio.PyAudio()

def get_audio_stream():
    ### Returns a PyAudio stream object with the default microphone as input

    # Returns
    # Audio stream in pyaudio.Stream
    return p.open(format=format, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk)