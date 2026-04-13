import sounddevice as sd
from scipy.io.wavfile import write

fs = 48000
seconds = 5

print("Recording... Speak now!")

recording = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1,
    device=18
)

sd.wait()

write("chant.wav", fs, recording)

print("Saved as chant.wav")