import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
audio, sr = librosa.load("chant1.wav")

print("Sampling Rate:", sr)
print("Audio shape:", audio.shape)

# Waveform
plt.figure(figsize=(10,3))
librosa.display.waveshow(audio, sr=sr)
plt.title("Waveform")
plt.show()

# Create spectrogram
spectrogram = librosa.amplitude_to_db(
    abs(librosa.stft(audio)),
    ref=np.max
)

plt.figure(figsize=(10,4))
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Spectrogram")
plt.show()