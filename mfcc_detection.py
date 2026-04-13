import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio
audio, sr = librosa.load("chant1.wav")

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

print("MFCC shape:", mfccs.shape)

# Plot MFCC
plt.figure(figsize=(10,4))

librosa.display.specshow(
    mfccs,
    x_axis='time',
    y_axis='mel',
    sr=sr,
    cmap='magma'
)

plt.colorbar()

plt.title("MFCC Features")

plt.tight_layout()

plt.show()