import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
audio, sr = librosa.load("chant1.wav")

# Detect pitch using pyin
pitch, voiced_flag, voiced_probs = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

pitch_clean = np.nan_to_num(pitch)

# Time axis
times = librosa.times_like(pitch_clean)

# Print first few pitch values
print(pitch_clean[:])

# Plot pitch vs time
plt.figure(figsize=(10,4))
plt.plot(times, pitch_clean)
plt.xlabel("Time (seconds)")
plt.ylabel("Pitch (Hz)")
plt.title("Pitch vs Time")
plt.show()