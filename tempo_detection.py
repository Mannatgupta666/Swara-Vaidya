import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio
audio, sr = librosa.load("chant1.wav")

# Detect tempo and beats
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

print("Tempo (BPM):", tempo)

# Convert beat frames to time
beat_times = librosa.frames_to_time(beats, sr=sr)

print("Beat timestamps (seconds):")
print(beat_times)

# Plot waveform with beats
plt.figure(figsize=(10,4))

librosa.display.waveshow(audio, sr=sr)

plt.vlines(
    beat_times,
    ymin=min(audio),
    ymax=max(audio),
    color='r',
    label="Beats"
)

plt.title("Beat Detection")
plt.legend()
plt.show()