import librosa
import numpy as np
import json

# Load audio
audio, sr = librosa.load("chant1.wav")

# -------------------------
# 1. Pitch detection
# -------------------------
pitch, voiced_flag, voiced_probs = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

pitch_clean = np.nan_to_num(pitch)

# -------------------------
# 2. Tempo detection
# -------------------------
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

# -------------------------
# 3. MFCC extraction
# -------------------------
mfccs = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

# Convert numpy arrays → list (required for JSON)
features = {
    "pitch_contour": pitch_clean.tolist(),
    "tempo": float(tempo),
    "beat_times": beats.tolist(),               
    "duration": len(audio) / sr,               
    "mfcc": mfccs.tolist()
}

# -------------------------
# Save features
# -------------------------
with open("reference_features.json", "w") as f:
    json.dump(features, f, indent=4)

print("Reference features saved!")