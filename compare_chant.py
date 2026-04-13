import librosa
import numpy as np
import json
import whisper
from sklearn.preprocessing import StandardScaler
from librosa.sequence import dtw
from difflib import SequenceMatcher
import re

# -------------------------
# Load reference features
# -------------------------
with open("reference_features.json", "r") as f:
    reference = json.load(f)

ref_pitch = np.array(reference["pitch_contour"])
ref_tempo = reference["tempo"]
ref_mfcc = np.array(reference["mfcc"])

# -------------------------
# Load reference audio
# -------------------------
ref_audio, ref_sr = librosa.load("chant1.wav")
ref_audio, _ = librosa.effects.trim(ref_audio)

# -------------------------
# Load user chant
# -------------------------
audio, sr = librosa.load("user_chant2.wav")
audio, _ = librosa.effects.trim(audio)

# Pitch
pitch, _, _ = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)
pitch = np.nan_to_num(pitch)

# Tempo
tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

# MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# -------------------------
# 1. Pitch similarity
# -------------------------
pitch_voiced = pitch[pitch > 0]
ref_pitch_voiced = ref_pitch[ref_pitch > 0]

if len(pitch_voiced) == 0 or len(ref_pitch_voiced) == 0:
    pitch_similarity = 0.0
else:
    pitch_log = np.log2(pitch_voiced)
    ref_pitch_log = np.log2(ref_pitch_voiced)
    pitch_norm     = (pitch_log - np.mean(pitch_log))     / (np.std(pitch_log)     + 1e-9)
    ref_pitch_norm = (ref_pitch_log - np.mean(ref_pitch_log)) / (np.std(ref_pitch_log) + 1e-9)

    D, wp = dtw(pitch_norm.reshape(1, -1), ref_pitch_norm.reshape(1, -1), metric='euclidean')
    dtw_distance = D[-1, -1] / len(wp)
    pitch_similarity = 100 * np.exp(-dtw_distance / 5)

# -------------------------
# 2. Rhythm similarity
# (Onset envelope — works for slow chants, no beat needed)
# -------------------------
def get_onset_envelope(y, sr):
    """
    Captures syllable attack patterns over time.
    Much better than beat_track for mantras/chants
    which have no drum beats.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.max() > 0:
        onset_env = onset_env / onset_env.max()
    return onset_env
 
user_onset = get_onset_envelope(audio, sr)
ref_onset  = get_onset_envelope(ref_audio, ref_sr)
 
# DTW comparison — handles different lengths naturally
D, wp = dtw(
    user_onset.reshape(1, -1),
    ref_onset.reshape(1, -1),
    metric='euclidean'
)
onset_dist       = D[-1, -1] / len(wp)
tempo_similarity = 100 * np.exp(-onset_dist / 0.3)

# -------------------------
# 3. Voice similarity (MFCC)
# -------------------------
mfcc_scaled = StandardScaler().fit_transform(mfcc.T).T
ref_mfcc_scaled = StandardScaler().fit_transform(ref_mfcc.T).T

delta = librosa.feature.delta(mfcc_scaled)
ref_delta = librosa.feature.delta(ref_mfcc_scaled)
mfcc_combined = np.vstack([mfcc_scaled, delta])
ref_mfcc_combined = np.vstack([ref_mfcc_scaled, ref_delta])

D, wp = dtw(mfcc_combined, ref_mfcc_combined, metric='euclidean')
mfcc_distance = D[-1, -1] / len(wp)
k = 50
mfcc_similarity = 100 * np.exp(-mfcc_distance / k)

# --------------------------------------------------------
# 4. Text similarity (No eSpeak / No phonemizer needed)
# --------------------------------------------------------
model = whisper.load_model("base")

# Transcribe user audio
# Tip: initial_prompt guides Whisper toward Sanskrit mantra sounds
ref_mantra = reference["mantra_text"]
user_result = model.transcribe(
    "user_chant2.wav",
    language="en",
    initial_prompt=f"This is a Sanskrit mantra chant: {ref_mantra}"
)
user_text = user_result["text"].lower().strip()
ref_text = ref_mantra.lower().strip()

print("Reference text :", ref_text)
print("User text      :", user_text)

# -------------------------
# Clean text
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

ref_clean = clean_text(ref_text)
user_clean = clean_text(user_text)

print("Cleaned reference :", ref_clean)
print("Cleaned user      :", user_clean)

# -------------------------
# Sanskrit syllable splitter
# (much better than eSpeak for mantras)
# -------------------------
def split_syllables(text):
    """
    Splits romanized Sanskrit/mantra text into syllables.
    Captures consonant clusters + vowel + optional trailing consonant.
    Works well for: om, namah, shivaya, ganapataye, etc.
    """
    return re.findall(r'[bcdfghjklmnpqrstvwxyz]*[aeiou]+[bcdfghjklmnpqrstvwxyz]*', text)

ref_syllables  = split_syllables(ref_clean)
user_syllables = split_syllables(user_clean)

print("Reference syllables :", ref_syllables)
print("User syllables      :", user_syllables)

# -------------------------
# 3-level text similarity
# -------------------------

# Level 1: Full character sequence (catches overall structure)
char_sim = SequenceMatcher(None, ref_clean, user_clean).ratio()

# Level 2: Word-level (checks if right words spoken)
ref_words  = ref_clean.split()
user_words = user_clean.split()
word_sim = SequenceMatcher(None, ref_words, user_words).ratio()

# Level 3: Syllable-level (best for Sanskrit pronunciation accuracy)
syl_sim = SequenceMatcher(None, ref_syllables, user_syllables).ratio()

# Weighted combination — syllable similarity weighted highest
text_similarity = (
    0.20 * char_sim +
    0.30 * word_sim +
    0.50 * syl_sim
) * 100

print("\nChar similarity     :", round(char_sim * 100, 2), "%")
print("Word similarity     :", round(word_sim * 100, 2), "%")
print("Syllable similarity :", round(syl_sim * 100, 2), "%")
print("Text similarity     :", round(text_similarity, 2), "%")

# -------------------------
# 5. Overall score
# -------------------------
overall = (
    0.40 * text_similarity  +   # pronunciation accuracy
    0.25 * mfcc_similarity  +   # voice quality / timbre
    0.20 * pitch_similarity +   # pitch accuracy
    0.15 * tempo_similarity     # rhythm
)

# -------------------------
# Results
# -------------------------
print("\n========== RESULTS ==========")
print("Text Similarity  :", round(text_similarity, 2), "%")
print("Pitch Similarity :", round(pitch_similarity, 2), "%")
print("Rhythm Similarity:", round(tempo_similarity, 2), "%")
print("Voice Similarity :", round(mfcc_similarity,  2), "%")
print("Overall Score    :", round(overall, 2), "%")
print("==============================")