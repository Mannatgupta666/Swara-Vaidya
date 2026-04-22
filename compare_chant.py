import librosa
import numpy as np
import json
import whisper
from sklearn.preprocessing import StandardScaler
from librosa.sequence import dtw
from difflib import SequenceMatcher
import re
from scipy.ndimage import uniform_filter1d         # Smooth pitch curve
from scipy.signal import find_peaks, resample      # Peak detection, resamp
from feedback_generator import generate_feedback

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
audio, sr = librosa.load("user_chant1.wav")
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
# We compare how similar the pitch movements are between reference and user.
# Steps:
#   1. Keep only voiced frames (where pitch > 0, i.e. voice is active)
#   2. Convert to log scale (human ears hear pitch on a log scale)
#   3. Z-score normalize (removes gender/voice type difference)
#   4. Resample to same length (DTW works better with equal length)
#   5. DTW comparison
#   6. Convert distance to similarity score
 
pitch_voiced     = pitch[pitch > 0]
ref_pitch_voiced = ref_pitch[ref_pitch > 0]
 
if len(pitch_voiced) == 0 or len(ref_pitch_voiced) == 0:
    pitch_similarity = 0.0
 
else:
    # Log scale — because a jump from 100Hz to 200Hz feels the same
    # as a jump from 200Hz to 400Hz to the human ear
    pitch_log     = np.log2(pitch_voiced)
    ref_pitch_log = np.log2(ref_pitch_voiced)
 
    # Z-score normalization:
    #   subtract mean → removes absolute pitch level (gender difference)
    #   divide by std  → removes pitch range difference
    # Now both curves are centered at 0 and have same scale.
    # DTW only sees the SHAPE of the melody, not the absolute key.
    pitch_norm     = (pitch_log - np.mean(pitch_log))         / (np.std(pitch_log)         + 1e-9)
    ref_pitch_norm = (ref_pitch_log - np.mean(ref_pitch_log)) / (np.std(ref_pitch_log)     + 1e-9)
 
    # Resample both to length 100 — equal length improves DTW accuracy
    # (Idea from the improved pitch analysis document)
    pitch_resampled     = resample(pitch_norm,     100)
    ref_pitch_resampled = resample(ref_pitch_norm, 100)
 
    # DTW = Dynamic Time Warping
    # Finds the best alignment between two sequences even if they have
    # different timing (user chanted slower or faster than reference).
    # D[-1,-1] = total cost of the best alignment path
    # wp       = the actual alignment path
    # We divide by len(wp) to get average cost per step (normalize for length)
    D, wp = dtw(
        pitch_resampled.reshape(1, -1),
        ref_pitch_resampled.reshape(1, -1),
        metric='euclidean'
    )
    dtw_distance     = D[-1, -1] / len(wp)
 
    # Convert distance to 0-100% score using exponential decay:
    # distance=0   → score=100% (perfect)
    # distance=5   → score=37%
    # distance=10  → score=14%
    pitch_similarity = 100 * np.exp(-dtw_distance / 5)
 
 
# =============================================================================
# 1-a PITCH ZONE ANALYSIS (Vedic Accent: Udatta / Anudatta / Svarita)
# =============================================================================
# Based on research by Begus (2016) and BHASHA 2025 paper.
#
# Vedic Sanskrit has 3 pitch accent zones:
#   Anudatta  = low flat region (before the accent)
#   Udatta    = rising region   (the accent peak)
#   Svarita   = falling region  (after the peak)
#
# Instead of comparing just the pitch numbers,
# we detect WHICH ZONE is happening at each moment using slope:
#   slope > 0  → pitch is rising  → Udatta
#   slope < 0  → pitch is falling → Svarita
#   slope ≈ 0  → pitch is flat    → Anudatta
 
def extract_pitch_zones(pitch_contour):
    """
    IMPROVED VERSION — from Document 8 (upgraded pitch zone extraction).
    
    Takes a raw pitch array.
    Returns list of zones detected: anudatta / udatta / svarita
    and the normalized pitch array.
    
    Upgrades over old version:
      - Uses find_peaks() for robust peak detection (handles multiple peaks)
      - Finds dominant peak near CENTER of contour (more reliable for mantras)
      - Handles plateau shapes (long flat sections don't confuse the detector)
      - Slope threshold prevents noise from creating false zone switches
    """
    # Only keep voiced frames (where pitch > 0)
    voiced = [(i, p) for i, p in enumerate(pitch_contour) if p > 0]
 
    if len(voiced) < 6:
        return [], np.array([])
 
    pitches = np.array([v[1] for v in voiced])
 
    # --- Step 1: Smooth pitch ---
    # Moving average with window=5 removes frame-to-frame jitter
    # without destroying the overall shape of the curve
    smoothed = uniform_filter1d(pitches, size=5)
 
    # --- Step 2: Z-score normalize (speaker/gender independent) ---
    # After this step: mean=0, std=1 for both male and female voices
    # DTW comparison becomes gender-neutral
    norm = (smoothed - np.mean(smoothed)) / (np.std(smoothed) + 1e-9)
 
    # --- Step 3: Compute slope (gradient) ---
    # np.gradient = rate of change at each point
    # slope[i] > 0  → pitch rising at frame i
    # slope[i] < 0  → pitch falling at frame i
    # slope[i] ≈ 0  → pitch flat at frame i
    slope = np.gradient(norm)
 
    # --- Step 4: Find dominant peak near center ---
    # find_peaks() finds ALL local maxima in the array
    # We then pick the one closest to the CENTER of the contour
    # Why center? Because in most mantras the main accent peak
    # falls roughly in the middle of the phrase, not at the edges.
    # This is more robust than just taking np.argmax() which can
    # pick an edge artifact.
    peaks, _ = find_peaks(norm)
    if len(peaks) == 0:
        # No peaks found — fall back to global maximum
        peak_idx = int(np.argmax(norm))
    else:
        center   = len(norm) // 2
        # Pick the peak closest to center
        peak_idx = peaks[int(np.argmin(np.abs(peaks - center)))]
 
    # --- Step 5: Classify each frame into a zone using slope ---
    # threshold=0.05 means slope must be meaningfully positive/negative
    # to count as rising/falling. This prevents noise spikes from
    # creating spurious zone changes.
    zones        = []
    current_zone = None
    start        = 0
    threshold    = 0.05
 
    for i in range(len(norm)):
        if slope[i] > threshold:
            zone = "udatta"      # rising → Udatta accent region
        elif slope[i] < -threshold:
            zone = "svarita"     # falling → Svarita (post-peak fall)
        else:
            zone = "anudatta"    # flat/low → Anudatta (pre-accent region)
 
        if current_zone is None:
            current_zone = zone
            start        = i
        elif zone != current_zone:
            zones.append({
                "zone":      current_zone,
                "avg_pitch": float(np.mean(norm[start:i])),
                "length":    i - start
            })
            current_zone = zone
            start        = i
 
    # Save last zone
    zones.append({
        "zone":      current_zone,
        "avg_pitch": float(np.mean(norm[start:])),
        "length":    len(norm) - start
    })
 
    return zones, norm
 
 
def compare_pitch_zones(ref_zones, user_zones):
    """
    Compares Vedic accent zones between reference and user.
    Returns: zone_similarity (0-100), list of feedback messages
    """
    if not ref_zones or not user_zones:
        return 0.0, ["Could not detect pitch zones — recording may be too quiet"]
 
    feedback = []
    scores   = []
 
    # Convert list of zones to dict for easy lookup
    # Note: if same zone appears multiple times, last one wins
    # This is fine for a first pass
    ref_dict  = {z["zone"]: z for z in ref_zones}
    user_dict = {z["zone"]: z for z in user_zones}
 
    for zone in ["anudatta", "udatta", "svarita"]:
        if zone not in ref_dict:
            continue   # reference doesn't have this zone, skip
 
        if zone not in user_dict:
            feedback.append(f"Missing {zone} section in your chant")
            scores.append(0.0)
            continue
 
        ref_z  = ref_dict[zone]
        user_z = user_dict[zone]
 
        # Compare normalized pitch levels
        # Since both are normalized, difference should be small for good match
        pitch_diff = abs(ref_z["avg_pitch"] - user_z["avg_pitch"])
        zone_score = max(0.0, 1.0 - pitch_diff)
        scores.append(zone_score)
 
        # Generate specific feedback
        if zone == "udatta" and zone_score < 0.7:
            feedback.append("Udatta (rising accent) is not rising high enough")
        elif zone == "svarita" and zone_score < 0.7:
            feedback.append("Svarita (falling tone after peak) is not dropping correctly")
        elif zone == "anudatta" and zone_score < 0.7:
            feedback.append("Anudatta (low starting region) pitch is unstable")
 
    # Check peak timing
    # The Beguš paper shows that WHEN the peak happens matters —
    # independent svarita peaks earlier (compressed udatta)
    ref_total  = sum(z["length"] for z in ref_zones)
    user_total = sum(z["length"] for z in user_zones)
 
    if ref_total > 0 and user_total > 0:
        ref_peak_ratio  = ref_dict.get("udatta",  {}).get("length", 0) / ref_total
        user_peak_ratio = user_dict.get("udatta", {}).get("length", 0) / user_total
        timing_diff     = abs(ref_peak_ratio - user_peak_ratio)
 
        if timing_diff > 0.2:
            feedback.append(
                f"Pitch peak timing is off — "
                f"reference peaks at {round(ref_peak_ratio * 100)}% of chant, "
                f"you peaked at {round(user_peak_ratio * 100)}%"
            )
 
    zone_similarity = float(np.mean(scores)) * 100 if scores else 0.0
    return zone_similarity, feedback
 
 
# --- Run zone analysis ---
ref_zones,  ref_norm  = extract_pitch_zones(ref_pitch)
user_zones, user_norm = extract_pitch_zones(pitch)
 
zone_sim, zone_feedback = compare_pitch_zones(ref_zones, user_zones)
 
 
# =============================================================================
# 1-b — ACCENT LEVEL CLASSIFIER (BHASHA 2025 paper)
# =============================================================================
# The BHASHA paper found that Rigvedic Sanskrit has 3 pitch accent levels:
#   Anudatta = low    (label 1)
#   Neutral  = medium (label 2)
#   Udatta   = high   (label 3)
#
# A wrong accent CHANGES THE MEANING of the Sanskrit word.
# Your pitch contour DTW catches overall shape, but a user could follow
# the general melody while placing the HIGH tone on the WRONG syllable.
# This classifier catches that specific error.
#
# How it works:
#   - Take all voiced (non-zero) pitch values
#   - Find the 33rd percentile (low_thresh) and 66th percentile (high_thresh)
#   - Classify every frame: below low_thresh=Anudatta, above high_thresh=Udatta
#   - Compare the sequence of labels using SequenceMatcher
 
def classify_accent_levels(pitch_contour):
    """
    Classifies each pitch frame into one of 3 Vedic accent levels.
    Returns array of labels: 0=silence, 1=Anudatta(low), 2=neutral, 3=Udatta(high)
    
    Based on BHASHA 2025 paper — accent placement is linguistically meaningful.
    A wrong label at a syllable = wrong accent = possibly wrong meaning.
    """
    voiced = pitch_contour[pitch_contour > 0]
    if len(voiced) == 0:
        return np.array([])
 
    # Percentile thresholds divide pitch range into 3 equal parts
    # 33rd percentile = lower third of pitch range = Anudatta region
    # 66th percentile = upper third of pitch range = Udatta region
    low_thresh  = np.percentile(voiced, 33)
    high_thresh = np.percentile(voiced, 66)
 
    levels = []
    for p in pitch_contour:
        if p == 0:
            levels.append(0)    # silence — not voiced
        elif p < low_thresh:
            levels.append(1)    # Anudatta — low pitch
        elif p < high_thresh:
            levels.append(2)    # neutral — middle pitch
        else:
            levels.append(3)    # Udatta — high pitch (accent peak)
    return np.array(levels)
 
ref_accent_levels  = classify_accent_levels(ref_pitch)
user_accent_levels = classify_accent_levels(pitch)
 
# Compare accent label sequences using SequenceMatcher
# This checks: does the HIGH label appear at the same relative positions?
# If user has Udatta (3) where reference has Anudatta (1) — that's an accent error.
if len(ref_accent_levels) > 0 and len(user_accent_levels) > 0:
    accent_sim = SequenceMatcher(
        None,
        ref_accent_levels.tolist(),
        user_accent_levels.tolist()
    ).ratio() * 100
else:
    accent_sim = 0.0
 
print("Accent Level Similarity:", round(accent_sim, 2), "%")
 
 
# =============================================================================
# 1-c — INTRA-SYLLABLE PITCH SHAPE DETECTOR (Beguš 2016 paper)
# =============================================================================
# The Beguš paper explains that within a single syllable,
# the DIRECTION of pitch movement identifies the accent type:
#
#   Udatta   = rising pitch within the syllable     (low → high)
#   Anudatta = falling pitch within the syllable    (high → low)
#   Svarita  = rising-then-falling (circumflex)     (low → high → low)
#   Flat     = sustained note, no clear direction
#
# This is DIFFERENT from the zone analysis (Section 5a) which looks at
# the full contour. This looks at small windows (individual syllables).
#
# Why does this matter?
# Example: user chants a syllable with a flat tone where
# reference has a rising (Udatta) tone. Zone analysis might miss this
# if the overall contour shape is similar. This detector catches it.
 
def detect_pitch_shape(pitch_segment):
    """
    Classifies the pitch movement WITHIN a short segment (one syllable).
    
    Returns one of: 'rising', 'falling', 'rising_falling', 'flat'
    
    Based on Beguš 2016:
      rising        = Udatta accent
      falling       = Anudatta accent  
      rising_falling = Svarita (independent svarita = compressed udatta)
      flat          = sustained, no accent
    """
    # Need at least 3 frames to detect a shape
    if len(pitch_segment) < 3:
        return 'flat'
 
    mid   = len(pitch_segment) // 2
    first = np.mean(pitch_segment[:mid])    # average pitch in first half
    last  = np.mean(pitch_segment[mid:])    # average pitch in second half
    peak  = np.max(pitch_segment)           # highest point in the segment
 
    # 1.05 threshold = must be 5% higher to count as meaningful change
    # This prevents tiny fluctuations from being classified as movement
    rising          = last  > first * 1.05
    falling         = first > last  * 1.05
    # Peak in middle = pitch went up then came down within this segment
    has_peak_middle = (peak > first * 1.05) and (peak > last * 1.05)
 
    if has_peak_middle:
        return 'rising_falling'   # Svarita — circumflex shape
    elif rising:
        return 'rising'           # Udatta — accent rising
    elif falling:
        return 'falling'          # Anudatta — falling after accent
    else:
        return 'flat'             # sustained, no clear accent
 
 
def analyze_syllable_shapes(pitch_contour, n_segments=8):
    """
    Divides the pitch contour into n_segments equal windows
    and classifies the pitch shape of each window.
    
    Returns list of shape labels like:
    ['flat', 'rising', 'rising_falling', 'falling', 'flat', ...]
    
    n_segments=8 approximates syllable-level analysis for a short mantra.
    For longer mantras you can increase this.
    """
    voiced = pitch_contour[pitch_contour > 0]
    if len(voiced) < n_segments:
        return []
 
    # Split voiced frames into n equal windows
    segments = np.array_split(voiced, n_segments)
    shapes   = [detect_pitch_shape(seg) for seg in segments]
    return shapes
 
 
def shape_sequence_similarity(ref_shapes, user_shapes):
    """
    Compares the sequence of pitch shapes between reference and user.
    Returns similarity score 0-100 and list of feedback messages.
    
    Example:
      ref:  ['flat', 'rising', 'rising_falling', 'falling']
      user: ['flat', 'flat',   'rising_falling', 'falling']
      → segment 2 is wrong: user has 'flat' where ref has 'rising' (Udatta)
    """
    if not ref_shapes or not user_shapes:
        return 0.0, []
 
    # Compare shape sequences using SequenceMatcher
    score    = SequenceMatcher(None, ref_shapes, user_shapes).ratio() * 100
    feedback = []
 
    # Find specific segments that differ
    matcher = SequenceMatcher(None, ref_shapes, user_shapes)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            ref_shape  = ref_shapes[i1]
            user_shape = user_shapes[j1] if j1 < len(user_shapes) else 'unknown'
 
            # Give specific Vedic accent feedback
            if ref_shape == 'rising' and user_shape != 'rising':
                feedback.append(
                    f"Syllable group {i1+1}: should have an Udatta (rising accent) "
                    f"but yours sounds {user_shape}"
                )
            elif ref_shape == 'rising_falling' and user_shape != 'rising_falling':
                feedback.append(
                    f"Syllable group {i1+1}: should have a Svarita "
                    f"(rising-then-falling circumflex) but yours sounds {user_shape}"
                )
            elif ref_shape == 'falling' and user_shape != 'falling':
                feedback.append(
                    f"Syllable group {i1+1}: should have an Anudatta "
                    f"(falling tone) but yours sounds {user_shape}"
                )
 
    return score, feedback
 
 
# Run syllable shape analysis
ref_shapes  = analyze_syllable_shapes(ref_pitch)
user_shapes = analyze_syllable_shapes(pitch)
 
shape_sim, shape_feedback = shape_sequence_similarity(ref_shapes, user_shapes)
 
print("Syllable Shape Similarity:", round(shape_sim, 2), "%")
print("Ref  shapes:", ref_shapes)
print("User shapes:", user_shapes)
 
 
# --- Pitch slope similarity ---
# The Beguš paper: independent svarita = same pitch targets but STEEPER slope
# So we also compare the rate of pitch change (velocity/slope)
 
def pitch_slope_similarity(ref_p, user_p):
    """
    Compares how fast pitch rises and falls.
    Captures steepness difference between udatta and independent svarita.
    """
    def get_slopes(p):
        voiced = p[p > 0]
        if len(voiced) < 2:
            return np.array([0.0])
        log_p = np.log2(voiced)
        return np.diff(log_p)   # differences between consecutive pitch values
 
    ref_slopes  = get_slopes(ref_p)
    user_slopes = get_slopes(user_p)
 
    if len(ref_slopes) == 0 or len(user_slopes) == 0:
        return 0.0
 
    D, wp = dtw(
        ref_slopes.reshape(1, -1),
        user_slopes.reshape(1, -1),
        metric='euclidean'
    )
    slope_dist = D[-1, -1] / len(wp)
    return 100 * np.exp(-slope_dist / 0.5)
 
slope_sim = pitch_slope_similarity(ref_pitch, pitch)
 
# --- Combined pitch score ---
# We now have 5 pitch measurements, each from a different research insight:
#
#   pitch_similarity  — DTW on z-scored pitch contour (overall shape)
#                       catches: melody wrong, wrong key relative movement
#
#   zone_sim          — Udatta/Anudatta/Svarita zone presence & levels
#                       catches: missing the rising/falling arc of the mantra
#                       (from Beguš + BHASHA papers)
#
#   slope_sim         — rate of pitch change (steepness)
#                       catches: svarita compressed vs expanded wrong
#                       (from Beguš 2016 — independent svarita = steeper udatta)
#
#   accent_sim        — frame-level Anudatta/Neutral/Udatta label sequence
#                       catches: Udatta placed on WRONG syllable
#                       (from BHASHA 2025 — wrong accent = wrong meaning)
#
#   shape_sim         — intra-syllable pitch shape (rising/falling/circumflex)
#                       catches: wrong shape on individual syllables
#                       (from Beguš 2016 — svarita = circumflex within syllable)
 
pitch_combined = (
    0.35 * pitch_similarity +   # overall contour shape (most important)
    0.25 * zone_sim         +   # Vedic accent zone accuracy
    0.15 * slope_sim        +   # pitch transition steepness
    0.15 * accent_sim       +   # frame-level accent label placement
    0.10 * shape_sim            # intra-syllable pitch shape
)
 

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
model = whisper.load_model("medium")

# Transcribe user audio
# Tip: initial_prompt guides Whisper toward Sanskrit mantra sounds
ref_mantra = reference["mantra_text"]
user_result = model.transcribe(
    "user_chant1.wav",
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
    0.20 * pitch_combined +   # pitch accuracy
    0.15 * tempo_similarity     # rhythm
)

# -------------------------
# Results
# -------------------------
print("\n========== RESULTS ==========")
print("Text Similarity  :", round(text_similarity, 2), "%")
print("Pitch Similarity :", round(pitch_combined, 2), "%")
print("Rhythm Similarity:", round(tempo_similarity, 2), "%")
print("Voice Similarity :", round(mfcc_similarity,  2), "%")
print("Overall Score    :", round(overall, 2), "%")
print("==============================")

feedback_report = generate_feedback({
    "text_similarity": text_similarity,
    "pitch_combined": pitch_combined,
    "tempo_similarity": tempo_similarity,
    "mfcc_similarity": mfcc_similarity,
    "overall": overall,
    "zone_feedback": zone_feedback,
    "shape_feedback": shape_feedback,
    "accent_sim": accent_sim,
    "slope_sim": slope_sim,
})

print("\n========== FEEDBACK REPORT ==========")
print(json.dumps(feedback_report, indent=2, ensure_ascii=False))
print("=====================================")