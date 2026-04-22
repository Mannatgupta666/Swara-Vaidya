from __future__ import annotations

from typing import Dict, Iterable, List


def _to_float(results: dict, keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in results and results[key] is not None:
            try:
                return float(results[key])
            except (TypeError, ValueError):
                return default
    return default


def _to_list(results: dict, keys: Iterable[str]) -> List[str]:
    for key in keys:
        if key in results and results[key] is not None:
            value = results[key]
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str) and value.strip():
                return [value.strip()]
    return []


def _clamp(score: float) -> float:
    return max(0.0, min(100.0, score))


def _band(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good but needs improvement"
    if score >= 50:
        return "Moderate issues"
    return "Significant issues"


def _band_sentence(score: float) -> str:
    label = _band(score)
    if label == "Excellent":
        return "Excellent performance."
    if label == "Good but needs improvement":
        return "Good performance with noticeable areas to refine."
    if label == "Moderate issues":
        return "Moderate issues are affecting consistency."
    return "Significant issues need focused correction."


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            output.append(item.strip())
    return output


def _pitch_quality_from_deep_feedback(zone_feedback: List[str], shape_feedback: List[str]) -> float:
    total_flags = len(zone_feedback) + len(shape_feedback)
    penalty = min(30.0, total_flags * 4.0)
    return 100.0 - penalty


def _build_summary(overall: float, categories: Dict[str, float]) -> str:
    weakest_name, weakest_score = min(categories.items(), key=lambda kv: kv[1])
    strongest_name, strongest_score = max(categories.items(), key=lambda kv: kv[1])

    line1 = f"Overall: {_band(overall)} ({overall:.1f}/100)."
    if weakest_name == strongest_name:
        line2 = f"{weakest_name.capitalize()} is the main focus area at {weakest_score:.1f}/100."
    else:
        line2 = (
            f"Strongest area: {strongest_name.capitalize()} ({strongest_score:.1f}/100); "
            f"focus next on {weakest_name.capitalize()} ({weakest_score:.1f}/100)."
        )
    return f"{line1} {line2}"


def generate_feedback(results: dict) -> dict:
    """
    Convert chant analysis metrics into structured user-friendly feedback.

    Expected input keys (aliases are supported):
      - text_similarity
      - pitch_combined
      - tempo_similarity
      - mfcc_similarity
      - overall / overall_score / overall score
      - zone_feedback (list[str])
      - shape_feedback (list[str])
      - accent_sim
      - slope_sim
    """
    text_similarity = _clamp(_to_float(results, ["text_similarity", "text", "text_score"]))
    pitch_combined = _clamp(_to_float(results, ["pitch_combined", "pitch_similarity", "pitch"]))
    tempo_similarity = _clamp(_to_float(results, ["tempo_similarity", "rhythm_similarity", "rhythm"]))
    mfcc_similarity = _clamp(_to_float(results, ["mfcc_similarity", "voice_similarity", "voice"]))
    accent_sim = _clamp(_to_float(results, ["accent_sim", "accent_similarity"], default=100.0))
    slope_sim = _clamp(_to_float(results, ["slope_sim", "slope_similarity"], default=100.0))

    overall_score = _to_float(results, ["overall_score", "overall", "overall score"], default=-1)
    if overall_score < 0:
        overall_score = (
            0.40 * text_similarity
            + 0.20 * pitch_combined
            + 0.15 * tempo_similarity
            + 0.25 * mfcc_similarity
        )
    overall_score = _clamp(overall_score)

    zone_feedback = _to_list(results, ["zone_feedback"])
    shape_feedback = _to_list(results, ["shape_feedback"])
    deep_pitch_quality = _pitch_quality_from_deep_feedback(zone_feedback, shape_feedback)

    text_category = 0.85 * text_similarity + 0.15 * accent_sim
    pitch_category = (
        0.55 * pitch_combined
        + 0.20 * accent_sim
        + 0.15 * slope_sim
        + 0.10 * deep_pitch_quality
    )
    rhythm_category = 0.85 * tempo_similarity + 0.15 * slope_sim
    voice_category = 0.80 * mfcc_similarity + 0.20 * text_similarity

    category_scores = {
        "text": _clamp(text_category),
        "pitch": _clamp(pitch_category),
        "rhythm": _clamp(rhythm_category),
        "voice": _clamp(voice_category),
    }

    categories = {
        "text": (
            f"{_band(category_scores['text'])}: "
            f"Pronunciation and syllable matching are at {text_similarity:.1f}/100. "
            f"{_band_sentence(category_scores['text'])}"
        ),
        "pitch": (
            f"{_band(category_scores['pitch'])}: "
            f"Pitch alignment is {pitch_combined:.1f}/100, accent match is {accent_sim:.1f}/100, "
            f"and transition control is {slope_sim:.1f}/100."
        ),
        "rhythm": (
            f"{_band(category_scores['rhythm'])}: "
            f"Rhythm similarity is {tempo_similarity:.1f}/100. "
            f"{_band_sentence(category_scores['rhythm'])}"
        ),
        "voice": (
            f"{_band(category_scores['voice'])}: "
            f"Voice/timbre similarity is {mfcc_similarity:.1f}/100. "
            f"{_band_sentence(category_scores['voice'])}"
        ),
    }

    issues: List[str] = []
    issues.extend(zone_feedback)
    issues.extend(shape_feedback)

    if accent_sim < 70:
        issues.append(
            "Accent placement differs from reference (Udatta/Anudatta positions need correction)."
        )
    if slope_sim < 70:
        issues.append(
            "Pitch transitions are not close to reference (rises/falls are too flat or too abrupt)."
        )
    if text_similarity < 70:
        issues.append("Text accuracy is low; several words or syllables differ from the reference.")
    if tempo_similarity < 70:
        issues.append("Rhythm pacing is inconsistent with the reference chant.")
    if mfcc_similarity < 60:
        issues.append("Voice quality/tone projection differs strongly from the reference sample.")

    suggestions: List[str] = []
    if text_similarity < 85:
        suggestions.append("Focus on correct syllable pronunciation.")
    if tempo_similarity < 85:
        suggestions.append("Maintain consistent rhythm and syllable spacing.")
    if mfcc_similarity < 85:
        suggestions.append("Improve breath support and keep tone steady across the chant.")
    if accent_sim < 70:
        suggestions.append("Practice accent placement: mark Udatta, Anudatta, and Svarita before chanting.")
    if slope_sim < 70:
        suggestions.append("Practice controlled pitch glides to improve rise-and-fall transitions.")

    zone_text = " ".join(zone_feedback).lower()
    if "udatta" in zone_text or any("udatta" in z.lower() for z in zone_feedback):
        suggestions.append("Improve rising pitch in Udatta sections.")
    if "svarita" in zone_text or any("svarita" in z.lower() for z in zone_feedback):
        suggestions.append("Shape the Svarita as a clear rise followed by a controlled fall.")
    if "anudatta" in zone_text or any("anudatta" in z.lower() for z in zone_feedback):
        suggestions.append("Keep Anudatta regions lower and more stable before the accent peak.")

    if not suggestions:
        suggestions.append("Continue regular practice and maintain this consistency across longer chants.")

    issues = _unique(issues)
    suggestions = _unique(suggestions)

    return {
        "overall_score": round(overall_score, 2),
        "summary": _build_summary(overall_score, category_scores),
        "categories": categories,
        "issues": issues,
        "suggestions": suggestions,
    }
