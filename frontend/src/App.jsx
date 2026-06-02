import { useState, useRef, useEffect } from "react";

const API_URL = "https://mannatgupta512-svara-vaidya.hf.space";

// ─── Helpers ──────────────────────────────────────────────────────────────────
const gradeColor = (score) => {
  if (score >= 85) return "#2d6a4f";
  if (score >= 70) return "#b5621a";
  if (score >= 50) return "#c2410c";
  return "#991b1b";
};

const gradeEmoji = (score) => {
  if (score >= 85) return "✓";
  if (score >= 70) return "◐";
  if (score >= 50) return "△";
  return "✗";
};

// ─── Score bar ────────────────────────────────────────────────────────────────
const ScoreBar = ({ label, score, sublabel }) => (
  <div style={{ marginBottom: 18 }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
      <span style={{ fontSize: 13, color: "#4a4540", fontFamily: "'Crimson Text', Georgia, serif", letterSpacing: "0.01em" }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 600, color: gradeColor(score), fontFamily: "monospace" }}>
        {score}% {gradeEmoji(score)}
      </span>
    </div>
    <div style={{ background: "#e8e2d9", borderRadius: 2, height: 4 }}>
      <div style={{
        width: `${Math.min(score, 100)}%`, height: 4, borderRadius: 2,
        background: gradeColor(score), transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)"
      }} />
    </div>
    {sublabel && <div style={{ fontSize: 11, color: "#9c9189", marginTop: 4 }}>{sublabel}</div>}
  </div>
);

// ─── Word svara card ──────────────────────────────────────────────────────────
const WordCard = ({ item }) => {
  const ok = item.accent_ok && item.shape_ok;
  return (
    <div style={{
      background: ok ? "#f0f7f4" : "#fdf4f4",
      border: `1px solid ${ok ? "#a8d5c2" : "#e8b4b4"}`,
      borderRadius: 4, padding: "14px 18px", marginBottom: 10
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: ok ? 0 : 12 }}>
        <span style={{
          background: ok ? "#2d6a4f" : "#991b1b",
          color: "#fff", borderRadius: 2, padding: "2px 10px",
          fontSize: 11, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase"
        }}>
          Word {item.word_num}
        </span>
        <span style={{ fontSize: 17, fontWeight: 700, color: "#1a1614", fontFamily: "'Crimson Text', Georgia, serif" }}>
          '{item.word}'
        </span>
        <span style={{ marginLeft: "auto", fontSize: 14, color: ok ? "#2d6a4f" : "#991b1b" }}>{ok ? "✓" : "✗"}</span>
      </div>
      {!item.accent_ok && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 12, color: "#6b5c52", marginBottom: 6, fontFamily: "'Crimson Text', Georgia, serif" }}>
            <span style={{ color: "#991b1b", fontWeight: 600 }}>Accent — </span>
            Expected <span style={{ background: "#e8f5ef", color: "#2d6a4f", padding: "1px 7px", borderRadius: 2, fontWeight: 600 }}>{item.ref_accent}</span>
            {"  "}·{"  "}
            Sung <span style={{ background: "#fef2f2", color: "#991b1b", padding: "1px 7px", borderRadius: 2, fontWeight: 600 }}>{item.user_accent}</span>
          </div>
          {item.accent_fix && (
            <div style={{ background: "#f8f6f0", borderLeft: "3px solid #b5621a", padding: "10px 14px", borderRadius: 2, fontSize: 12, color: "#4a3728", lineHeight: 1.7, fontFamily: "'Crimson Text', Georgia, serif" }}>
              {item.accent_fix}
            </div>
          )}
        </div>
      )}
      {!item.shape_ok && (
        <div>
          <div style={{ fontSize: 12, color: "#6b5c52", marginBottom: 6, fontFamily: "'Crimson Text', Georgia, serif" }}>
            <span style={{ color: "#991b1b", fontWeight: 600 }}>Shape — </span>
            Expected <span style={{ background: "#e8f5ef", color: "#2d6a4f", padding: "1px 7px", borderRadius: 2, fontWeight: 600 }}>{item.ref_shape}</span>
            {"  "}·{"  "}
            Sung <span style={{ background: "#fef2f2", color: "#991b1b", padding: "1px 7px", borderRadius: 2, fontWeight: 600 }}>{item.user_shape}</span>
          </div>
          {item.shape_fix && (
            <div style={{ background: "#f8f6f0", borderLeft: "3px solid #b5621a", padding: "10px 14px", borderRadius: 2, fontSize: 12, color: "#4a3728", lineHeight: 1.7, fontFamily: "'Crimson Text', Georgia, serif" }}>
              {item.shape_fix}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ─── Feedback list ────────────────────────────────────────────────────────────
const FeedbackList = ({ items, variant = "default" }) => (
  <>
    {items.map((item, i) => (
      <div key={i} style={{
        background: variant === "error" ? "#fdf4f4" : "#f8f6f0",
        borderLeft: `3px solid ${variant === "error" ? "#c9a0a0" : "#b5a88a"}`,
        padding: "10px 14px", borderRadius: 2,
        fontSize: 13, color: "#3d3028", lineHeight: 1.75, marginBottom: 8,
        fontFamily: "'Crimson Text', Georgia, serif"
      }}>
        {item}
      </div>
    ))}
  </>
);

// ─── Tab button ───────────────────────────────────────────────────────────────
const Tab = ({ label, active, onClick }) => (
  <button onClick={onClick} style={{
    padding: "7px 16px", borderRadius: 2,
    border: active ? "1px solid #8b7355" : "1px solid #ddd6cc",
    cursor: "pointer", fontSize: 12, fontWeight: 600,
    background: active ? "#8b7355" : "#faf8f5",
    color: active ? "#fff" : "#6b5c52",
    transition: "all 0.15s ease",
    letterSpacing: "0.04em",
    textTransform: "uppercase",
    fontFamily: "'Crimson Text', Georgia, serif"
  }}>
    {label}
  </button>
);

// ─── Divider ──────────────────────────────────────────────────────────────────
const Divider = ({ label }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 14, margin: "28px 0 20px" }}>
    <div style={{ flex: 1, height: 1, background: "#e0d8d0" }} />
    {label && <span style={{ fontSize: 10, color: "#b0a898", letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 600 }}>{label}</span>}
    <div style={{ flex: 1, height: 1, background: "#e0d8d0" }} />
  </div>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [recording,  setRecording]  = useState(false);
  const [audioBlob,  setAudioBlob]  = useState(null);
  const [audioURL,   setAudioURL]   = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);
  const [activeTab,  setActiveTab]  = useState("svara");
  const [apiStatus,  setApiStatus]  = useState("checking");

  const mediaRef     = useRef(null);
  const chunksRef    = useRef([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API_URL}/health`);
        setApiStatus(r.ok ? "ok" : "down");
      } catch { setApiStatus("down"); }
    };
    check();
    const interval = setInterval(check, 600000);
    return () => clearInterval(interval);
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        setAudioBlob(blob); setAudioURL(URL.createObjectURL(blob));
        stream.getTracks().forEach(t => t.stop());
        setResult(null); setError(null);
      };
      mr.start(); mediaRef.current = mr;
      setRecording(true); setResult(null); setError(null);
    } catch {
      setError("Microphone access denied. Please allow microphone permission in your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRef.current) { mediaRef.current.stop(); setRecording(false); }
  };

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setAudioBlob(file); setAudioURL(URL.createObjectURL(file));
    setResult(null); setError(null);
  };

  const analyze = async () => {
    if (!audioBlob) return;
    setLoading(true); setError(null);
    try {
      const form = new FormData();
      form.append("file", audioBlob, "user_chant.wav");
      const resp = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Server error" }));
        throw new Error(err.detail || "Analysis failed");
      }
      const data = await resp.json();
      setResult(data); setActiveTab("svara");
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const r  = result;
  const sc = r?.scores || {};
  const d  = r?.dimensions || {};
  const wordDetail = r?.svara_word_detail || [];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Cormorant+Garamond:wght@300;400;500;600&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: #f5f1eb; }
        audio { filter: none; color-scheme: light; }
        audio::-webkit-media-controls-panel { background: #c8b99a; }
        audio::-webkit-media-controls-current-time-display { color: #1a1614 !important; }
        audio::-webkit-media-controls-time-remaining-display { color: #1a1614 !important; }
        audio::-webkit-media-controls-timeline { background: #8b7355; }
        audio::-webkit-media-controls-play-button { filter: invert(0.8); }
        button:hover { opacity: 0.88; }
      `}</style>

      <div
        style={{
          minHeight: "100vh",
          backgroundImage: "url('/svarab.png')",
          backgroundSize: "cover",
          backgroundPosition: "center top",
          backgroundAttachment: "fixed",
          backgroundColor: "#f5f1eb",
          color: "#1a1614",
          fontFamily: "'Crimson Text', Georgia, serif",
          padding: "0 0 60px",
        }}
      >
        {/* ── Top bar ── */}
        <div
          style={{
            borderBottom: "1px solid #ddd6cc",
            background: "#faf8f5",
            padding: "12px 24px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span
            style={{
              fontSize: 11,
              color: "#9c9189",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              fontWeight: 600,
            }}
          >
            Svara Vaidya · स्वर वैद्य
          </span>
          <span
            style={{
              fontSize: 11,
              padding: "3px 10px",
              borderRadius: 2,
              background:
                apiStatus === "ok"
                  ? "#e8f5ef"
                  : apiStatus === "down"
                    ? "#fef2f2"
                    : "#f5f1eb",
              color:
                apiStatus === "ok"
                  ? "#2d6a4f"
                  : apiStatus === "down"
                    ? "#991b1b"
                    : "#9c9189",
              border: `1px solid ${apiStatus === "ok" ? "#a8d5c2" : apiStatus === "down" ? "#e8b4b4" : "#ddd6cc"}`,
              letterSpacing: "0.06em",
            }}
          >
            {apiStatus === "ok"
              ? "● Connected"
              : apiStatus === "down"
                ? "● Offline"
                : "● Checking"}
          </span>
        </div>

        <div style={{ maxWidth: 680, margin: "0 auto", padding: "0 20px" }}>
          {/* ── Header ── */}
          <div
            style={{
              textAlign: "center",
              padding: "56px 0 40px",
              borderBottom: "1px solid #ddd6cc",
            }}
          >
            <div
              style={{
                fontSize: 13,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "#9c9189",
                fontWeight: 600,
                marginBottom: 16,
              }}
            >
              Vedic Mantra Chanting Analyzer
            </div>
            <h1
              style={{
                fontFamily: "'Cormorant Garamond', Georgia, serif",
                fontSize: 52,
                fontWeight: 300,
                margin: "0 0 8px",
                color: "#1a1614",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              Svara Vaidya
            </h1>
            <p
              style={{
                fontFamily: "'Cormorant Garamond', Georgia, serif",
                fontSize: 20,
                color: "#6b5c52",
                margin: 0,
                fontWeight: 400,
                fontStyle: "italic",
              }}
            >
              स्वर वैद्य — The Voice Healer
            </p>
          </div>

          {/* ── Record / Upload card ── */}
          <div style={{ padding: "36px 0 0" }}>
            <div
              style={{
                fontSize: 11,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "#9c9189",
                fontWeight: 600,
                marginBottom: 20,
              }}
            >
              01 · Record or Upload
            </div>

            <div
              style={{
                display: "flex",
                gap: 10,
                flexWrap: "wrap",
                marginBottom: 20,
                justifyContent: "center",
              }}
            >
              {!recording ? (
                <button
                  onClick={startRecording}
                  disabled={loading}
                  style={{
                    background: loading ? "#e8e2d9" : "#1a1614",
                    color: loading ? "#9c9189" : "#faf8f5",
                    border: "none",
                    borderRadius: 2,
                    padding: "12px 28px",
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: loading ? "not-allowed" : "pointer",
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    fontFamily: "'Crimson Text', Georgia, serif",
                  }}
                >
                  ● Record
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  style={{
                    background: "#faf8f5",
                    color: "#991b1b",
                    border: "2px solid #991b1b",
                    borderRadius: 2,
                    padding: "12px 28px",
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: "pointer",
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    fontFamily: "'Crimson Text', Georgia, serif",
                  }}
                >
                  ■ Stop
                </button>
              )}

              <button
                onClick={() => fileInputRef.current.click()}
                disabled={loading}
                style={{
                  background: "#faf8f5",
                  color: "#4a4540",
                  border: "1px solid #c8c0b4",
                  borderRadius: 2,
                  padding: "12px 28px",
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.5 : 1,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  fontFamily: "'Crimson Text', Georgia, serif",
                }}
              >
                Upload Audio
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.m4a,.ogg,.flac"
                style={{ display: "none" }}
                onChange={handleFile}
              />
            </div>

            {recording && (
              <div
                style={{
                  textAlign: "center",
                  color: "#991b1b",
                  fontSize: 13,
                  marginBottom: 16,
                  letterSpacing: "0.04em",
                }}
              >
                Recording in progress — chant your mantra, then press Stop
              </div>
            )}

            {audioURL && (
              <audio
                controls
                src={audioURL}
                style={{
                  width: "100%",
                  borderRadius: 2,
                  marginBottom: 18,
                  border: "1px solid #8b7355",
                  backgroundColor: "#c8b99a",
                  colorScheme: "light",
                }}
              />
            )}

            {audioBlob && !loading && !recording && (
              <button
                onClick={analyze}
                style={{
                  background: "#8b7355",
                  color: "#faf8f5",
                  border: "none",
                  borderRadius: 2,
                  padding: "15px 0",
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: "pointer",
                  width: "100%",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  fontFamily: "'Crimson Text', Georgia, serif",
                }}
              >
                Analyse Chant
              </button>
            )}

            {loading && (
              <div
                style={{
                  background: "#faf8f5",
                  border: "1px solid #ddd6cc",
                  borderRadius: 2,
                  padding: "28px 24px",
                  textAlign: "center",
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    color: "#6b5c52",
                    marginBottom: 6,
                    letterSpacing: "0.04em",
                  }}
                >
                  Analysing your chant...
                </div>
                <div style={{ fontSize: 12, color: "#9c9189" }}>
                  MMS acoustic embedding + pitch analysis — 15 to 30 seconds
                </div>
              </div>
            )}

            {error && (
              <div
                style={{
                  background: "#fdf4f4",
                  border: "1px solid #e8b4b4",
                  borderRadius: 2,
                  padding: 14,
                  color: "#7f1d1d",
                  fontSize: 13,
                  marginTop: 14,
                  fontFamily: "'Crimson Text', Georgia, serif",
                }}
              >
                {error}
              </div>
            )}
          </div>

          {/* ── Results ── */}
          {r && (
            <>
              <Divider label="Analysis Results" />

              {/* Overall score */}
              <div
                style={{
                  background: "#faf8f5",
                  border: "1px solid #ddd6cc",
                  borderRadius: 2,
                  padding: "28px 28px 24px",
                  marginBottom: 16,
                }}
              >
                <div style={{ textAlign: "center" }}>
                  <div
                    style={{
                      fontSize: 11,
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
                      color: "#9c9189",
                      fontWeight: 600,
                      marginBottom: 8,
                    }}
                  >
                    Resonance Score
                  </div>
                  <div
                    style={{
                      fontFamily: "'Cormorant Garamond', Georgia, serif",
                      fontSize: 64,
                      fontWeight: 300,
                      lineHeight: 1,
                      color: gradeColor(r.overall_score || sc.overall || 0),
                    }}
                  >
                    {r.overall_score || sc.overall || 0}
                    <span style={{ fontSize: 28 }}>%</span>
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <span
                      style={{
                        fontFamily: "'Cormorant Garamond', Georgia, serif",
                        fontSize: 22,
                        fontWeight: 500,
                        color: "#1a1614",
                      }}
                    >
                      {r.overall_grade}
                    </span>
                    <span
                      style={{
                        fontSize: 13,
                        color: "#9c9189",
                        marginLeft: 10,
                        fontStyle: "italic",
                      }}
                    >
                      {r.overall_grade === "Uttama"
                        ? "Excellent — high acoustic fidelity"
                        : r.overall_grade === "Madhyama"
                          ? "Good — minor errors present"
                          : r.overall_grade === "Adi"
                            ? "Foundation level — practice needed"
                            : "Beginning stage"}
                    </span>
                  </div>
                </div>

                {r.user_transcript && (
                  <div
                    style={{
                      marginTop: 18,
                      paddingTop: 18,
                      borderTop: "1px solid #e8e2d9",
                    }}
                  >
                    <span
                      style={{
                        fontSize: 11,
                        letterSpacing: "0.1em",
                        textTransform: "uppercase",
                        color: "#9c9189",
                        fontWeight: 600,
                      }}
                    >
                      Mantra —{" "}
                    </span>
                    <span
                      style={{
                        fontSize: 14,
                        color: "#4a4540",
                        fontStyle: "italic",
                      }}
                    >
                      {r.mantra}
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        color: "#b0a898",
                        margin: "0 8px",
                      }}
                    >
                      ·
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        letterSpacing: "0.1em",
                        textTransform: "uppercase",
                        color: "#9c9189",
                        fontWeight: 600,
                      }}
                    >
                      Heard —{" "}
                    </span>
                    <span
                      style={{
                        fontSize: 14,
                        color: "#4a4540",
                        fontStyle: "italic",
                      }}
                    >
                      "{r.user_transcript}"
                    </span>
                  </div>
                )}
              </div>

              {/* Dimension scores */}
              <div
                style={{
                  background: "#faf8f5",
                  border: "1px solid #ddd6cc",
                  borderRadius: 2,
                  padding: "24px 28px",
                  marginBottom: 16,
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "#9c9189",
                    fontWeight: 600,
                    marginBottom: 20,
                  }}
                >
                  Dimension Scores
                </div>
                <ScoreBar
                  label="Ucchāraṇa — Pronunciation"
                  score={d.uccharana?.score ?? sc.text_similarity ?? 0}
                />
                <ScoreBar
                  label="Nāda — Voice Quality"
                  score={d.nada?.score ?? sc.mfcc_similarity ?? 0}
                />
                <ScoreBar
                  label="Svara — Pitch & Accent"
                  score={d.svara?.score ?? sc.pitch_combined ?? 0}
                />
                <div
                  style={{
                    paddingLeft: 16,
                    borderLeft: "1px solid #e8e2d9",
                    marginBottom: 18,
                  }}
                >
                  <ScoreBar
                    label="↳ Contour shape"
                    score={
                      sc.pitch_breakdown?.contour_shape ?? d.svara?.score ?? 0
                    }
                  />
                  <ScoreBar
                    label="↳ Zone accuracy"
                    score={sc.pitch_breakdown?.zone_accuracy ?? 0}
                  />
                  <ScoreBar
                    label="↳ Accent placement"
                    score={sc.pitch_breakdown?.accent_placement ?? 0}
                  />
                  <ScoreBar
                    label="↳ Syllable shape"
                    score={sc.pitch_breakdown?.syllable_shape ?? 0}
                  />
                  <ScoreBar
                    label="↳ Slope / steepness"
                    score={sc.pitch_breakdown?.slope_steepness ?? 0}
                  />
                </div>
                <ScoreBar
                  label="Laya — Rhythm & Tempo"
                  score={d.laya?.score ?? sc.tempo_similarity ?? 0}
                />
              </div>

              {/* Priority focus areas */}
              {r.priority && r.priority.length > 0 && (
                <div
                  style={{
                    background: "#faf8f5",
                    border: "1px solid #ddd6cc",
                    borderRadius: 2,
                    padding: "24px 28px",
                    marginBottom: 16,
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
                      color: "#9c9189",
                      fontWeight: 600,
                      marginBottom: 18,
                    }}
                  >
                    Focus Areas — prioritise these
                  </div>
                  {r.priority.map((p, i) => (
                    <div
                      key={i}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "12px 0",
                        borderBottom:
                          i < r.priority.length - 1
                            ? "1px solid #ece7e0"
                            : "none",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 12,
                        }}
                      >
                        <span
                          style={{
                            color:
                              i === 0
                                ? "#991b1b"
                                : i === 1
                                  ? "#b5621a"
                                  : "#9c9189",
                            fontSize: 11,
                            fontWeight: 700,
                            width: 16,
                            textAlign: "center",
                          }}
                        >
                          {i + 1}
                        </span>
                        <span style={{ fontSize: 14, color: "#3d3028" }}>
                          {p.dimension}
                        </span>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div
                          style={{
                            fontSize: 14,
                            fontWeight: 600,
                            color: gradeColor(p.score),
                            fontFamily: "monospace",
                          }}
                        >
                          {p.score}%
                        </div>
                        <div style={{ fontSize: 11, color: "#b0a898" }}>
                          impact {p.impact}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Praises */}
              {r.all_praises && r.all_praises.length > 0 && (
                <div
                  style={{
                    background: "#f0f7f4",
                    border: "1px solid #a8d5c2",
                    borderRadius: 2,
                    padding: "20px 24px",
                    marginBottom: 16,
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
                      color: "#2d6a4f",
                      fontWeight: 600,
                      marginBottom: 14,
                    }}
                  >
                    Sādhu — what you did well
                  </div>
                  {r.all_praises.map((p, i) => (
                    <div
                      key={i}
                      style={{
                        fontSize: 13,
                        color: "#1a4a35",
                        marginBottom: 7,
                        fontFamily: "'Crimson Text', Georgia, serif",
                        lineHeight: 1.6,
                      }}
                    >
                      ✓ {p}
                    </div>
                  ))}
                </div>
              )}

              {/* Detail tabs */}
              <div
                style={{
                  background: "#faf8f5",
                  border: "1px solid #ddd6cc",
                  borderRadius: 2,
                  padding: "24px 28px",
                  marginBottom: 16,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    gap: 8,
                    marginBottom: 22,
                    flexWrap: "wrap",
                  }}
                >
                  <Tab
                    label="Svara"
                    active={activeTab === "svara"}
                    onClick={() => setActiveTab("svara")}
                  />
                  <Tab
                    label="Ucchāraṇa"
                    active={activeTab === "uccharana"}
                    onClick={() => setActiveTab("uccharana")}
                  />
                  <Tab
                    label="Laya"
                    active={activeTab === "laya"}
                    onClick={() => setActiveTab("laya")}
                  />
                  <Tab
                    label="Nāda"
                    active={activeTab === "nada"}
                    onClick={() => setActiveTab("nada")}
                  />
                </div>

                {activeTab === "svara" && (
                  <div>
                    {d.svara?.issues && d.svara.issues.length > 0 && (
                      <div style={{ marginBottom: 18 }}>
                        <div
                          style={{
                            fontSize: 11,
                            letterSpacing: "0.1em",
                            textTransform: "uppercase",
                            color: "#9c9189",
                            fontWeight: 600,
                            marginBottom: 10,
                          }}
                        >
                          Summary
                        </div>
                        <FeedbackList items={d.svara.issues} />
                      </div>
                    )}
                    <div
                      style={{
                        fontSize: 11,
                        letterSpacing: "0.1em",
                        textTransform: "uppercase",
                        color: "#9c9189",
                        fontWeight: 600,
                        marginBottom: 12,
                      }}
                    >
                      Word-by-word breakdown
                    </div>
                    {wordDetail.length === 0 ? (
                      <div
                        style={{
                          background: "#f0f7f4",
                          border: "1px solid #a8d5c2",
                          borderRadius: 2,
                          padding: 14,
                          color: "#2d6a4f",
                          fontSize: 14,
                          fontFamily: "'Crimson Text', Georgia, serif",
                        }}
                      >
                        All words carry correct svara accent and shape.
                      </div>
                    ) : (
                      wordDetail.map((item, i) => (
                        <WordCard key={i} item={item} />
                      ))
                    )}
                    {d.svara?.fixes && d.svara.fixes.length > 0 && (
                      <div style={{ marginTop: 18 }}>
                        <div
                          style={{
                            fontSize: 11,
                            letterSpacing: "0.1em",
                            textTransform: "uppercase",
                            color: "#9c9189",
                            fontWeight: 600,
                            marginBottom: 10,
                          }}
                        >
                          Practice techniques
                        </div>
                        <FeedbackList items={d.svara.fixes} />
                      </div>
                    )}
                  </div>
                )}

                {activeTab === "uccharana" && (
                  <div>
                    <div
                      style={{
                        display: "flex",
                        gap: 10,
                        marginBottom: 18,
                        flexWrap: "wrap",
                      }}
                    >
                      {[
                        ["Char", sc.text_similarity],
                        ["Word", d.uccharana?.word_sim],
                        ["Syllable", d.uccharana?.syl_sim],
                      ].map(
                        ([label, val]) =>
                          val != null && (
                            <div
                              key={label}
                              style={{
                                background: "#f5f1eb",
                                border: "1px solid #ddd6cc",
                                borderRadius: 2,
                                padding: "14px 18px",
                                flex: 1,
                                minWidth: 80,
                              }}
                            >
                              <div
                                style={{
                                  fontSize: 11,
                                  color: "#9c9189",
                                  letterSpacing: "0.08em",
                                  textTransform: "uppercase",
                                  marginBottom: 4,
                                }}
                              >
                                {label}
                              </div>
                              <div
                                style={{
                                  fontSize: 24,
                                  fontWeight: 600,
                                  color: gradeColor(val),
                                  fontFamily: "monospace",
                                }}
                              >
                                {Math.round(val)}%
                              </div>
                            </div>
                          ),
                      )}
                    </div>
                    {d.uccharana?.issues && d.uccharana.issues.length > 0 ? (
                      <div style={{ marginBottom: 14 }}>
                        <div
                          style={{
                            fontSize: 11,
                            letterSpacing: "0.1em",
                            textTransform: "uppercase",
                            color: "#9c9189",
                            fontWeight: 600,
                            marginBottom: 10,
                          }}
                        >
                          Issues
                        </div>
                        <FeedbackList
                          items={d.uccharana.issues}
                          variant="error"
                        />
                      </div>
                    ) : (
                      <div
                        style={{
                          color: "#2d6a4f",
                          fontSize: 14,
                          marginBottom: 14,
                          fontFamily: "'Crimson Text', Georgia, serif",
                        }}
                      >
                        No pronunciation issues detected.
                      </div>
                    )}
                    {d.uccharana?.fixes && d.uccharana.fixes.length > 0 && (
                      <div>
                        <div
                          style={{
                            fontSize: 11,
                            letterSpacing: "0.1em",
                            textTransform: "uppercase",
                            color: "#9c9189",
                            fontWeight: 600,
                            marginBottom: 10,
                          }}
                        >
                          How to improve
                        </div>
                        <FeedbackList items={d.uccharana.fixes} />
                      </div>
                    )}
                  </div>
                )}

                {activeTab === "laya" && (
                  <div>
                    <div style={{ display: "flex", gap: 10, marginBottom: 18 }}>
                      <div
                        style={{
                          background: "#f5f1eb",
                          border: "1px solid #ddd6cc",
                          borderRadius: 2,
                          padding: "18px 22px",
                          flex: 1,
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            color: "#9c9189",
                            letterSpacing: "0.08em",
                            textTransform: "uppercase",
                            marginBottom: 6,
                          }}
                        >
                          Rhythm Score
                        </div>
                        <div
                          style={{
                            fontFamily: "'Cormorant Garamond', Georgia, serif",
                            fontSize: 40,
                            fontWeight: 400,
                            color: gradeColor(d.laya?.score ?? 0),
                          }}
                        >
                          {d.laya?.score ?? 0}%
                        </div>
                      </div>
                      {d.laya?.duration_ratio != null && (
                        <div
                          style={{
                            background: "#f5f1eb",
                            border: "1px solid #ddd6cc",
                            borderRadius: 2,
                            padding: "18px 22px",
                            flex: 1,
                            textAlign: "center",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 11,
                              color: "#9c9189",
                              letterSpacing: "0.08em",
                              textTransform: "uppercase",
                              marginBottom: 6,
                            }}
                          >
                            Duration ratio
                          </div>
                          <div
                            style={{
                              fontFamily:
                                "'Cormorant Garamond', Georgia, serif",
                              fontSize: 40,
                              fontWeight: 400,
                              color:
                                Math.abs(d.laya.duration_ratio - 1) < 0.15
                                  ? "#2d6a4f"
                                  : "#b5621a",
                            }}
                          >
                            {d.laya.duration_ratio}×
                          </div>
                          <div
                            style={{
                              fontSize: 12,
                              color: "#9c9189",
                              marginTop: 4,
                            }}
                          >
                            {d.laya.duration_ratio > 1.15
                              ? "Too slow"
                              : d.laya.duration_ratio < 0.85
                                ? "Too fast"
                                : "Good pace"}
                          </div>
                        </div>
                      )}
                    </div>
                    {d.laya?.issues && d.laya.issues.length > 0 && (
                      <FeedbackList items={d.laya.issues} variant="error" />
                    )}
                    {d.laya?.fixes && d.laya.fixes.length > 0 && (
                      <FeedbackList items={d.laya.fixes} />
                    )}
                    {(!d.laya?.issues || d.laya.issues.length === 0) && (
                      <div
                        style={{
                          color: "#2d6a4f",
                          fontSize: 14,
                          fontFamily: "'Crimson Text', Georgia, serif",
                        }}
                      >
                        Laya is steady — good rhythmic flow.
                      </div>
                    )}
                  </div>
                )}

                {activeTab === "nada" && (
                  <div>
                    <div
                      style={{
                        background: "#f5f1eb",
                        border: "1px solid #ddd6cc",
                        borderRadius: 2,
                        padding: "20px 24px",
                        textAlign: "center",
                        marginBottom: 18,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 11,
                          color: "#9c9189",
                          letterSpacing: "0.08em",
                          textTransform: "uppercase",
                          marginBottom: 6,
                        }}
                      >
                        Voice Quality Score
                      </div>
                      <div
                        style={{
                          fontFamily: "'Cormorant Garamond', Georgia, serif",
                          fontSize: 56,
                          fontWeight: 300,
                          color: gradeColor(d.nada?.score ?? 0),
                        }}
                      >
                        {d.nada?.score ?? 0}%
                      </div>
                    </div>
                    {d.nada?.issues && d.nada.issues.length > 0 ? (
                      <FeedbackList items={d.nada.issues} variant="error" />
                    ) : (
                      <div
                        style={{
                          color: "#2d6a4f",
                          fontSize: 14,
                          marginBottom: 14,
                          fontFamily: "'Crimson Text', Georgia, serif",
                        }}
                      >
                        Nāda quality is authentic.
                      </div>
                    )}
                    {d.nada?.fixes && d.nada.fixes.length > 0 && (
                      <FeedbackList items={d.nada.fixes} />
                    )}
                    {d.nada?.praises && d.nada.praises.length > 0 && (
                      <div style={{ marginTop: 12 }}>
                        {d.nada.praises.map((p, i) => (
                          <div
                            key={i}
                            style={{
                              fontSize: 13,
                              color: "#2d6a4f",
                              fontFamily: "'Crimson Text', Georgia, serif",
                            }}
                          >
                            ✓ {p}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Sadhana tip */}
              <div
                style={{
                  background: "#faf8f5",
                  border: "1px solid #c8c0b4",
                  borderRadius: 2,
                  padding: "24px 28px",
                  marginBottom: 32,
                  borderLeft: "3px solid #8b7355",
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "#8b7355",
                    fontWeight: 600,
                    marginBottom: 12,
                  }}
                >
                  Daily Sādhana Recommendation
                </div>
                <p
                  style={{
                    fontFamily: "'Crimson Text', Georgia, serif",
                    fontSize: 15,
                    color: "#3d3028",
                    lineHeight: 1.8,
                    margin: 0,
                  }}
                >
                  {r.sadhana_tip}
                </p>
              </div>
            </>
          )}

          {/* Footer */}
          <div
            style={{
              textAlign: "center",
              color: "#c0b8ae",
              fontSize: 13,
              paddingTop: 20,
              borderTop: "1px solid #e8e2d9",
              fontStyle: "italic",
            }}
          >
            ॐ शान्तिः शान्तिः शान्तिः
          </div>
        </div>
      </div>
    </>
  );
}
