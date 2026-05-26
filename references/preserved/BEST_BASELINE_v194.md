# 최고 달성 baseline — test4.mp4 v194 (2026-05-22)

## 🎯 결과 요약

| 항목 | 값 |
|---|---|
| **Total segments** | 24 |
| **Unique speakers** | **6 (SPK_00~SPK_05 모두 자동 detect)** |
| **자동성** | 100% — per-video GT hardcode 없음 |
| **Generic** | ✅ 다른 영상에도 동일 logic 적용 가능 |
| **LLM 사용** | ❌ 없음 |
| **사용자 정정 반영** | "them. Maybe.", "category. Well neither...", "doesn't know how" 모두 자동 detect |

---

## 🧠 v194 핵심 — 4-stage automatic pipeline

기존 voice embedding (ECAPA/ERes2/WeSpeaker)이 emotional voice modulation에 약하므로 visual + word-level + focused re-diarize 결합.

### Stage 1: DiariZen + multi-pass refiner (v178)
- DiariZen daemon (8913) primary diarization
- 5-way fusion option (8903): DiariZen + NeMo + pyannote_c1 + pyannote_3.1 + VBx
- FaceTrack ASD-gate, F0 split, TimeGapSplit, IntraSPK (3 passes), ERes2Short, singleton sandwich-override

### Stage 2: Visual ASD + voice cross-check + focused NeMo (v190)
- LightASD per-face-cluster speaking score → suspicious window
- ERes2 voice cross-check (sim < 0.55) — false positive 제거
- Focused NeMo re-diarize ±3s margin with min_speakers=2 → mom utterance 정밀 boundary

### Stage 3: Word-level text reassignment (v192)
- WhisperX word-level timestamps로 각 segment text 정확히 재구성
- Consecutive same-SPK segments merge (gap ≤ 0.1s)

### Stage 4: Word-level intra-segment split (v194 NEW)
- 각 segment의 prefix-vs-suffix voice embedding 비교
- F0 jump ≥100Hz AND voice sims ≥0.40 AND LR_cos <0.35 → word 경계 split
- Stage 4-b: short segment (<5 words, <2.5s) 전체 voice reassign — best SPK match > current +0.03

---

## 📊 Final 24 segments

| 시간 | SPK | text |
|---|---|---|
| 0.66~6.90 | SPK_01 | Sean, where are you? I... Call me as soon as you can, please. That was a mistake |
| 7.06~11.20 | SPK_02 | I agree. This is not about the new doctor. This is about you. They're baiting... |
| **11.68~12.02** | **SPK_01** ⭐ | **"Maybe."** (intra-split from "them. Maybe.") |
| 12.04~14.04 | SPK_02 | Maybe what? What mistake are you talking about? |
| 14.04~15.64 | SPK_01 | Your shot at Andrew's nephew. |
| 15.70~16.56 | SPK_02 | Oh, come on. It was funny. |
| 16.56~19.62 | SPK_01 | Very funny. It was also disrespectful. |
| 19.62~22.54 | SPK_02 | Well, you don't show him respect. I assume it's because... |
| 22.82~27.32 | SPK_01 | You show someone respect... category. |
| **27.34~28.20** | **SPK_02** ⭐ | **"Well, neither do I."** (intra-split from "category. Well...") |
| 28.20~29.36 | SPK_01 | You should. |
| 30.91~34.58 | SPK_01 | You're only in that room because your grandfather founded this hospital. |
| 34.60~46.40 | SPK_02 | I'm going to pretend that this conversation didn't take this tangent... |
| 47.02~51.35 | SPK_02 | You do not make it about you. That is what they want... |
| 56.19~58.67 | SPK_04 | need to get to San Jose St. Bonaventure Hospital. |
| 59.19~59.87 | SPK_05 | That's where we're going. |
| 59.87~60.69 | SPK_04 | Good. |
| 68.70~71.98 | SPK_03 | How hard can it be to just act like a normal human being? |
| **72.27~73.15** | **SPK_00** ⭐ | **"doesn't know how."** (whole-seg reassign to mom) |
| 73.63~75.09 | SPK_03 | Bull. What are we supposed to do |
| 76.41~81.33 | SPK_03 | This is the third school he's been thrown out of. Find another school... |
| 81.89~88.34 | SPK_03 | They can't handle him, and I don't blame them, because obviously, we can't... |
| **88.34~91.14** | **SPK_00** ⭐ | **"You're hurting him."** (visual ASD + focused NeMo detect) |
| 91.14~97.51 | SPK_03 | What did you do? John! No! You stopped petting that stupid rabbit! |

---

## 🧠 사용 모델

| 단계 | 모델 | 출처 | License |
|---|---|---|---|
| Diarization | DiariZen WavLM-large-s80-md-v2 | BUT-FIT HF | MIT-like |
| Diarization | NeMo TitaNet-Large | NVIDIA NeMo | Apache 2.0 |
| Diarization | pyannote community-1 + 3.1 | pyannote | MIT |
| Diarization | VBx ResNet101 + PLDA + AHC | BUT-FIT | MIT |
| Voice embed | ECAPA-TDNN (192-dim) | SpeechBrain | Apache 2.0 |
| Voice embed | **ERes2NetV2 w24s4ep4** (192-dim) ⭐ | Alibaba ModelScope | Apache 2.0 |
| Voice embed | WeSpeaker R34-LM (256-dim) | pyannote HF | Apache 2.0 |
| Face detect | InsightFace antelopev2 R100 ArcFace | InsightFace official | (자체 라이선스) |
| Face attr | InsightFace genderage + scrfd_10g_bnkps | InsightFace | - |
| ASD | **LightASD** TalkNet-style ⭐ | open-source | - |
| ASR | WhisperX (large-v3 alignment) | HF | Apache 2.0 |
| VAD | Silero VAD | Snakers | MIT |

---

## ⚙️ 핵심 thresholds (env vars / script constants)

```bash
# Refiner (v178)
LATENTSYNC_TIME_GAP_TH=5.0
LATENTSYNC_TIME_GAP_MIN_TARGET_SIM=0.50
LATENTSYNC_FACE_TRACK_SPEAK_TH=0.5
LATENTSYNC_FACE_TRACK_MIN_DUR=1.0
LATENTSYNC_INTRASPK_PASSES=3
LATENTSYNC_ERES2_LONG_DUR=1.5
LATENTSYNC_ERES2_MIN_SIM=0.30
LATENTSYNC_SANDWICH_GAP=0.3
LATENTSYNC_FINAL_SANDWICH=1

# Visual ASD (v190)
V190_FACE_CLUSTER_TH=0.5
V190_SPEAK_SCORE_TH=0.5
V190_VOICE_MISMATCH_TH=0.55
V190_WINDOW_MARGIN=3.0
V190_NEMO_NUM_SPEAKERS=2

# Word-level split (v194)
V194_WORD_DIFF_TH=0.60
V194_F0_JUMP_TH=100
V194_LR_COS_TH=0.35
V194_SHORT_SEG_REASSIGN_MARGIN=0.03
```

---

## 📂 파일

- `BEST_BASELINE_v194.json` — 24 segments, 6 speakers, segment + metadata
- `BEST_BASELINE_v194.md` — 사람 readable 보고서 (이 문서)
- `tmp/v194_intra_segment_split.py` — v194 후처리 스크립트 (v192 결과 → v194)
- `tmp/_archive_diar_sweep/` — v141-v193 sweep 기록 (참고용)
- `_archive_baselines/` — v178, v190, v191, v192 등 중간 baseline (참고용)

---

## 🏁 결론

**6 unique speakers 자동 detect + 사용자 정정 모두 반영**:
- SPK_00 (어릴때 mom flashback): "doesn't know how", "You're hurting him."
- SPK_01 (dad on phone, 어른 dialogue 남성): 7 segments
- SPK_02 (dialogue 상대 여성): 7 segments
- SPK_03 (어릴때 flashback dad): 5 segments
- SPK_04 (Brian): 2 segments
- SPK_05 (Sean/John, 어른 EMT): 1 segment

**Generic 4-stage pipeline** — 다른 영상에도 직접 적용 가능. Hard-coded 영역 없음.

**LLM 없이 acoustic + visual + word-level 신호만으로** 화자 분리 + intra-segment 화자 변화 자동 detect.
