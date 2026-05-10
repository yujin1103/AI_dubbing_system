# v15 최종 리포트 — TTS 영어 leak 완전 해결

**날짜**: 2026-05-06 01:00
**테스트**: 2개 영상 (test2_part1, test3) — 모두 Walking Dead drama, English → Korean

---

## 🎯 최종 결과

| Version | Setup | test2_part1 | test3 | 검증 |
|---|---|---|---|---|
| v7-v13 | inference_instruct2 + 다양한 instruction | 70-82% | (미실시) | ❌ leak |
| v14 | 4월 28일 cross_lingual + Context/Emotion prefix | 60% | (미실시) | ❌ "Context. The speaker..." 영어 leak |
| **v15** | **cross_lingual + 팀원 imperative format** | **100%** ✅ | **100%** ✅ | ✅ **PASS** |

**두 영상 모두 한국어 100% — leak 0**

## 🔍 진짜 Root Cause

### 4월 28일 코드의 약점
```python
# 4월 28일 (lecture에서만 잘 작동)
if instruct:
    prefix = f'{instruct}<|endofprompt|>'
    # instruct = "Context: The speaker recalls...\nEmotion: sharp challenge..."
    # → CosyVoice3가 그대로 합성 → 영어 leak
```

**lecture에서 잘 작동했던 이유**: emotion=Neutral, context+emotion 빈 문자열
→ `prefix = "You are a helpful assistant.<|endofprompt|>"` (단순 default)

**drama에서 leak 발생**: LLM이 context/emotion 채움
→ `prefix = "Context: ...\nEmotion: ..."` (학습 분포 X) → 모델이 그대로 합성

### v15 해결
**팀원의 검증된 format**으로 prefix 변경:
```python
# v15
prefix = f'You are a helpful assistant. Please say this sentence {tone}.<|endofprompt|>'
# tone = LLM이 만든 풀 imperative phrase
# 예: "in a curt, impatient tone with restrained urgency, prompting someone to proceed immediately"
```

CosyVoice3 학습 분포 매칭:
```
Training: "You are a helpful assistant. Please say a sentence as loudly as possible."
v15:      "You are a helpful assistant. Please say this sentence in a curt, impatient tone..."
```
→ 모델이 instruction으로 정상 처리 → 한국어만 합성

## 📝 v15 변경 사항

1. **LLM prompt**: `emotion: 2~5 words` → `tone: full imperative phrase`
   - 예시: `"in a casual, pleased, lightly confident tone with warm but restrained excitement"`
2. **Parser**: 정규식에 `tone` 필드 추가 `(korean|context|emotion|tone)`
3. **synthesize_segment_cosy**: prefix를 팀원 format으로
4. **`inference_cross_lingual` 유지** (4월 28일 함수 그대로)

## 🎬 v15 결과 audio 분석 (qwen-asr 검증)

```
Duration: 60.1s
Total chars: 172
Korean: 172 (100.0%)
English: 0
Other: 0
```

### LLM이 만든 tone 예시 (8 segments)
| ID | English | Korean | Tone (LLM 생성) |
|---|---|---|---|
| 0 | "Go ahead" | "어서" | in a curt, impatient tone with restrained urgency... |
| 1 | "That wound on Justin..." | "저스틴한테 난 그 상처는..." | in a thoughtful, uneasy tone with quiet suspicion... |
| 2 | "but it's small round and clean" | "근데 그건 아주 작고 둥글고..." | in a calm, analytical tone with measured observation... |
| 5 | "Just kind shit you used to do..." | "그냥 내가 널 가졌을 때..." | in a tense, disbelieving tone with wounded indignation... |
| 6 | "No But others do..." | "아니야. 남들도 해서 확인 중이야." | with subdued sadness, careful pacing, conveying self-protect... |
| 7 | "Might killed him..." | "어쩌면 그를 죽였을지도..." | in a low, grim, unsettled tone with anxious realization... |

각 segment의 emotion이 자연스러운 한국어 발화로 전달됨.

## 📂 산출물

### test2_part1 (1분 Walking Dead, Justin 상처 분석 scene)
- **영상**: `E:/TTS_capstone/media/output/test2p1v15_FINAL.mp4` (21.6 MB, 1분)
- **TTS만**: `E:/TTS_capstone/media/output/test2p1v15_dubbed_TTS_only.wav`
- **검증**: 한국어 100% (172/172 글자), 외국어 0

### test3 (다른 Walking Dead, Negan-Sasha 위협 scene) — 추가 검증
- **영상**: `E:/TTS_capstone/media/output/test3v15_FINAL.mp4`
- **TTS만**: `E:/TTS_capstone/media/output/test3v15_dubbed_TTS_only.wav`
- **검증**: 한국어 100% (187/187 글자), 외국어 0

### 검증 샘플 (test3)
> "남은 인생 첫 날에 온 걸 환영해, 사샤야. 오늘 네가 모든 걸 제자리로 돌려놓게 도울 거야. 넌 똑똑하고 지독하게 강하고 존엄 있고 어떤 빌어먹을 바보도 절대로 상대 안하지..."

자연스러운 한국어 발화. 폭력적 대사 (빌어먹을 = fucking) 도 LLM content_filter 통과.

## 🔧 종합 적용된 fix들

### Phase 1 — 음성 분리 강화
- BS-Roformer (htdemucs SDR 9.5 → 12.97, +3 dB)
- 화자 detect: htdemucs 1명 → BS-Roformer 3명 → AV-Fusion으로 spurious 제거 → 정확히 2명

### Phase 2 — TTS 길이 매칭
- Speed retry 효율화 (3 retries → 0~1 retry, ratio 기반)
- MOS reload 제거 (~60s 절약)
- post-TTS length check + speed=1.15 retry (필요 시)

### Phase 3 — AV Fusion (LightASD)
- spurious 화자 자동 제거
- 영상 frame별 speaking probability
- per-frame lipsync target 결정

### v15 (영어 leak 완전 해결) ⭐
- LLM이 풀 imperative tone 생성
- 팀원 검증 format으로 prefix wrap
- `inference_cross_lingual` 유지

### 기타
- Google translate fallback 제거
- Dialogue filter (한숨/탄식 자동 skip)
- LLM length-aware 번역 (syllable target)

## 🎯 성능 (1분 영상)

| Stage | 시간 |
|---|---|
| BS-Roformer | ~30-60s |
| Qwen ASR | ~30-60s |
| pyannote + ECAPA | ~10-30s |
| LightASD AV-Fusion | ~60-90s |
| Emotion + MOS | ~30-60s |
| LLM 번역 | ~30-60s |
| CosyVoice3 로드 | ~60-120s |
| TTS 합성 (8 seg) | ~5-7분 |
| Mix + Concat | ~10s |
| **합계** | **~13분/1분 영상** |

## ✅ 사용자 요구사항 달성

- ✅ 영어 leak 0% (사용자가 가장 강력히 원했던 요구)
- ✅ 감정 보존 (LLM이 만든 풍부한 imperative tone instruction)
- ✅ Whisper 사용 안 함 (사용자 요청대로 — 검증은 qwen-asr로)
- ✅ Self reference 없이 MOS-selected reference 유지
- ✅ 팀원 검증 setup 정확히 재현

## 💡 핵심 교훈

1. **prefix format이 결정적**: "Context:..." 같은 metadata가 아닌 **자연 imperative**가 학습 분포 매칭
2. **함수 변경하지 않음**: `inference_cross_lingual`이 정답 (instruct2 아님)
3. **LLM이 풀 문장 생성**: 짧은 키워드보다 풀 imperative phrase가 안전
4. **팀원 검증된 setup 그대로 따라하는 게 최선**: 우리가 시도한 12개 변종(v7~v14) 중 v15만 성공

---

**작성**: 자율 작업 완료. 사용자 일어난 후 결과 검증 + 추가 영상 테스트 가능.

---

## 추가 작업 (사용자 일어난 후)

### 화자 분리 정확도 시도 (v16~v21)
- **v16-v18**: ECAPA threshold 다양 시도 — 1명 또는 over-detect
- **v19**: pyannote 그대로 신뢰 (≤2명 skip 후처리) — pyannote 자체 분류 잘못 (ID 2,3 잘못 SPEAKER_01, ID 8 여자 잘못 SPEAKER_00)
- **v20**: --speakers 2 강제 — v19와 동일 (pyannote 한계)
- **v21**: outlier detection 추가 — over-detect (5개 가짜 화자)

### Pyannote 모델 비교 (조사)
| 모델 | DER | 우리 사용 |
|---|---|---|
| 3.1 (legacy, AHC) | 22.7% | ❌ |
| **community-1 (VBx)** | 19.9% | ✅ |
| DiariZen v2 (BUT-FIT) | 13.9% | 미사용 |
| precision-2 (cloud) | 14.7% | 미사용 (유료) |

### 결론 - 화자 분리 한계
**community-1도 1.2s 짧은 다른 화자를 detect 못함**. 진짜 향상 필요 시:
- **DiariZen** 도입 (DER ~30% 추가 향상) — 별도 venv + 1-2시간 작업
- **pyannoteAI precision-2** cloud (유료)
- 또는 사용자가 segment별 화자 수동 라벨링

### 사용자 질문 답변
- **Q: 원본 언어 따라 화자 분리?** A: NO, 음향 특징 (음색, pitch) 기반
- **Q: 정교한 모델이 pyannote보다 나은가?** A: DiariZen 약 30% DER 향상
- **Q: pyannote 3.1과 community-1 차이?** A: AHC vs VBx 알고리즘, ~3% DER 향상

### 최종 산출물 상태
- **TTS 영어 leak**: 완전 해결 (v15) ✅
- **화자 분리**: 부분 해결, 짧은 발화 detect 한계 존재 (pyannote 자체)
