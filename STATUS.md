# Korean Dubbing Pipeline — Project Status

**최종 갱신**: 2026-05-15 (오후, v28 준비)
**현재 버전**: v27 완료 / v28 준비됨 (run 대기)
**호스트**: Windows + Docker (`dubbing_pipeline` 컨테이너)

---

## 🎯 현재 목표

**영어 영상 → 한국어 더빙 + lipsync 파이프라인**의 화자 분리 정확도를 사용자 ground truth에 맞추는 것.

### 우선순위
1. **화자 분리 정확도** ← 사용자가 가장 강조 ("화자 분리가 제일 중요한 관건")
2. 발화 끝 잘림 (trim) 감소
3. 마지막 토끼 sentence split 해소
4. 기계음 완화

### 영상 평가 결과 (사용자 기준)
- v17: 일부 기계음, 첫 부분 잘림 해소
- v18: SPEAKER_02 여성 → 남성화 (큰 회귀), 기계음 악화
- v19: 회귀 fix, 8초 "좋아" 해결
- v20: 가짜 SPEAKER_99/98/97 양산 (OUTLIER 회귀)
- v21: OUTLIER OFF로 fix, 화자 6명 복원
- v22~v25: Face Recognition 시도, 우리 영상에서 효과 없음 (face_id 신호 약함)
- v26: Face Recognition OFF + audio 튜닝 + CAPEL 버그 fix, **안정 baseline 달성**
- v27: 같은 화자 인접 발화 안전 병합 (분할 회복) — SPEAKER_05 5→4 groups

---

## 📋 사용자 Ground Truth (test4.mp4)

### 전체 구성
- **총 6명** 화자

### 화자별 정답
| 화자 | 정답 | 출처 |
|------|------|------|
| **SPEAKER_05** | **영상 중 1번만 발화** | 사용자 여러 차례 강조 |
| **id 0**: "mistake" | **SPEAKER_01** | v15 피드백 |
| **id 0**: "I agree" | **SPEAKER_02** | v15 피드백 (병합되면 안 됨) |
| **id 15** | **SPEAKER_05** | v9 피드백 |
| **id 16** | **SPEAKER_00** | v9 피드백 |
| **id 17** | **SPEAKER_00** ("Good") | v9 피드백 |

### GT 검증 파일
- `media/gt/test4_gt.json`
- `tmp/eval_speaker_count.py` — 자동 검증 스크립트

---

## 🔍 핵심 발견 (v26 → v27 시점)

**사용자 통찰**: "문장의 길이를 제한걸어놔서 저렇게 나오는 거 아닌가"

### 분석 결과 (사용자 의심 정확함)

SPEAKER_05 5 groups가 *진짜 5번 발화*가 아니라 **우리 코드가 같은 발화를 0.4~1.5s gap에서 분할**한 것:

```
[59.20~59.84]  "거기로 가"        ─┐
[60.48~69.36]  "좋아, 그대로 가"   ├ 한 발화 (gap 0.64s, 0.56s)
[69.92~72.00]  "그냥 평범한 사람…" ─┘
[84.40~85.20]  "걔 못 감당해"     ← 떨어진 발화 (gap 12.4s)
[95.92~96.56]  "그만좀"           ← 떨어진 발화 (gap 10.7s)
```

→ 사용자 GT (1번 발화) + 떨어진 2건 = **실제 3 groups가 정답**, 우리 5 → 4 → 3으로 줄여야

---

## 🗺 화자 분리 메커니즘 (8단계)

```
[1] ASR (Qwen3-ASR)                words + timing
       ↓
[2] DiariZen daemon                 39 raw turns, 6 speakers
       ↓
[3] post_process_diarization (line 1059)
    ├ ECAPA centroid 병합
    ├ 짧은 turn 재할당
    ├ ★ 인접 같은 화자 turn 병합 (gap < 0.3s 하드코딩) ← v27에서 1.0s 환경변수화
    └ 너무 짧은 turn 흡수
       ↓
[4] AV-Fusion (LightASD)            spurious 화자 제거 + face owner
       ↓
[5] AV-Reassign                     face owner 기반 라벨 교정
       ↓
[6] LLM 구두점 복원 + _match_punctuated_to_words
    └ ★ punctuation 위치에서 sentence 분리
       └ fallback: gap > 0.4s 에서 분리
       ↓
[7] _split_long_sentence            15s 초과 시 쉼표 → 접속사 → gap 분할
       ↓
[8] _merge_short_sentences          짧은 sentence 인접과 병합
    └ ★ SENT_MERGE_CAP 제한
       ↓
[9] _merge_same_speaker_adjacent (v27 신규) ← 같은 화자 + gap<1.0s + ≤14s 안전 병합
       ↓
[10] _split_groups_by_speaker        sentence 내 화자 변화 split
```

---

## 📏 모든 길이 제한 (현재 v27)

| # | 제한 | 값 (v27) | 환경변수 | 영향 |
|---|------|---------|---------|------|
| 1 | post_process 인접 화자 gap | **1.0s** | `LATENTSYNC_POSTPROC_SAME_SPK_GAP` | 0.3s 하드코딩 → 환경변수화 |
| 2 | fallback gap 분리 | 0.4s | ❌ 하드코딩 | sentence boundary |
| 3 | SENT_MIN_DURATION | 1.5s | `LATENTSYNC_SENT_MIN_DURATION` | 짧은 sentence 병합 |
| 4 | SENT_MIN_WORDS | 3 | `LATENTSYNC_SENT_MIN_WORDS` | 짧은 sentence 병합 |
| 5 | SENT_MAX_DURATION | 15s | ❌ 하드코딩 | 긴 sentence 분할 |
| 6 | SENT_MERGE_CAP | **14s** | `LATENTSYNC_SENT_MERGE_CAP` | v22 10s → v27 14s |
| 7 | SAME_SPK_GAP (신규 v27) | 1.0s | `LATENTSYNC_SAME_SPK_GAP` | 같은 화자 안전 병합 임계 |
| 8 | SAME_SPK_MERGE_CAP (신규 v27) | 14s | `LATENTSYNC_SAME_SPK_MERGE_CAP` | 같은 화자 합쳐도 한계 |
| 9 | SPLIT_NO_SHORT_WORDS | 8 | `LATENTSYNC_SPLIT_NO_SHORT_WORDS` | 짧은 group split 안 함 |
| 10 | SPLIT_MAJORITY_TH | 0.80 | `LATENTSYNC_SPLIT_MAJORITY_TH` | majority 통일 |

---

## 📊 측정 비교 (v17 ~ v27)

| 버전 | runtime | 화자 수 | SPEAKER_05 | trim | 비고 |
|------|---------|---------|-----------|------|------|
| v17 | 9m 50s | 6 | 4 | (없음) | C 작업: soft range + CAPEL |
| v18 | 12m 10s | 6 | 6 | (악화) | SPEAKER_02 남성화 ❌ |
| v19 | 13m 18s | 6 | 4 | (개선) | v18 회귀 fix |
| v20 | 10m 38s | **9** ❌ | - | - | OUTLIER 가짜 99/98/97 양산 |
| v21 | **9m 42s** | 6 | 5 | 8 | OUTLIER OFF fix |
| v22 (재설계) | 미완 | - | - | - | 사용자 지적으로 중단 |
| v23 | 30m 54s | 4 ❌ | - | 6 | Face Recognition over-merge |
| v24 | 16m 47s | 6 | 5 | 7 | Face threshold 0.60 안전화 |
| v25 | 46m 50s | 6 | 5 | 7+ | Face threshold 0.45 / CAPEL 버그 |
| **v26** | **15m 47s** | 6 | 5 | **5** | **안정 baseline** (Face OFF + CAPEL fix) |
| **v27** | **15m 34s** | 6 | **4** | 8 | **분할 회복 (3건 안전 병합 발동)** |

---

## ⚠️ v27 회귀 발견 (2026-05-15 오후 분석)

**v27이 count metric으론 좋아졌지만 라벨 정확도는 회귀.** `tmp/eval_speaker_count.py`는 그룹 수만 보므로 라벨 오염을 못 잡음.

### v26 vs v27 — 59~72s 영역
| Time | v26 | v27 | 평가 |
|------|-----|-----|------|
| 59.20-59.84 | SPK_05 | SPK_05 | 같음 ✓ |
| 60.48-69.36/68.96 | **SPK_05** ✓ | **SPK_04** ✗ | v27 회귀 |
| 68.96-70.24 | (위에 포함) | **SPK_03** ✗ | v27 회귀 (split) |
| 69.92/70.24-72.00 | SPK_05 | SPK_05 | 같음 ✓ |

### 원인 (코드 추적)
`LATENTSYNC_POSTPROC_SAME_SPK_GAP` 0.3→1.0 변경이 `post_process_diarization`에서 같은 화자 인접 turn을 더 길게 합치면서 **ECAPA centroid 분포를 흔듦** → 짧은 turn 재할당이 잘못된 화자로 됨. 의도와 정반대 효과.

### 다른 잔여 이슈
- trim 5→8 증가 (같은 화자 합쳐서 긴 segment → atempo 한계)
- 마지막 토끼 split (SPK_05 + SPK_03)

---

## 🐛 TTS 기계음 — 원인 분석 (2026-05-15 오후)

### 🔴 1순위 (명백한 버그, fix됨): cosyvoice_daemon fade-in/out 누락
- `synthesize_segment_cosy` (orchestrator inline path): 50ms fade-in + 30ms fade-out ✓
- `cosyvoice_daemon.py` `/synthesize` (production path): **둘 다 없음** ✗
- `_check_cosy_daemon()` 가 production을 데몬 path로 라우팅 → v17~v27 모든 segment cold-start click
- **fix**: `patches/cosyvoice_daemon.py` 직접 수정 (native sr에서 fade 적용, 인라인과 동일)

### 🟡 2순위: selfref 사용 (특정 화자만)
- profile 못 만든 화자가 segment 자체를 reference로 사용
- v27 SPK_05 4개 중 3개가 selfref → 짧고 BGM 잔여 가능, quality 저하

### 🟢 3순위: atempo 한계 (MAX 1.15, MIN 0.90)
- rubberband WSOLA로 1.15까지 자연. 1.40+은 즉시 trim fallback
- ratio가 큰 segment에서 부자연

### 🟢 4순위: 본질적 한계
- CosyVoice3 영어 ref + 한국어 cross-lingual prosody 어색 (모델 한계)

---

## 📊 평가 방법 (자동 시스템 원칙)

**중요**: 자동 다국어 더빙 시스템이므로 per-video GT 라벨링은 사용자에게 요구하지 않음. 평가는 다음으로:

1. **`tmp/eval_speaker_count.py`** — coarse 제약 검증 (count 위주, 한계 있음)
2. **`tmp/eval_diarization_auto.py`** (신규, 2026-05-15) — GT 없는 self-consistency metric:
   - group count per speaker
   - selfref usage rate
   - fragmented sentence count (한 sentence가 여러 화자로 쪼개진 경우)
   - rapid speaker transitions (<1s)
   - extreme short/long groups
   - per-speaker dominant gap distribution
3. **개발자 시각 비교** — 출력 영상 직접 확인 (사용자 라벨링 X)

다른 영상에도 그대로 적용 가능 (영상별 GT 불필요).

---

## 🎯 v28 계획 (준비 완료, run 대기)

`tmp/run_test4_v28.sh`:
- **[A] `LATENTSYNC_POSTPROC_SAME_SPK_GAP` 언셋** — default 0.3 복원 (v27 회귀 원인 제거)
- **[B] `LATENTSYNC_SAME_SPK_GAP=1.0` / `LATENTSYNC_SAME_SPK_MERGE_CAP=14.0`** 유지
  - sentence-level merge는 안전 (같은 화자만 합침, label 변경 X)
  - 이 단계가 v26 베이스에서 SPK_05 3개 group → 1개로 안전 회복 시킬 것
- **[C] daemon fade fix 적용됨** — 데몬 재시작 필수

### 데몬 재시작 (사용자 실행 필요)
```bash
# 1. 호스트 파일을 컨테이너로 복사
docker cp E:/TTS_capstone/patches/cosyvoice_daemon.py dubbing_pipeline:/workspace/patches/cosyvoice_daemon.py

# 2. 컨테이너 안에서 데몬 프로세스 재시작 (8901 port)
docker exec dubbing_pipeline bash -c "pkill -f 'cosyvoice_daemon.py' || true; sleep 2; nohup /opt/venv_cosy/bin/python /workspace/patches/cosyvoice_daemon.py --port 8901 > /tmp/cosy_daemon.log 2>&1 &"

# 3. 헬스 체크
docker exec dubbing_pipeline curl -s http://127.0.0.1:8901/health
```

### v28 실행
```bash
docker cp E:/TTS_capstone/tmp/run_test4_v28.sh dubbing_pipeline:/tmp/run_test4_v28.sh
docker cp E:/TTS_capstone/tmp/eval_diarization_auto.py dubbing_pipeline:/workspace/tmp/eval_diarization_auto.py
docker exec dubbing_pipeline bash /tmp/run_test4_v28.sh
```

### v28 기대 효과
- SPK_05 = 3 groups (1 main merged 59-72s + 84.40 + 95.92)
- 59-72s 영역 모든 라벨 SPK_05 (v26 수준 회복)
- 모든 segment 음질 개선 (fade ramp 적용)

---

## 🛠 기술 스택

### TTS
- **CosyVoice3** (FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- `inference_instruct2` + `<|endofprompt|>` token
- Daemon: `cosyvoice_daemon.py` port 8901

### ASR
- **Qwen3-ASR-1.7B** daemon port 8902

### Speaker Diarization
- **DiariZen** (BUT-FIT/diarizen-wavlm-large-s80-md-v2) daemon port 8903
- **ECAPA-TDNN** centroid post-processing
- **LightASD** + face owner mapping (face Recognition은 v26에서 OFF)

### Audio Source Separation
- **BS-RoFormer** + **MDX23C** Subtractive Ensemble

### LLM Translation
- VectorEngine (gpt-5.4)
- Soft syllable target + CAPEL countdown for short utterances

### Lipsync
- **LatentSync 1.6** (UNet3D)

---

## 🗂 핵심 파일

### 메인 파이프라인
- `orchestrator.py` — 전체 파이프라인 (~5000 lines)
- `patches/cosyvoice_daemon.py` — TTS daemon
- `patches/av_fusion.py` — AV diarization + reassign
- `patches/face_id_embedder.py` — InsightFace face_id (v26+ OFF)

### 검증/실행
- `media/gt/test4_gt.json` — 사용자 ground truth
- `tmp/eval_speaker_count.py` — 자동 화자 검증
- `tmp/run_test4_v27.sh` — 최신 측정 스크립트

### 백업
- `patches/cosyvoice_frontend_original.py` — CosyVoice frontend 백업
- `patches/cosyvoice_frontend_v20_patched.py` — #1400 패치 적용본

---

## 🎬 다음 단계 옵션

### A. v28 실행 (준비됨) ← **권장**
위 v28 계획대로 daemon 재시작 + run. auto metric으로 회귀 자동 감지.

### B. v28 결과 따라 추가 튜닝
- 라벨 회복 OK + SPK_05 안 합쳐지면 → `LATENTSYNC_SAME_SPK_GAP` 1.0→1.2 (gap 0.64s, 0.56s 다 통과)
- 라벨 회복 됐지만 다른 화자에서 회귀 발생 → 영상 비교 후 변경 결정

### C. DiariZen native 파라미터 노출
지금까지 한 번도 안 건드림. `ahc_threshold=0.6` (default) 등이 있어서 후처리 우회 가능성:
- `patches/diarize_daemon.py` 에 `DiariZenPipeline(config_parse={...})` 옵션 추가
- 안전한 우선 시도: `ahc_threshold` 0.6 → 0.55 (clustering merge 약간 더 적극)

### D. selfref 빈도 줄이기
- REF_MIN_DUR 6.0 → 4.0 검토 (짧은 ref 단점과 trade-off)
- 또는 다른 시점의 같은 화자 segment를 cross-reference로 활용

### E. DiariZen 자체 교체 (큰 작업, 보류)
NeMo Sortformer는 4-speaker 한계로 test4 (6명)에 부적합. pyannoteAI Precision-2는 상용. DiariZen 유지가 합리.

### F. TTS 교체 검토 (보류)
GPT-SoVITS v4 한국어 운율 우수, fine-tune 화자당 ~1.5분 (1-2일).

---

## 🚫 NOT-TODO (설계 원칙)

- 영상별 GT 라벨링 작업 (사용자가 영상 보며 누가 언제 발화하는지 입력)
  - 이유: 자동 다국어 더빙 시스템. 최종 사용자는 영상 안 보고 input.
  - 평가는 위 §평가 방법 참고.

---

## 🔄 새 세션에서 이어가기

```cmd
cd /d E:\TTS_capstone
claude --continue        :: 가장 최근 세션
claude --resume          :: 세션 목록에서 선택
```

세션 ID: `3af61dcf-4ebb-493f-ba3d-82f0e63c9dd3`
세션 파일: `C:\Users\jsje1\.claude\projects\E--TTS-capstone\3af61dcf-4ebb-493f-ba3d-82f0e63c9dd3.jsonl`

## 📁 컨테이너 작업 패턴

```bash
# 호스트 파일 컨테이너로 push
docker cp "E:/TTS_capstone/path/file" dubbing_pipeline:/workspace/scripts/file

# 컨테이너에서 측정 실행
docker exec dubbing_pipeline bash /tmp/run_test4_vXX.sh

# 결과 영상 위치
E:\TTS_capstone\media\output\test4_vXX_ko_*.mp4
```
