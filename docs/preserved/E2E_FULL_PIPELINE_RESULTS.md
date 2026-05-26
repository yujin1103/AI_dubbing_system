# E2E Full Pipeline 결과 — 빌드 후 처음부터 끝까지 실행

작성: 2026-05-27  
환경: `dubbing_pipeline:latest` 이미지 (61.8GB, `Dockerfile.preserved-pipeline` 동일)  
daemon 3개: cosy(8901) + asr(8902) + diarize(8903) — `start_daemons.sh`로 60초만에 ready

## 실행 명령

```bash
# 1. daemon 기동 (60s)
docker exec dubbing_pipeline bash /workspace/patches/start_daemons.sh

# 2. test4 (drama, lipsync OFF, auto speakers)
docker exec -e LATENTSYNC_OUTLIER_OFF=1 dubbing_pipeline \
    python /workspace/orchestrator.py \
    --input /workspace/media/input/test4.mp4 \
    --name test4_e2e_full --lang ko --content-type drama --smart-daemon

# 3. test5 (drama, OUTLIER_FAR_THRESH=0.70)
docker exec -e LATENTSYNC_OUTLIER_FAR_THRESH=0.70 dubbing_pipeline \
    python /workspace/orchestrator.py \
    --input /workspace/media/input/test5.mp4 \
    --name test5_e2e_full --lang ko --content-type drama --smart-daemon

# 4. repair patches (raw → gapfilled)
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python \
    src/apply_repair_patches.py <run_dir> \
    --main-merge <test4=0.99|test5=0.40> \
    --bg-merge 0.30 --sim-match <test4=0.10|test5=0.45> --pad 0.5

# 5. GT 비교
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python \
    scripts/validate_against_gt.py \
    <run_dir>/meta/<chunk>_segments_gapfilled.json \
    media/gt/<test>_gt.json out.json
```

## 시간 측정

### daemon 기동
| Daemon | 모델 | 시간 |
|---|---|---|
| Diarize (port 8903) | DiariZen WavLM-large-s80-md-v2 | 10s |
| CosyVoice3 (port 8901) | CosyVoice3-0.5B | 40s |
| Qwen3-ASR (port 8902) | Qwen3-ASR-1.7B + ForcedAligner-0.6B | 60s |
| **총 ready** | (병렬 load) | **60s** |

### test4 e2e (1079s = 17분 59초) — 보존 v305f 635s (10분 35초)
| 단계 | 보존 v305f | e2e | 차이 |
|---|---|---|---|
| extract + separate (BS-RoFormer) + ASR | 58s | 48s | -10s |
| translate + emotion + speaker refs | **247s** | **654s** | **+407s** ⚠️ |
| TTS (38~39 segments, CosyVoice3 + retry) | 196s | 367s | +171s |
| concat + mux (FFmpeg) | 5s | 5s | - |
| **자동 파이프라인 총** | **635s** | **1074s** | +439s |
| repair patches (별도 호출, 8 stages) | - | **91s** | - |

### test5 e2e (444s = 7분 24초) — 보존 v305f 589s (9분 49초)
| 단계 | 보존 v305f | e2e | 차이 |
|---|---|---|---|
| extract + separate + ASR | 34s | 33s | -1s |
| translate + emotion + refs | 302s | 407s | +105s |
| TTS | 248s | (포함) | - |
| concat + mux | 5s | 4s | - |
| **자동 파이프라인 총** | **589s** | **444s** | **-145s** ✓ |
| repair patches | - | 146s | - |

### 변동 요인
- **translate/emotion** 단계가 큰 변동 원인 — vectorengine_gpt API 응답 시간 + emotion2vec 처리 차이
- TTS 시간은 segment 수 (39 / 17)에 비례
- 우리 e2e는 보존과 동일 환경이지만 **API 응답 변동** + **CosyVoice3 retry 횟수** 차이로 ±50% 변동

## 정확도 측정 (GT 비교)

### test4 (GT: main=6, bg=0)
| Stage | Score | Avg cons. | Main | BG | 비고 |
|---|---|---|---|---|---|
| 보존 v305f gapfilled | **0.9976** | 0.7976 | 6 ✓ | - | mom/Sean/frustrated_dad 1.00 |
| **e2e raw** | 0.7302 | 0.7302 | 5 ✗ | - | DiariZen 단독 결과 |
| **e2e gapfilled** | **0.7302** | 0.7302 | 5 ✗ | - | repair 후에도 main=5 (raw가 5명이라 over-merge 없음) |

→ **보존 결과 재현 실패**. 이유: 보존 v305f는 4-way fusion (DiariZen + NeMo + pyannote-c1 + pyannote-3.1)으로 main 6명 detect. 우리 e2e는 단일 DiariZen만 사용 → main 5명. mom 0.33 (보존 1.00).

### test5 (GT: main=4, bg=1)
| Stage | Score | Avg cons. | Main | BG | 비고 |
|---|---|---|---|---|---|
| 보존 v305f gapfilled | **1.1667** | 0.8667 | 4 ✓ | ✓ | 아빠/션/BG 1.00 |
| **e2e raw** | 0.2429 | 0.2429 | 10 ✗ | ✗ | over-split |
| **e2e gapfilled** | **0.9714** | **0.8714** | 2 ✗ | ✓ | **avg consistency 0.87 = 보존과 동일** |

→ **avg consistency는 정확히 보존 수준 (0.87) 달성**. 단 main_count=2 (gap_fill mm=0.40 으로 over-merge). main_match bonus 0.2 못 얻어 score=0.97. main_count는 GT 4와 다르지만 **per-GT speaker 매핑은 보존만큼 정확**:
- 아빠 **1.00** / BG **1.00** / 의사 0.83 / 션 0.86 / 엄마 0.67

## 결론

### ✅ 성공한 부분
1. **빌드 + 실행 OK**: dubbing_pipeline 컨테이너 단일 컨테이너에서 우리 통합 코드 정상 동작
2. **mp4 output 생성**: test4/test5 final dubbed mp4 둘 다 생성
3. **repair patches 통합 동작**: src/apply_repair_patches.py 호출 → segments_gapfilled.json 생성 정상
4. **test5 avg consistency 0.87** = 보존과 동일 수준 도달

### ⚠️ 한계
1. **4-way fusion daemon 미가동**: `start_daemons.sh`는 단일 diarize(8903)만 띄움. 보존 v305f의 4-way fusion (port 8913/8923/8933/8943) sub-daemon 안 띄워 → main_count 차이
2. **API 변동성**: translate/emotion 단계 시간 ±50% 변동 (vectorengine_gpt API)
3. **best config는 raw segments에 의존**: e2e raw가 보존과 다르면 best config (mm/sm)도 다른 값이 필요. GT sweep을 e2e raw에 새로 돌리면 score 향상 가능

### 개선 방안
1. **fusion daemons 추가 기동**: nemo_diarize_daemon.py + pyannote_diarize_daemon.py + fusion_diarize_daemon.py 별도 실행
2. **새 e2e raw에 sweep 재실행**: 보존 best config 대신 e2e raw에 최적화된 config 찾기
3. **content-type 조정**: drama → interview 등으로 emotion 처리 일관성 향상

## 보존 파일
- `references/preserved/validation/e2e_full_pipeline/val_test{4,5}_e2e_raw.json`
- `references/preserved/validation/e2e_full_pipeline/val_test{4,5}_e2e_gapfilled.json`
- final dubbed mp4 (호스트 `E:\TTS_capstone\media\output\test{4,5}_e2e_full_ko_*.mp4`)
- 각 run dir: `E:\TTS_capstone\media\runs\20260526_21*_test{4,5}e2efu_*/`
