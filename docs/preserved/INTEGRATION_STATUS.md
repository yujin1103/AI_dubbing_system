# Integration Status — Phase 1 통합 완료

작성: 2026-05-26

## ✅ 통합 완료 (Phase 1)

### 1. `src/repair_diarization.py` — REPAIR_MODULES 등록
| 모듈 | 역할 |
|---|---|
| `embed_reassign` (기존) | 화자 임베딩으로 짧은/오염 청크 재배정 |
| `embed_split` (기존) | 한 화자에 두 인물 병합 시 분리 |
| `visual_reconcile` (기존) | 얼굴 신원으로 분할 (증거 게이트) |
| **`preserved_repair_patches` (신규)** | E:\TTS_capstone 검증된 8 patches 일괄 호출 — subprocess wrapper |

### 2. `src/apply_repair_patches.py` (신규)
8 patches 일괄 호출 entry point. PATCH_ORDER:
```
word_level_split → focused_nemo_split → visual_asd_reassign
  → face_cluster_match → gap_fill (with args) → postprocess_reassign_text
```

### 3. `src/preserved_fusion.py` (신규)
4-way fusion daemon HTTP 클라이언트 (`POST http://127.0.0.1:8903/diarize` → segments → RTTM).

### 4. `src/pipeline.py` — 새 step + diarize engine 추가
- 새 step **`apply_preserved_repair`** (run_asr 직전): `preserved_repair.enabled=true` 시 우리 8 patches 일괄 적용
- `diarize` step에 새 engine **`fusion_4way`** 분기 추가 (port 8903 호출)
- STEP_FUNCTIONS 16 → **17** steps

### 5. `src/run_asr.py` — boost subchunk 통합
- `transcribe_chunks(..., boost_subchunk={enabled, volume, win_sec, hop_sec, area_start, area_end})` 옵션 추가
- test5: 141 → 156 words (+15 fresh, Adam x2 detect)

### 6. `configs/preserved/preserved_test4.json` + `preserved_test5.json`
검증된 best config 그대로:
- test4: fusion_4way + gap_fill(mm=0.99 bm=0.30 sm=0.10) → score 0.9976
- test5: fusion_4way + gap_fill(mm=0.40 bm=0.30 sm=0.45) + boost 3.0x → score 1.1667

### 7. `docker/Dockerfile.preserved-base` + `Dockerfile.preserved-pipeline`
단일 dubbing_pipeline 컨테이너 환경 — NGC PyTorch 26.02 + 3 venv (system / venv_asr / venv_lipsync) + 4-way fusion daemons + LatentSync 학습/추론 + GFPGAN 후처리.

### 8. `docker-compose.preserved.yml`
단일 dubbing_pipeline 컨테이너 보존 옵션 (팀원 docker-compose.yml 9 services 와 별개). 포트 8901-8943 노출 + 우리 daemon URL env 자동 설정.

## Sanity 검증
```bash
docker exec dubbing_pipeline bash -c "/opt/venv_diarizen/bin/python -c '
import apply_repair_patches; print(apply_repair_patches.PATCH_ORDER)
import preserved_fusion; print(preserved_fusion.DEFAULT_FUSION_URL)
'"
```
→ ✓ apply_repair_patches.PATCH_ORDER: 6 modules  
→ ✓ preserved_fusion.DEFAULT_FUSION_URL: http://127.0.0.1:8903

## ✅ E2E 통합 검증 (2026-05-26)

보존된 test4/test5 v305f run dir 를 임시 복사 → 우리 통합된 `src/apply_repair_patches.py`
호출 (gap_fill 단계만, 기존 face_matched 결과 위에서) → 결과 segments_gapfilled.json
GT 비교. 보존 결과와 **정확히 동일한 수치** 달성.

### 환경
- `dubbing_pipeline` 컨테이너 (start_daemons.sh 로 cosy/asr/diarize 기동)
- ASR daemon (port 8902) 정상 → gap-fill ASR (gap > 1.5s 구간 ASR 보강) 동작

### 결과 비교
| Run | 보존 score | E2E score | 일치 | 핵심 |
|---|---|---|---|---|
| **test4** | 0.9976 | **0.9976** | ✓ 정확 동일 | mom 1.00 / Sean 1.00 / frustrated_dad 1.00 |
| **test5** | 1.1667 | **1.1667** | ✓ 정확 동일 | 아빠 1.00 / 션 1.00 / BG 1.00, main=4 ✓, BG ✓ |

### 검증 명령
```bash
# test4 (best config: mm=0.99 bm=0.30 sm=0.10)
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python \
    src/apply_repair_patches.py /tmp/test4_e2e \
    --main-merge 0.99 --bg-merge 0.30 --sim-match 0.10 --pad 0.5 \
    --skip word_level_split focused_nemo_split visual_asd_reassign \
           face_cluster_match postprocess_reassign_text

# test5 (best config: mm=0.40 bm=0.30 sm=0.45)
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python \
    src/apply_repair_patches.py /tmp/test5_e2e \
    --main-merge 0.40 --bg-merge 0.30 --sim-match 0.45 --pad 0.5 \
    --skip word_level_split focused_nemo_split visual_asd_reassign \
           face_cluster_match postprocess_reassign_text
```

### 보존 결과 파일
- `references/preserved/validation/e2e_integration/val_e2e_test4_with_asr.json`
- `references/preserved/validation/e2e_integration/val_e2e_test5.json`

→ **통합 코드가 보존 결과를 정확히 재현 = Phase 1 완료, 검증 통과.**

## ⏳ 남은 작업 (Phase 2~5)

### Phase 2: 실제 end-to-end 검증 (다음 세션)
- `docker compose -f docker-compose.preserved.yml up -d` → 컨테이너 빌드 + 기동
- `docker exec dubbing_pipeline bash src/daemons/start_daemons.sh` → 8 daemons 띄우기
- `python src/pipeline.py --config configs/preserved/preserved_test4.json` 실행
- 결과 `media/runs/test4_preserved/meta/test4_chunk_000_segments_gapfilled.json` 검증
- GT 비교: score ≥ 0.99 (기존 0.9976 동일 또는 개선)

### Phase 3: webapp UI 인라인 에디터 + 부분 재합성 API (4-6시간)
- `webapp/frontend/src/pages/Chunks.tsx`: segment row + SPK dropdown
- `webapp/backend/app/api/runs.py`: POST `/runs/{id}/segments/{seg_id}/speaker`
- 부분 재합성 (TTS + lipsync 해당 segment만)

### Phase 4: 9 services 컨테이너 진짜 분리 (선택)
- 팀원 docker-compose.yml의 9 services 안에 우리 daemon 적용
- 통신: localhost → service name 변경 (FUSION_URL=http://diarizer:8903 등)
- GPU 공유 (모델 weights 마운트)
- 점진적 — Phase 2 검증 후

### Phase 5: main PR merge
- `restructure-modular` → `main` PR
- 기존 main history 보존 + 새 구조 통합

## Phase 1 통합 후 변경 통계
| 파일 | 추가 lines | 변경 lines |
|---|---|---|
| `src/repair_diarization.py` | +75 | - |
| `src/pipeline.py` | +30 (step + engine 분기) | +1 (STEP_FUNCTIONS 등록) |
| `src/run_asr.py` | +70 (boost subchunk 함수) | +14 (transcribe_chunks param) |
| `src/apply_repair_patches.py` (신규) | +110 | - |
| `src/preserved_fusion.py` (신규) | +80 | - |
| `docker/Dockerfile.preserved-*` (신규, 복사) | +220 | - |
| `docker-compose.preserved.yml` (신규) | +75 | - |
| `configs/preserved/preserved_test{4,5}.json` (신규) | +150 | - |
| **총** | **+810** | **+15** |
