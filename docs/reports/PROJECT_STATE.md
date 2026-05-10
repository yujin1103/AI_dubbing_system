# PROJECT STATE — TTS Capstone (한국어 더빙 + 립싱크)

> **마지막 업데이트**: 2026-05-04 (4일차, 🎉 LoRA 학습 50k 완료 + latentsync_ko.pt 생성)
> **목적**: 컨텍스트 손실 / 새 세션 시작 시 작업 재개용 종합 문서
> **위치**: `E:/TTS_capstone/PROJECT_STATE.md`

---

## 🚨 새 세션 시작 시 — 빠른 가이드

```bash
# 1. 컨테이너 살아있는지 확인 (이미 떠있으면 OK)
docker ps --filter name=dubbing_pipeline

# 2. 안 떠있으면 시작
docker compose -f E:/TTS_capstone/docker-compose.yml up -d

# 3. AIHub 다운로드 상태 확인
ls "E:/Download/009.립리딩(입모양)_음성인식_데이터/01.데이터/1.Training/원천데이터/" | wc -l
# 기대: TS{1,10,20,30,40,50}.tar.part* 파일들

# 4. 다운로드 끝났으면:
bash scripts/extract_aihub.sh                                          # 추출
docker exec dubbing_pipeline /opt/venv_lipsync/bin/python \
  /workspace/scripts/aihub_face_crop.py \
  --json_root /workspace/media/aihub_extracted/labels_train \
  --video_root /workspace/media/aihub_extracted/video_train \
  --out_dir /workspace/media/aihub_processed/train \
  --filter_angle A                                                    # 전처리 (multi-angle 옵션 추가 필요)

# 5. 학습 config 만들고 학습 시작
# (configs/lora_full_train.yaml 작성 → "학습 명령어" 섹션 참고)
```

---

## 🎯 프로젝트 개요

영어 영상을 **한국어로 더빙 + 립싱크 자동화** 파이프라인. 최종 목표는 **AIHub 한국어 데이터로 LatentSync LoRA 파인튜닝**해서 한국어 발음에 정확한 립싱크 모델 만들기.

### 사용 환경
- **OS**: Windows 11 + Docker Desktop + WSL2
- **GPU**: RTX 5080 (16GB VRAM, sm_120 Blackwell)
- **RAM**: 63GB (Docker 50g 할당)
- **Disk**: E: drive 1.9TB
- **Container**: `dubbing_pipeline` (이미지 56GB)

---

## 📅 작업 히스토리 (3일)

### Day 1 (2026-04-30)
- LatentSync 1.6 통합한 더빙 파이프라인 구축
- 결과물: `test_ko_20260430_..._lipsync.mp4` (LatentSync 결과, 어제까지 reference)

### Day 2 (2026-05-01) — MuseTalk 시도
- LatentSync → MuseTalk 1.5(HD) 전환 시도
- 6가지 빌드 버그 수정 (mmpose, weights_only, blending, FA_LANDMARK, face-parse-bisent, ffmpeg path)
- GFPGAN face restoration 후처리 추가
- **결과 v2**: `test_ko_..._lipsync_v2.mp4` (1080p + GFPGAN, but 영상-오디오 sync 깨짐)

### Day 3-1 오전 (2026-05-02) — Sync fix + 비교
- **음성-영상 sync 버그 수정**: `synthesize_chunk` trim + `mix_audio` `duration=shortest`
- **화자 분리 후처리**: ECAPA-TDNN centroid 기반 (`post_process_diarization`)
- **GFPGAN fp16 fallback** 버그 fix ("FloatTensor"/"HalfTensor" 키워드 검출)
- **결과 v3final**: `test_ko_20260502_041603_test_06bda5_lipsync_v3final.mp4`
  - 1080p + GFPGAN + sync 일치 (video=audio=64.04s)
- 사용자 평가: **MuseTalk 부정확. LatentSync로 회귀하기로 결정**

### Day 3-2 오후 (2026-05-02) — LatentSync 학습 가능성 검증 ✅
- LatentSync `stage2_efficient.yaml` 분석 → 20GB VRAM 권장이지만 14GB로 fit 시도
- 메모리 최적화 → **검증 완료**:
  - `pixel_space_supervise: false` (VAE decode 스킵, ~3-4GB 절감)
  - `num_frames: 16 → 2` (~6GB 절감)
  - `perceptual_loss_weight: 0.0` (LPIPS off)
  - **8-bit AdamW (bitsandbytes)** — optimizer state 75% 절감 ⭐ 핵심
  - `gc.collect() + torch.cuda.empty_cache()` 5 step마다
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6`
- **테스트 결과**:
  - Smoke 10 step: ✅ peak 9.9GB
  - Stress 100 step: ✅ peak 15.1GB
  - Dummy 1000 step: ✅ peak 15.84GB, **12분 27초**, loss 0.017→0.00181
- **속도**: 1.34 it/s
- **50k step 예상**: ~10시간 23분 (밤새 1회 가능)

### Day 3-3 저녁 — AIHub 데이터 준비 + LoRA 학습 인프라 + 마스킹 개선
- AIHub 538 데이터셋 분석 — 5,750h, 1,107 화자, 9 angles, 6 noise envs
- TL/TS 매핑 검증:
  - 14개 TL 다운로드 + JSON 분석 → noise/demographics 확정
  - **TL1~50 = 무음 (noise1)** ✅
  - **TL60~90 = 생활 (noise2)**
  - TL100=교통, TL130=산업, TL174=자연, TL200=기타
- **사용자 다운로드 중**: TS1, TS10, TS20, TS30, TS40, TS50 (6개 무음 TS, 510GB)

### Day 3-4 저녁 — LoRA 통합 + 마스킹 개선 + 정리 ✅
- **LoRA 학습 환경 구축** (Partial FT 대비 1/68 params, 1/600 파일 크기):
  - PEFT 0.10 + LoraConfig (target: to_q/k/v/out)
  - r=32, alpha=16, target_modules 확장
  - LoRA+ (A/B matrix 분리 lr, B=16x)
  - 검증: smoke 30 step ✅, peak VRAM 14.5GB
- **MuseTalk 완전 제거** + 폴더 구조 정리 (scripts/, configs/, patches/, logs/)
  - 삭제: musetalk_runner.py, musetalk_skip_patch.py, 모든 run_*.log, build*.log, gfpgan/ runtime cache
  - 이동: 루트 11개 → scripts/(4) + configs/(2) + patches/(6) + logs/
  - orchestrator.py에서 MuseTalk 코드 ~10KB 제거
  - dockerfile.orchestrator MuseTalk 섹션 제거
- **다국어 LoRA 지원 인프라**:
  - `resolve_lipsync_ckpt(tgt_lang, ...)` — 자동으로 `latentsync_<lang>.pt` 검색
  - 한국어 학습 후 `media/lora/latentsync_ko.pt` 두면 `--lang ko` 자동 사용
  - 일본어/스페인어 등 추가 학습 가능 (코드 변경 불필요)
- **마스킹 개선 코드** (사용자 보고 "옆모습 마스크 보임" 해결):
  - `patches/soft_mask_patch.py` — Gaussian feather 50px (build 시 자동)
  - `scripts/lipsync_postprocess.py` — GFPGAN + LAB color matching
  - `patches/face_padding_patch.py` — crop_ratio × 1.15 (선택, 학습 분포 일치 필요)
  - 검증: 어제 LatentSync 결과 후처리 → 얼굴 디테일 ↑↑, 마스크 경계 거의 사라짐
- **학습 모니터** (`scripts/monitor_train.py`):
  - CLI 모드: 실시간 dashboard (step/loss/VRAM/ETA)
  - Library 모드: UI에서 import 가능 (`parse_train_log_tail`, `get_gpu_status`)
  - JSON 출력 모드 (UI 연동용)
  - 자동 알림 (학습 완료/실패 시 텍스트 파일)
  - 4가지 경고 감지: GPU peak 92%+, VRAM 누적, 디스크 부족, 학습 멈춤

---

## 📂 폴더 구조 (정리됨, 2026-05-02)

```
E:/TTS_capstone/
├── orchestrator.py          # 더빙 파이프라인 메인 (LatentSync only)
├── asr_worker.py            # qwen-asr subprocess
├── mos_evaluator.py         # TTS 품질 평가
├── train_mos.py             # MOS 모델 학습
├── video_extraction.py      # 영상 분할 유틸
├── dockerfile.base          # 베이스 이미지
├── dockerfile.orchestrator  # 통합 빌드 (LatentSync + LoRA 학습 패치 자동 적용)
├── docker-compose.yml       # 컨테이너 실행 (RAM 50g, IPC host)
├── PROJECT_STATE.md         # 이 파일
├── requirements_installed.txt
├── .env, .dockerignore, .gitignore
│
├── scripts/                 # 보조 실행 스크립트
│   ├── aihub_face_crop.py   # 1080p mp4 → 256x256 25fps 16kHz
│   ├── extract_aihub.sh     # AIHub Innorix 분할 tar 합치기
│   ├── inspect_unet.py      # LoRA 부착 layer 분석 (디버깅)
│   ├── lora_merge.py        # LoRA → 베이스 merge (한국어 .pt 생성)
│   ├── lipsync_postprocess.py  # GFPGAN + LAB color matching (마스킹 개선)
│   └── monitor_train.py     # 🆕 학습 모니터 (CLI + Library)
│
├── configs/                 # 학습 yaml
│   ├── lora_smoke.yaml      # LoRA 검증용 (10~30 step)
│   └── partial_ft_smoke.yaml  # Partial FT (legacy 비교용)
│
├── patches/                 # 빌드 시 자동 적용 (dockerfile COPY)
│   ├── latentsync_train_patch.py  # 8-bit AdamW + gc/empty_cache
│   ├── lora_patch.py              # use_lora 옵션 (PEFT)
│   ├── lora_save_patch.py         # LoRA-only checkpoint 저장
│   ├── lora_plus_patch.py         # LoRA+ (A/B 분리 lr)
│   ├── soft_mask_patch.py         # 🆕 mask.png Gaussian feather (마스킹 부드럽게)
│   └── face_padding_patch.py      # 🆕 crop_ratio × 1.15 (선택, 수동 적용)
│
├── logs/                    # 학습/추론 로그 (현재 비어있음)
│
└── media/                   # 데이터 (mounted to /workspace/media in container)
    ├── input/               # 입력 영상
    ├── output/              # 더빙된 영상 + lipsync mp4
    ├── runs/                # run_id별 작업 디렉터리
    ├── reports/             # JSON 리포트
    ├── model_cache/         # HF/torch 가중치 캐시
    │   ├── latentsync/checkpoints/latentsync_unet.pt   # 베이스 5GB
    │   └── musetalk/, huggingface/, modelscope/, torch/
    ├── lora/                # 🌐 lang별 LoRA 가중치 (자동 인식)
    │   └── latentsync_<lang>.pt  (학습 후 여기 두면 --lang 따라 자동 사용)
    ├── aihub_extracted/     # AIHub raw 추출본
    │   ├── labels_train/, labels_val/  # JSON
    │   └── video_train/, video_val/    # 1080p mp4
    ├── aihub_processed/     # face-crop + 25fps + 16kHz mono
    │   ├── train/, val/
    │   └── fileslist_*.txt
    ├── audio_cache/         # whisper feature 캐시
    └── training_data/, training_outputs/
```

## 🚮 삭제됨 (2026-05-02)

- MuseTalk 관련 모든 파일 (musetalk_runner.py, musetalk_skip_patch.py)
- 모든 run_*.log, build*.log 파일
- gfpgan/ runtime 캐시
- _test_lora.py, _smoke_test_config.yaml (deprecated)

### 산출물 (호스트 영구)
| 파일 | 설명 |
|---|---|
| `media/output/test_ko_20260430_..._lipsync.mp4` | **LatentSync 어제 결과 (reference)** |
| `media/output/test_ko_..._lipsync_postprocess.mp4` | LatentSync + GFPGAN 후처리 (마스킹 개선 검증용) |
| **`media/lora/latentsync_ko.pt`** | ⭐ **한국어 LoRA merge 결과 (4.10 GB)** — 5/4 생성, best ckpt=40k |
| `media/training_outputs/lora_full_train/train-2026_05_03-05:32:36/checkpoints/checkpoint-{5,10,15,20,25,30,35,40,45,50}000.pt` | 모든 LoRA-only 체크포인트 (45MB × 10) |

### 🎉 LoRA 학습 결과 (2026-05-03 ~ 05-04, 20시간 10분)
- **데이터**: 588 영상 (multi-angle, AIHub 538 6 TS)
- **설정**: LoRA r=32, alpha=16, target=q/k/v/out, batch=2, num_workers=8, 8-bit AdamW
- **best ckpt**: 40k (SyncNet 1.86, AV offset -1)
- **최종 결과**: `media/lora/latentsync_ko.pt` (베이스+LoRA merge, 4.10 GB)

| Step | SyncNet | AV offset |
|---|---|---|
| 5k | 0.94 | 0 |
| 10k | 1.08 | 1 |
| 15k | 1.51 | 0 |
| 20k | 1.67 | 0 |
| 25k | 1.44 | 0 |
| 30k | 0.89 | 0 |
| 35k | 1.46 | 0 |
| **40k** | **1.86** ⭐ | -1 |
| 45k | 1.38 | 0 |
| 50k | 1.46 | 0 |

### 모델 가중치 (호스트 영구)
- `media/model_cache/latentsync/checkpoints/latentsync_unet.pt` (5GB) — 학습 base
- `media/model_cache/musetalk/models/gfpgan/GFPGANv1.4.pth` (~350MB) — 후처리용 GFPGAN
- `media/lora/latentsync_<lang>.pt` (학습 후 생성) — 언어별 LoRA

### 컨테이너 내 상태 (재시작 시 유지, 재빌드 시 자동 재생성)
- `/opt/LatentSync/` — 학습 코드 (clone, 모든 패치 적용됨)
- `/opt/LatentSync/checkpoints/` → 호스트 캐시 심볼릭 링크
- `/opt/LatentSync/latentsync/utils/mask.png` — Gaussian feather 적용된 부드러운 mask
- `/opt/venv_lipsync/` — 학습/추론용 venv (PyTorch 2.11+cu130, PEFT 0.10, GFPGAN, bitsandbytes)

---

## 🛠 핵심 기술 결정 + 이유

### 1. LatentSync stage2_efficient (16GB VRAM fit)
**why**: stage2_512 (30GB) + stage2 (20GB) 모두 OOM. efficient는 motion_modules + attn2만 학습 (부분 fine-tune).

### 2. `pixel_space_supervise: false`
**why**: VAE decode 단계가 OOM 주범. False로 두면 latent recon loss만 사용 → 한국어 viseme 적응에는 충분. 정확도 trade-off 미미.

### 3. `num_frames: 2`
**why**: 16 → 8 → 4 모두 OOM 누적. 2까지 줄여야 14GB fit. 시간 일관성 학습은 적은 frame으로도 가능.

### 4. 8-bit AdamW (bitsandbytes)
**why**: optimizer state (m, v) fp32 → int8 압축 → ~75% 절감. **학습 가능성의 결정적 요소**. 정확도 영향 거의 없음.

### 5. 8개 TL 받아서 매핑 확인 후 TS 다운로드
**why**: TS는 80GB+ 큼. 무작정 받으면 비효율. TL (700MB)로 noise/demographics 확인 → 무음(noise1)인 TS만 선택적 다운.

### 6. AIHub 정면(A) 위주 + 측면 일부
**why**: LatentSync는 정면 위주 학습. 측면 너무 많으면 catastrophic forgetting. 옵션 A 권장: A 100% + B/C/F/G 50%.

### 7. LoRA over Partial FT (Day 3-4 결정)
**why**: 4M trainable params (vs 285M partial), 베이스 영어/중국어 100% 보존, 다국어 swap 쉬움. 데이터 50h ≈ 4M params 매칭으로 partial FT 수준 quality 가능. LoRA+ (A/B 분리 lr) 추가로 품질 boost.

### 8. LoRA target_modules: to_q/k/v/out (r=32, alpha=16)
**why**: attn1+attn2+motion_modules 모든 cross/self-attention 부착. r=32로 capacity 확보. alpha=r/2 (smoother transfer). Day 3-4 검증: 30 step ✅.

### 9. GFPGAN 후처리 (학습 무관)
**why**: 마스크 경계 보임 + 색상 불일치를 face restoration으로 동시 해결. 학습한 LoRA와 독립적이라 안전. 처리 시간 +21min/영상.

### 10. soft_mask Gaussian feather (build 시 자동)
**why**: mask.png를 빌드 시 Gaussian blur 50px 적용. 마스크 경계가 거의 안 보이게. 학습/추론 모두 영향. dockerfile에 영구화.

---

## 🚀 학습 명령어 (재시작용)

### 환경 진입
```powershell
docker compose -f E:/TTS_capstone/docker-compose.yml up -d
docker exec -it dubbing_pipeline bash
```

### LoRA Smoke test (검증)
```bash
docker cp configs/lora_smoke.yaml dubbing_pipeline:/opt/LatentSync/lora_smoke.yaml
docker exec dubbing_pipeline bash -lc 'cd /opt/LatentSync && \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6" \
  /opt/venv_lipsync/bin/python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29501 \
    -m scripts.train_unet --unet_config_path lora_smoke.yaml'
```

### AIHub 추출 (다운로드 완료 후)
```bash
bash scripts/extract_aihub.sh   # host Git Bash
```

### 전처리
```bash
docker exec dubbing_pipeline /opt/venv_lipsync/bin/python /workspace/scripts/aihub_face_crop.py \
  --json_root /workspace/media/aihub_extracted/labels_train \
  --video_root /workspace/media/aihub_extracted/video_train \
  --out_dir /workspace/media/aihub_processed/train \
  --filter_angle A
```

### 본 학습 (LoRA r=32 + LoRA+, ~8h for 25k step)
```bash
# configs/lora_full_train.yaml 작성 후 (lora_smoke.yaml + max_train_steps: 25000 + train_data_dir 변경)
docker exec dubbing_pipeline bash -lc 'cd /opt/LatentSync && \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6" \
  /opt/venv_lipsync/bin/python -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29501 \
    -m scripts.train_unet --unet_config_path lora_full_train.yaml'
```

### 학습 후 — LoRA → 한국어 .pt 생성
```bash
docker exec dubbing_pipeline /opt/venv_lipsync/bin/python /workspace/scripts/lora_merge.py \
  --base /opt/LatentSync/checkpoints/latentsync_unet.pt \
  --lora /workspace/media/training_outputs/lora_full_train/checkpoints/checkpoint-25000.pt \
  --out /workspace/media/lora/latentsync_ko.pt \
  --r 32 --alpha 16 --target to_q to_k to_v to_out.0
```

### 추론 (한국어 LoRA 자동 적용)
```bash
docker exec dubbing_pipeline python /workspace/orchestrator.py \
  --input /workspace/media/input/test.mp4 --lang ko \
  --speakers 1 --content-type lecture --enable-lipsync
# → media/lora/latentsync_ko.pt 자동 인식
```

### 마스킹 개선 후처리 (선택)
```bash
docker exec dubbing_pipeline /opt/venv_lipsync/bin/python /workspace/scripts/lipsync_postprocess.py \
  --input /workspace/media/output/test_ko_..._lipsync.mp4 \
  --output /workspace/media/output/test_ko_..._lipsync_final.mp4 \
  --weight 0.5
```

---

## 📋 모든 패치 (dockerfile build 시 자동)

빌드만 하면 LatentSync에 자동 적용 (수동 작업 불필요):

1. **8-bit AdamW + gc/empty_cache** (`patches/latentsync_train_patch.py`)
2. **use_lora 옵션** (`patches/lora_patch.py`) — PEFT LoraConfig 부착
3. **LoRA-only checkpoint 저장** (`patches/lora_save_patch.py`) — ~16MB만 저장
4. **LoRA+ A/B 분리 lr** (`patches/lora_plus_patch.py`)
5. **soft_mask Gaussian feather** (`patches/soft_mask_patch.py`) — mask.png 교체
6. **(선택)** face_padding ratio × 1.15 (`patches/face_padding_patch.py`) — 수동 적용

### 컨테이너 의존성 (dockerfile에 영구화됨)
- `peft==0.10.0` (LoRA 학습)
- `gfpgan==1.3.8`, `basicsr==1.4.2`, `facexlib==0.3.0` (마스킹 후처리)
- `kornia==0.8.0`, `lpips==0.1.4`, `DeepCache==0.1.1` (학습 보조)
- `scenedetect==0.6.1`, `lmdb`, `scikit-image`, `filterpy`, `tb-nightly`
- `insightface==0.7.3`
- `bitsandbytes>=0.49.0` (8-bit AdamW)
- `speechbrain` (ECAPA centroid 후처리, 이미 있음)

### basicsr PyTorch 2.x 패치
```python
# /opt/venv_lipsync/lib/python3.12/site-packages/basicsr/data/degradations.py
# torchvision.transforms.functional_tensor → torchvision.transforms.functional
```

---

## 📊 데이터 상태

### AIHub 538 — TL 매핑 (확정)
```
TL1~50: 무음 (noise1) ─ 가장 핵심 데이터
TL60~90: 생활 (noise2)
TL100~?: 교통 (noise3)
TL130~?: 산업 (noise4)
TL174~: 자연 (noise5)
TL200~: 기타 (noise6)
```

### 추출된 mp4 (현재)
| 위치 | 내용 |
|---|---|
| `media/aihub_extracted/labels_train/TL{1,10,20,30,40,50,60,70,80,90,100,130,174,200}/` | JSON 14 폴더 |
| `media/aihub_extracted/labels_val/VL11/` | Validation JSON |
| `media/aihub_extracted/video_train/TS174/` | TS174 영상 60개 (자연소음) ⭐ 학습 가능 상태 |
| `media/aihub_extracted/video_val/VS11/` | VS11 영상 60개 (생활소음) |
| `media/aihub_processed/train/` | 정면 12개 256×256 전처리 완료 |
| `media/aihub_processed/val/` | 정면 12개 256×256 전처리 완료 |

**다음 단계 — TS1/10/20/30/40/50 mp4 추출 필요**:
`scripts/extract_aihub.sh`로 .tar.part → mp4 추출. 한 TS당 ~80GB mp4 생성 예상.

### 다운로드 + .tar 풀기 ✅ 100% 완료 (2026-05-02 밤)

**.tar 풀기 = 1차 압축 해제** (Innorix `.tar` → `TS{N}.tar.part*` 분할 파일들):

| TS | 크기 | .tar.part | 상태 |
|---|---|---|---|
| TS1 (5명 20대 F 일반) | 92.41 GB | 93 parts | ✅ |
| TS10 (5명 30대 F 일반) | 80.00 GB | 81 parts | ✅ |
| TS20 (5명 40~50대 F 일반) | 86.80 GB | 87 parts | ✅ |
| TS30 (5명 20대 M 일반) | 81.10 GB | 82 parts | ✅ |
| TS40 (5명 30대 M 일반) | 94.20 GB | 95 parts | ✅ |
| TS50 (3명 20~40대 F 전문가) | 75.07 GB | 76 parts | ✅ |
| **합계** | **509.58 GB** | **514 parts** | |

위치: `E:/Download/009.립리딩(입모양)_음성인식_데이터/01.데이터/1.Training/원천데이터/`
파일 형식: `TS{N}.tar.part{byte_offset}` (Innorix 1GB 분할, mp4는 아직 미추출)

다운로드 후 예상 데이터:
- 6 화자 그룹 × 9 angles × 5분 영상
- 정면(A) 사용 시 ~324 영상 = ~27시간
- A+B+C+F+G 50% 사용 시 ~648 영상 = ~54시간

**디스크 정리 (2026-05-02 밤)**:
- TS174.tar.part 23개 삭제 (22.57GB) — TS174 mp4 이미 추출되어 있음
- download (10/11/12/13/14/15).tar 6개 삭제 (~600GB) — .tar.part 추출 완료된 원본
- **결과**: 22.57 → **517.41 GB free** (E: 드라이브)

---

## ✅ 검증된 사실

1. **Partial FT는 RTX 5080 16GB에서 학습 가능** (1000 step 완주, peak 15.8GB)
2. **LoRA 학습 가능 — 메모리 절반** (peak 8.4GB, 1000 step 완주)
3. **LoRA+ r=32 + bs=2 + grad_ckpt off** (peak 14.5GB, 30 step 검증, 데이터 효율 1.9×)
4. **AIHub 데이터 포맷**: 1920×1080 30fps 44.1kHz stereo, 5분 클립
5. **Soft mask + GFPGAN 후처리**: 어제 LatentSync 결과 검증 — 마스크 거의 안 보임 ⭐
6. **다국어 LoRA 인프라**: `latentsync_<lang>.pt` 자동 인식 (코드 변경 불필요)
7. **속도 비교**: Partial 0.75s/it / LoRA 1.04s/it / LoRA+ 1.10s/it (데이터 효율 LoRA 우위)
6. **전처리 속도**: 18~30초/영상 (1080p → 256×256 25fps + 16kHz mono)

---

## 🔄 미완료 작업 (다운로드 후)

### 즉시 (다운로드 완료 시점)
1. **6 TS 추출** — `bash scripts/extract_aihub.sh` (~10~30분, parts merge + tar 풀기)
2. **multi-angle 전처리** — `aihub_face_crop.py --filter_angles A,B,C,F,G` (~40분)
3. **fileslist 생성** (전처리 스크립트가 자동)

### 학습 준비
4. **`configs/lora_full_train.yaml` 작성** (lora_smoke.yaml 기반):
   - `max_train_steps: 25000` (LoRA r=32 + LoRA+ + bs=2 효율 활용)
   - `save_ckpt_steps: 5000`
   - `train_data_dir: /workspace/media/aihub_processed/train`
   - `train_output_dir: /workspace/media/training_outputs/lora_full_train`

### 학습 실행 (~8h, 밤새)
5. **LoRA+ 학습** (검증된 settings 그대로):
   ```
   use_lora: true, lora_r: 32, lora_alpha: 16
   target_modules: [to_q, to_k, to_v, to_out.0]
   use_lora_plus: true, lora_plus_ratio: 16
   batch_size: 2, num_frames: 2
   enable_gradient_checkpointing: false
   use_8bit_adam: true
   ```
6. **체크포인트 검증** — 5000/10000/25000 step별 inference 시도

### 학습 후
7. **LoRA merge → 한국어 .pt** (`scripts/lora_merge.py`)
   ```
   /workspace/media/lora/latentsync_ko.pt 저장 → 자동 인식
   ```
8. **추론 테스트** — `test.mp4`에 적용
9. **마스킹 개선 후처리** (선택): `scripts/lipsync_postprocess.py`
10. **3-way 비교**: baseline / LoRA / LoRA+postprocess

### 평가
11. **품질 비교 (시각)**:
    - 한국어 viseme 정확도
    - 마스킹 (옆모습/회전 시)
    - 시간 일관성 (떨림)
12. **다국어 demo (선택)**: 일본어/스페인어 영상 test (LoRA 없이 base 작동)

### 발표 준비
13. PROJECT_STATE 업데이트 + 결과 영상 정리
14. Architecture diagram (다국어 LoRA swap 시스템 강조)

---

## 🆘 트러블슈팅 가이드

### Docker 시작 안 됨
```powershell
# Docker Desktop 재시작
& "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
# 또는 메모리 한도 줄이기 (compose.yml 50g → 32g)
```

### 학습 OOM (LoRA 모드)
- `batch_size: 2 → 1` (가장 효과 큼)
- `enable_gradient_checkpointing: false → true` (~2GB 절감, 시간 +25%)
- `lora_r: 32 → 16` (params 절반)
- `target_modules`에서 `to_out.0` 제거

### LoRA 학습 시작 안 됨 — PEFT import 실패
```bash
docker exec dubbing_pipeline /opt/venv_lipsync/bin/pip install --force-reinstall "peft==0.10.0"
```
0.19+는 accelerate 0.34+ 필요 — 우리는 0.28이라 0.10 고정.

### bitsandbytes 안 됨
```bash
docker exec dubbing_pipeline /opt/venv_lipsync/bin/pip install --force-reinstall bitsandbytes
```

### LoRA 학습 후 추론 시 weight 불일치
- 학습 시 사용한 r/alpha/target_modules와 `lora_merge.py` 인자 일치시켜야 함
- 학습: `lora_r: 32, lora_alpha: 16, lora_target_modules: [to_q, to_k, to_v, to_out.0]`
- merge: `--r 32 --alpha 16 --target to_q to_k to_v to_out.0`

### 컨텍스트 손실 시 재시작
1. 이 PROJECT_STATE.md 읽기 (특히 "학습 명령어" 섹션)
2. `docker compose up -d` (컨테이너 살리기)
3. 진행 상황은 `media/aihub_processed/`, `media/training_outputs/`, `media/lora/` 확인
4. 학습 로그: `logs/` 폴더 확인

---

## 📌 사용자 결정 사항 (중요)

1. ✅ **MuseTalk 거절** (Day 3 오후) — 너무 부정확. LatentSync 우선
2. ✅ **LatentSync 학습 시도** (속도 < 품질) — Partial FT 검증 후 LoRA로 전환
3. ✅ **노트북 5080 사용 (cloud GPU 비사용)** — 시간 오래 걸려도 OK
4. ✅ **무음(noise1) + 다양한 demographics** 받기
5. ✅ **측면 angle도 학습에 포함** (단, 비율 조정)
6. ✅ **다운로드 + .tar 풀기 100% 완료** (TS1, 10, 20, 30, 40, 50 모두 완전 / TS174 mp4 추출됨). 디스크: **517.41GB free**
7. ✅ **LoRA over Partial FT** — 다국어 확장 + 베이스 보존 + 파일 작음
8. ✅ **MuseTalk 완전 삭제** + 폴더 구조 정리
9. ✅ **마스킹 개선 코드 작성** — soft mask + GFPGAN 후처리 (사용자 보고 이슈 해결)

---

## 🌐 다국어 확장 (4주 활용)

지금 인프라는 한국어 외 언어로 학습 가능 (코드 변경 무):

| 언어 | 데이터 소스 후보 | 필요 시간 | 학습 시간 |
|---|---|---|---|
| 🇰🇷 한국어 (현재) | AIHub 538 | 50h | ~8h |
| 🇯🇵 일본어 | JTubeSpeech | 5~15h | ~3~6h |
| 🇪🇸 스페인어 | TEDx + VoxCeleb2 | 10~20h | ~5~8h |
| 🇫🇷 프랑스어 | TEDx + VoxCeleb2 | 15~25h | ~6~10h |

학습 후 `cp checkpoint.pt media/lora/latentsync_<lang>.pt` 만 하면 자동 사용.

---

## 📊 5월 2일 검증 결과 요약

| 항목 | 검증 |
|---|---|
| Partial FT 1000 step | ✅ peak 15.84GB, 12분 27초 |
| LoRA r=16 1000 step | ✅ peak 8.4GB, 17분 23초, loss 0.000621 |
| LoRA+ r=32 + bs=2 30step | ✅ peak 14.5GB, 33초 (1.10s/it) |
| Soft mask 적용 | ✅ Gaussian feather 50px |
| GFPGAN 후처리 (1728frames@1080p) | ✅ 21분, 마스크 거의 사라짐, 디테일 복원 |
| MuseTalk 제거 + 폴더 정리 | ✅ scripts/, configs/, patches/, logs/ |

---

**문서 끝.** 컨텍스트 잘려도 이 파일 읽으면 바로 이어서 진행 가능.
