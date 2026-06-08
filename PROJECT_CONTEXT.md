# PROJECT CONTEXT — 현재 상태 핸드오프

**작성 2026-06-08** · 브랜치 `restructure-modular` · 검증 주체: Claude Code 세션
이 문서가 런타임/경로에 대한 **정본**입니다. `NEXT_SESSION.md`(2026-05-29)는 마운트·모델캐시 관련 서술이 **낡았으니** 이 문서를 우선하세요.

---

## 0. 폴더/런타임 핵심 (중요)

- **단일 프로젝트 루트 = `E:\Capstone_dub`** (190GB). 코드 + git + 모델캐시 전부 자족.
- **`E:\TTS_capstone`(옛 158GB 백업)은 2026-06-08 영구 삭제됨.** 삭제 전 유일 고유 파일 2개(`docs/preserved/algorithm_originals/_build_speakface.py`, `_sep_S2.py`)는 본 repo로 이관·푸시 완료(커밋 `ed4a3ef`). 나머지는 전부 중복/재다운로드 가능했음.
- **컨테이너 마운트는 이미 Capstone_dub 자신**: `docker-compose.yml`의 `./:/workspace/project`. `NEXT_SESSION.md`의 "Capstone_dub로 재마운트 금지(model_cache 없음)" 경고는 **무효** — model_cache(66G)가 여기 있음.

### 실행 방법
```powershell
# 컨테이너(모든 venv+모델 포함 dubbing_pipeline:full)를 controller 로 기동.
# base(docker-compose.yml) + override(docker-compose.override.yml) 자동 병합.
docker compose up -d controller
# 파이프라인 실행 (controller 안에서)
docker compose exec controller python src/pipeline.py --config configs/cosyvoice3-docker-draft.json ...
```
- 이미지 baked env 는 옛 경로(`/workspace/media/...`, `/project` 누락)라 **반드시 override 가 적용된 채로 실행**해야 함. `docker run` 수동 기동 시 override env 를 직접 넣어야 재다운로드 안 남.
- GPU juggling: `src/daemon_lifecycle.py` 가 step별로 diarize(8903)·asr(8902) 데몬만 ON, 나머지는 OFF. 데몬은 `env=os.environ.copy()` 로 떠서 controller 의 교정된 env 를 **상속**함.

---

## 1. 모델 경로 전수 검증 (2026-06-08, 컨테이너 내부 실측)

활성 config = `configs/cosyvoice3-docker-draft.json`. 실효 env 는 override 가 모두 `/workspace/project/...` 로 교정.

| 모델 | 로더가 찾는 경로 | 상태 |
|---|---|---|
| ASR Qwen3-ASR-1.7B | `.hf-cache/models--Qwen--Qwen3-ASR-1.7B/...` (config `asr`) | ✅ snapshot→blobs 정상, 재다운로드 X |
| TTS CosyVoice3-0.5B | `media/model_cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (18G) | ✅ |
| emotion2vec+ large | `MODELSCOPE_CACHE/hub/iic/emotion2vec_plus_large` (1.9G) | ✅ 오프라인 로드 0.1s |
| 음원분리(BS-RoFormer 등) | `models/separation` (1.1G, config `model_dir`) | ✅ |
| silero VAD | `models/vad/silero_vad.jit` (config `model_path`) | ✅ |
| insightface buffalo_l | `/root/.insightface/models/buffalo_l` (326M, **이미지 baked**) | ✅ onnx 5종 존재 |
| LightASD weight | `/opt/Light-ASD/weight` (**이미지 baked**) | ✅ |
| ERes2NetV2 | `media/model_cache/eres2netv2/...` | ✅ **픽스 후** (아래 §2) |
| MOS evaluator | `media/model_cache/mos_model/best.pt` (1.1G) | ✅ (MOS 채점 시) |
| torch.hub | `media/model_cache/torch/hub` (1.6G) | ✅ |

오프라인 로드 실증(`HF_HUB_OFFLINE=1`): whisper large-v3(1542M) · HF wav2vec2-base · modelscope emotion2vec 모두 마운트 캐시에서 로드 확인.

---

## 2. 적용한 픽스 — ERes2NetV2 재다운로드 차단

**증상(지난번 재다운로드 원인 중 하나):** `src/daemons/eres2netv2_helper.py` 의
`LOCAL_CACHE` 기본값이 옛 경로 `/workspace/media/model_cache/eres2netv2`(`/project` 누락).
repair_patches(`gap_fill.py`·`focused_nemo_split.py`·`face_identity_split.py`)와 `auto_refine`·`auto_thr_decision`
가 `get_eres2netv2_model()` 호출 → 옛 경로에 파일 없음 → `snapshot_download` 로 **208MB 재다운로드**.
(config `cosyvoice3-docker-draft.json` 의 `diarization.eres2netv2.enabled=true` 라 활성 경로임.)

**픽스:** `docker-compose.override.yml` controller env 에 추가 —
```yaml
ERES2NETV2_CACHE_DIR: /workspace/project/media/model_cache/eres2netv2
```
**검증:** override 만으로 helper 가 올바른 경로 사용 → 8.2s 캐시히트, 208MB 재전송 없음(컨테이너 실측).
실제 캐시에 `iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.ckpt` + modelscope 메타(`.msc/.mdl`) 존재 → modelscope 가 캐시로 인식.

---

## 3. 알려진 잠재 이슈 (현재 활성 경로엔 영향 없음, 추후 정리 권장)

- **`src/daemons/campplus_helper.py:21`** — `campplus.onnx` 경로가 옛 경로로 하드코딩(`/workspace/media/...`, env 아님).
  실제 데이터는 `/workspace/project/media/.../campplus.onnx`(27M)에 있음. **활성 diarize 는 DiariZen(8903)** 이라 campplus 미사용 → 무영향.
  campplus 변형을 쓰려면 코드 수정 필요(리터럴이라 env 로 못 고침).
- **`src/daemons/eres2netv2_helper.py` / `patches/eres2netv2_helper.py`** — 코드 기본값 자체가 옛 경로.
  override env 로 가렸지만, compose 밖 수동 실행 대비 코드 기본값도 `/project` 경로로 고치면 더 견고.
- **`src/preserved_orchestrator/orchestrator.py`** — 옛 통짜 orchestrator. speechbrain ECAPA(`/workspace/media/.../speechbrain/ecapa`),
  audio_separator, `/opt/LatentSync/checkpoints/latentsync_unet.pt` 등 옛/이미지 경로 다수. **새 `pipeline.py` 는 이걸 안 씀** → 무영향(참고용 보존).
- **lipsync(LatentSync/MuseTalk)** — `pipeline.py` 에 lipsync 단계 **없음**. 마운트에 캐시(latentsync 9.6G·musetalk 8G)는 있으나 현재 모듈 파이프라인에선 미사용.
  `/opt/LatentSync/checkpoints/latentsync_unet.pt` 는 이미지에 없음(MISSING) — lipsync 를 다시 도입하면 경로 재점검 필요.

---

## 4. 재다운로드 0 보장 체크리스트 (새 영상 돌리기 전)

1. `docker compose up -d controller` (override 포함) 로 기동 — 단독 `docker run` 금지(또는 override env 직접 주입).
2. 활성 config 의 `asr` 가 `/workspace/project/.hf-cache/...` (또는 상대경로)인지 확인. **옛 `configs/preserved/*.json` 은 옛 경로라 그대로 쓰지 말 것.**
3. eres2netv2 픽스(§2)가 override 에 있는지 확인.
4. 의심되면: `docker compose exec controller bash -lc 'export HF_HUB_OFFLINE=1; <로드>'` 로 오프라인 강제 → 캐시 미스면 즉시 에러(다운로드 대신)로 드러남.
