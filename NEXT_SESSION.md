# 다음 세션 핸드오프 — 화자 분리 정확도 (restructure-modular)

**작성 2026-05-29** · 브랜치 `restructure-modular` (E:\Capstone_dub) · 커밋 `6e9f549` 이후

---

## 0. 환경 (중요)

- **코드/git = `E:\Capstone_dub`**, **런타임 컨테이너 = `E:\TTS_capstone` 마운트(`/workspace`)**. 코드 편집은 Capstone_dub, 실행 전 `cp`로 `E:\TTS_capstone\src\...`에 동기화. (컨테이너를 Capstone_dub로 재마운트 금지 — model_cache 없어 모델 재다운로드됨.)
- 컨테이너: `dubbing_pipeline` (up). 컨테이너 내 명령은 PowerShell 또는 `MSYS_NO_PATHCONV=1` + `docker exec ... bash -lc "..."` (git-bash 경로 변환 회피).
- **LightASD는 `/opt/venv_lipsync/bin/python`에서만 동작** (python_speech_features 보유). repair patch/face_clustering은 `/opt/venv_diarizen/bin/python`.
- daemon 포트(실제): **8903=단일 DiariZen**, 8918=fusion(DZ+NeMo), 8923=NeMo, 8943=pyannote-3.1(model_loaded:false), 8901=cosy, 8902=ASR. daemon들은 `/workspace/patches/`(구경로)에서 실행 중.
- 테스트 run_dir: test4 = `media/runs/20260528_090246_test4fresh_74d347`, test5 = `media/runs/20260528_092242_test5fresh_86fb30`.

## 1. 이번 세션 성과 (커밋 6e9f549)

- **ASD 복구**: LightASD가 잘못된 venv로 조용히 죽어 repair 체인 no-op이던 것 수정. + repair_patch import 경로(patches→src/daemons) 4개, face_cluster_match frames/scores IndexError, face_clustering 다중프레임 임베딩/_final 제외/cache pkl+report persist, config venv.
- **근본 원인 발견**: test4 과분할(9 SPK)은 알고리즘이 아니라 **저장된 raw `_segments.json`이 outlier(95/96/99) 오염 옛 버전**이었음. 지금 daemon 재diarize하면 단일DiariZen·fusion 둘 다 **깨끗한 6 SPK**.
- GT 수정: test4 6화자(woman1/man1/sean/paramedic/mom/dad, 2F·4M, 3장면).

## 2. 현재 최고 결과 (uniform 파라미터, 영상 무관)

**파이프라인**: fresh diarize(8918 fusion) → repair `--main-merge 0.99 --bg-merge 0.30 --sim-match 0.45 --pad 0.5` (per-video 튜닝 없이 통일).

| 영상 | score | main_count | 화자별 |
|---|---|---|---|
| **test4** | **1.1167** | 6/6 ✓ | man1·woman1·sean·dad·mom = 1.0, **paramedic 0.5** (잔여) |
| **test5** | **0.9381** | 3/4 | 아빠·BG = 1.0, 션 0.86, **엄마 0.67·의사 0.67** (잔여) |

재현(test4 예):
```
# 1) fresh raw로 교체 (8918 fusion → groups 형식) — stale raw 회피
# 2) face_clustering (venv_lipsync) → cache pkl + report + face_clusters
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python src/face_clustering.py <RD>/chunks <RD>/meta/<chunk>_segments.json \
  --out-face-clusters <RD>/meta/face_clusters.json --out-remapped <RD>/meta/diarization_face_matched.json \
  --venv-python /opt/venv_lipsync/bin/python
# 3) repair
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python src/apply_repair_patches.py <RD> --main-merge 0.99 --bg-merge 0.30 --sim-match 0.45 --pad 0.5
# 4) score
docker exec dubbing_pipeline python scripts/validate_against_gt.py <RD>/meta/<chunk>_segments_gapfilled.json media/gt/test4_gt.json
```

## 2b. ★ 2026-05-29 추가 측정 — pyannote-3.1 살림 + fusion anchor 한계 (다음 세션 1순위)

**pyannote-3.1(8943) 로드 성공 방법 (중요):**
- venv 잘못이 원인이었음. **`/opt/venv_pyann/bin/python`** + **`LD_LIBRARY_PATH=""`**(번들 cuDNN 9.20 강제; 시스템 9.19와 충돌 회피)로 띄워야 로드됨. venv_diarizen은 lightning 불일치(PyanNet.load_from_checkpoint), venv_pyann 기본은 cuDNN 충돌.
- 정상 launch:
  ```
  pkill -f pyannote_diarize_daemon.py
  PYANNOTE_MODEL=pyannote/speaker-diarization-3.1 HF_HOME=/workspace/media/model_cache/huggingface \
  HF_TOKEN=<token> LD_LIBRARY_PATH="" \
    nohup /opt/venv_pyann/bin/python /workspace/src/daemons/pyannote_diarize_daemon.py --port 8943 &
  ```
  (파일은 `/workspace/src/daemons/`에 있음. `/workspace/patches/`엔 없음.) GPU 필요 → cosy(8901)·asr(8902) 내려서 확보(현재 내려둠; **사용자 승인: cosy/asr는 별도 컨테이너로 분리**).

**측정 결과 (단일 모델 화자 수):**
| 영상 | DiariZen(8903) | NeMo | pyannote-3.1(8943) | 3-way fusion(8918) | GT |
|---|---|---|---|---|---|
| test4 | **6 ✓** | - | 3 ✗ | **6 ✓** | 6 |
| test5 | 3 ✗ | - | **4 ✓** | 3 ✗ | 4 |

**핵심 한계:** fusion(8918)의 canonical 화자는 **DiariZen+NeMo만으로 결정**(`fusion_diarize_daemon.py` line 152-159), pyannote는 canonical에 안 들어감 → DiariZen이 약한 test5에선 pyannote의 4번째가 버려짐.

**★ 다음 세션 1순위 FIX (영상 무관 uniform):** fusion canonical anchor를 **"distinct 화자를 가장 많이 찾은 모델"**로 변경. test4→DiariZen(6), test5→pyannote(4) 자동 선택. 구현: `fuse()`에서 DZ/NeMo/pyannote 각각 per-frame 배열 만들고 max-speaker 모델을 canon으로, 나머지는 temporal-nearest 매핑. **주의: test4 1.1167을 깨지 않도록 양쪽 재검증 필수.** (pyannote frame 배열은 이미 fetch됨 — line 285 pyannote2_segs.)

## 3. 남은 hard case (다음 세션 목표)

### 3a. test5 — 엄마/아빠 "Adam" 외침 분리 (4번째 화자)
- 5-7s에 엄마(여)·아빠(남)가 **짧은 텀 두고 순차** 외침. 현재 둘 다 `SPEAKER_02`(엄마+아빠+BG catch-all)로 묶임.
- **시도했으나 실패한 것**: `f0_intra_split` → test5는 main 3→4(1.1048)로 오르나 **test4를 0.8889로 망침**(과분할). 또 test5에서 엄마/아빠가 아니라 의사를 쪼갬 → uniform-safe 아님.
- **VAD-gap 분할 검증 결과**: 6.45s에 에너지 dip(텀)은 있으나 **F0가 양쪽 모두 고음(380~496Hz)** → 외침이라 성별 구분 안 됨. voice 임베딩도 짧아 sim<0.55. **음향 feature로 두 조각을 서로 다른 화자로 라벨링할 신호가 없음.**
- **다음 후보 접근**:
  1. **VAD-gap split + 화자 프로필 매칭**: 6.45s 텀에서 split 후, 각 조각을 **영상 내 다른 곳의 아빠 정상발화("Somebody call 911" 7-10s)·엄마 발화 프로필**과 비교해 라벨 (짧은 외침 자체가 아니라 같은 화자의 긴 발화로 centroid 구성). aaba 외침↔911 voice 비교가 핵심.
  2. **bimodal-F0 gate**: f0_intra_split을 "cluster 내 F0가 두 개의 뚜렷한 모드(gap≥80-100Hz, 각 모드 ≥2 seg)일 때만 split"으로 제한 → test4(단일화자 unimodal) 안 망치고 test5만. (단 위 검증상 외침 구간은 unimodal high라 이것도 안 먹힐 수 있음 — 외침 외 구간 포함해 재검토 필요.)
  3. **overlap-aware diarization**: pyannote-3.1(8943) model_loaded:false → 로드해서 overlap 검출 활성화하면 겹/연속 화자 분리 개선 가능. (4-way fusion 정상화.)

### 3b. test4 — paramedic (구급차, 0.5)
- 구급대원(face 34·35, 같은 사람)과 sean(0.6s "That's where we're going")이 `SPEAKER_04`로 병합. sean 짧은 발화 voice 오배정.
- face-split(①)은 **역효과**(34·35가 같은 사람이라 paramedic을 쪼갬). 짧은 발화 voice 한계.

## 4. 효과 없음으로 확인된 것 (반복 금지)
- **voice_safe_merge** thr 0.55까지 내려도 병합 0 (짧은 조각 voice sim<0.55).
- **ASD-dominant reassign** (`asd_dominant_reassign.py`): on-camera 조각만 흡수, off-camera 무력. test4 score 변화 없음.
- **boost_diarize**: test4 안 망치나(6 유지) test5 4번째도 회복 못 함 → 기본 비활성 권장.
- **face-split(①)**: 같은 사람 다중 face cluster라 역효과.
- per-video mm 튜닝(test4 0.99/test5 0.40): mm bimodal(≤0.45 과병합, ≥0.6 과분할) + 원칙 위배 → uniform mm=0.99 사용.

## 5. 미정리 작업
- run_dir들의 raw `_segments.json`은 fresh/boost로 교체된 상태(`.orig.json`/`.preboost.json` 백업 있음). 깨끗한 재현은 §2 절차.
- `asd_dominant_reassign.py` / `boost_diarize.py` / `f0_intra_split.py` / `face_cluster_merge.py`는 커밋됨(6e9f549)이나 체인 미통합(standalone). 효과 검증은 §3·§4 참조.
- README.md 대규모 재작성도 6e9f549에 포함.
