# CosyVoice3 TensorRT-LLM: instruct2 포팅 + 3080(Ampere) 호환성

작성 2026-06-03. 대상: CosyVoice3-0.5B LLM(Qwen2ForCausalLM)을 TensorRT-LLM으로
가속하는 `CosyVoice/runtime/triton_trtllm` 런타임.

## 1. 3080(Ampere sm_86)에서 TRT를 쓸 수 있나? — 쓸 수 있다 (단, 각자 빌드)

**핵심: 컴파일된 `.engine`은 GPU 아키텍처 종속이라 공유 불가. 체크포인트는 이식 가능.**

- 현재 레포의 prebuilt 엔진(`trt_engines_bfloat16/rank0.engine`)은 **sm_120(RTX 5080/Blackwell)**
  용으로 빌드됨. TensorRT-LLM은 다른 SM용 엔진 로드를 거부하므로 **3080(sm_86)에선 못 씀**.
- 그러나 LLM 가중치 체크포인트(`hf_cosyvoice3_llm/model.safetensors`)는 아키텍처 무관·이식 가능.
  3080 머신에서 아래 2단계로 **자체 엔진을 빌드**하면 된다.
- bfloat16은 Ampere(sm_80/86)에서 지원되고 TensorRT-LLM 0.20.0도 Ampere를 지원하므로
  정밀도/버전 문제 없음. (BF16이라 음질·화자분리 영향 0 — zero-shot/instruct 동일)

### 3080 빌드 절차 (`run_cosyvoice3.sh` 기준)
```bash
# (1) HF 체크포인트 → TRT-LLM 체크포인트 형식
python3 scripts/convert_checkpoint.py \
    --model_dir  hf_cosyvoice3_llm \
    --output_dir trt_weights_bfloat16 \
    --dtype bfloat16

# (2) TRT-LLM 체크포인트 → 이 GPU(sm_86)용 .engine 컴파일
trtllm-build --checkpoint_dir trt_weights_bfloat16 \
             --output_dir trt_engines_bfloat16 \
             --max_batch_size 64 \
             --gemm_plugin bfloat16
```
→ 산출된 `trt_engines_bfloat16/rank0.engine`이 그 3080 전용. 다른 GPU와 공유 금지.
필요 패키지: `tensorrt_llm==0.20.0`(엔진 빌드 버전과 일치해야 로드 가능), CUDA 12.x+,
드라이버는 Blackwell/Ampere 모두 지원하는 신버전.

> 정리: "팀원 3080도 TRT 사용 가능. 단 prebuilt 엔진을 복사하지 말고 위 (1)(2)를 3080에서 직접 실행해 각자 엔진 빌드."

## 2. instruct2 포팅 (감정 보존)

문제: 기존 TRT 경로는 zero-shot만 지원 → 감정 instruct 손실. 메인 CosyVoice CLI의
`inference_instruct2`와 동일 의미를 TRT-LLM 입력에 이식한다.

### 메인 CLI의 instruct2 메커니즘 (`cosyvoice/cli/frontend.py`)
`frontend_instruct2(tts_text, instruct_text, prompt_wav)` =
`frontend_zero_shot(tts_text, prompt_text=instruct_text, prompt_wav)` 호출 후
`llm_prompt_speech_token` **삭제**. 즉:
- LLM 입력: `prompt_text`=instruct_text(감정지시), prompt speech token **없음**.
- FLOW(token2wav) 입력: `flow_prompt_speech_token` + speaker embedding은 **참조오디오 그대로**(음색 유지).

### TRT-LLM 입력 매핑 (`offline_inference.py` / Triton `cosyvoice3/1/model.py`)
LLM 입력 토큰열:
```
zero-shot : <|sos|> + reference_text(참조전사) + target_text + <|task_id|> + prompt_speech_tokens
instruct2 : <|sos|> + instruct_text(감정지시) + target_text + <|task_id|> + (빈값)
```
포팅 = `reference_text → instruct_text`, assistant content(`prompt_speech_tokens_str`) → `""`.
**token2wav 호출은 변경 없음**(참조오디오의 prompt_speech_tokens/feat/spk_embedding 그대로 사용).

### Triton 백엔드 패치 (`model_repo_cosyvoice3/cosyvoice3/1/model.py`)
1. `config.pbtxt`에 optional 입력 `instruct_text`(TYPE_STRING, optional:true) 추가.
2. `_process_request_offline`/`_process_request_streaming`에서 instruct_text 읽기:
   ```python
   instruct_text = self._get_optional_instruct(request)  # 없으면 ""
   if instruct_text:
       llm_ref_text = ('You are a helpful assistant.<|endofprompt|>' + instruct_text
                       if '<|endofprompt|>' not in instruct_text else instruct_text)
       llm_prompt_tokens = torch.zeros((1, 0), dtype=prompt_speech_tokens_for_llm.dtype)
   else:
       llm_ref_text, llm_prompt_tokens = reference_text, prompt_speech_tokens_for_llm
   all_token_ids = await self.forward_llm_offline(target_text, llm_ref_text, llm_prompt_tokens)
   # token2wav 는 항상 참조오디오 prompt_speech_tokens/feat/spk_embedding 사용 (변경 없음)
   ```
   (`forward_llm_offline`의 `_convert_speech_tokens_to_str(빈 tensor)` → `""` 보장)

## 3. 검증 (PASS — 2026-06-03, RTX 5080 sm_120)
`CosyVoice/runtime/triton_trtllm/verify_instruct2_trt.py` — prebuilt 엔진을 `ModelRunnerCpp`로 로드,
동일 target("안녕하세요. 만나서 정말 반갑습니다.")에 서로 다른 instruct(기쁨/슬픔/무지시)를 주어
각각 유효한 speech token 생성 + instruct별 출력 상이를 확인.

결과:
- instruct2=기쁨/흥분 → 213 speech tokens, OK
- instruct2=슬픔/낮은톤 → 195 speech tokens, OK
- 대조(instruct 없음) → 53 speech tokens, OK
- 모든 케이스 유효 토큰 생성 = True, instruct별 출력 상이 = True → **PASS**.

즉 포팅한 instruct2 입력 포맷(reference_text=instruct_text + prompt speech token 생략)이
실제 TRT-LLM 엔진에서 유효하게 작동하고, instruct 가 생성을 실제로 좌우함을 확인했다.
(3080 은 §1 빌드 후 동일 스크립트로 검증.)

### ⚠ 런타임 GOTCHA 2개 (3080 포함 모든 환경에서 동일하게 겪음 — 검증 중 발견)
1. **cuda-python 버전**: `pip install tensorrt_llm==0.20.0` 가 최신 `cuda-python 13.x` 를 끌어오는데
   tensorrt_llm 0.20.0 은 구 API(`from cuda import cuda, cudart`)를 기대 → `ImportError: cannot import name 'cuda' from 'cuda'`.
   해결: `pip install "cuda-python==12.6.0"` (12.x 로 다운그레이드).
2. **NVIDIA_TF32_OVERRIDE**: 엔진은 이 env 미설정(-1)으로 빌드됐는데 컨테이너 실행환경엔 `=1` 로 설정돼 있으면
   `ICudaEngine::createExecutionContextWithoutDeviceMemory: Myelin ... Inconsistent setting of NVIDIA_TF32_OVERRIDE`
   로 execution context 생성 실패(엔진 로드 자체는 성공). 해결: 실행 시 `env -u NVIDIA_TF32_OVERRIDE` 로 unset
   (또는 엔진 빌드 때와 동일하게 맞춤). 진단용 daemon 들이 이 env 를 설정하므로 dub-half 와 분리 실행 권장.

설치/검증 환경(검증 통과 기준): venv(py3.12) + tensorrt_llm 0.20.0 + tensorrt 10.10.0.31 + torch 2.7.0 +
cuda-python 12.6.0, 드라이버 596.36, 엔진=trt_engines_bfloat16(sm_120).
