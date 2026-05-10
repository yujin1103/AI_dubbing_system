# 다국어 자동 더빙 시스템 — 발표 대본

> **PPT**: `PIPELINE_OVERVIEW.pptx` (14 슬라이드)
> **분량**: 약 18~22분
> **대상**: 처음 보는 사람 (시스템 구조 + 모델 역할 + 입출력)
>
> 각 슬라이드마다 **(말할 내용)**과 **(보조 설명, 질문 대응용)**으로 구분.

---

## 슬라이드 1 — 표지 (약 30초)

**(말할 내용)**

안녕하세요. 오늘 소개해드릴 시스템은 **다국어 자동 더빙 시스템**입니다. 영어 영상을 입력으로 받아서 한국어로 더빙하고, 입모양까지 맞춰주는 파이프라인입니다.

처음 들으시는 분도 어떤 모델이 어디서 무슨 일을 하는지 한눈에 파악하실 수 있도록, 5개 스테이지와 12개 모델의 흐름을 따라가면서 설명드리겠습니다.

**(보조 설명)**
- 시스템은 Docker 컨테이너 안에서 멀티 venv로 운영
- 메인 모델 3개는 daemon으로 메모리 상주

---

## 슬라이드 2 — 시스템 한눈에 (약 1분 30초)

**(말할 내용)**

전체 흐름은 5단계로 구성됩니다.

1단계는 **오디오 분리**입니다. 영상에서 오디오를 추출하고, 보컬과 배경음악을 분리합니다. 한국어로 더빙할 때 배경음악과 효과음은 그대로 살리고 사람 목소리만 한국어로 교체해야 하니까요.

2단계는 **누가 / 뭐라고 / 어떤 감정으로 말했는지** 분석하는 단계입니다. 받아쓰기, 화자 분리, 감정 인식이 동시에 일어납니다.

3단계는 화자별로 가장 깨끗한 음성 샘플을 골라내고, 영어를 한국어로 번역하는 단계입니다.

4단계는 골라낸 음성을 reference로 삼아서, 그 사람 목소리로 한국어를 합성합니다. 이걸 **zero-shot voice cloning**이라고 합니다.

5단계는 합성된 한국어 음성과 원본 배경음을 결합하고, 영상의 입모양을 한국어 발음에 맞춰 다시 그립니다.

전체로 보면 **12개 모델이 협력**해서 한 영상을 처리하고, 15초 영상 기준 약 17분 정도 소요됩니다. daemon 구조 덕분에 재실행 시 모델 로딩 1~2분이 절감됩니다.

**(보조 설명)**
- "Stage 1~3은 오디오 + 메타데이터 추출, Stage 4는 합성, Stage 5는 비디오 통합"이라고 정리 가능

---

## 슬라이드 3 — 사용 모델 인벤토리 (약 1분)

**(말할 내용)**

전체 모델 인벤토리입니다. 12개 모델이 각자의 역할을 맡고 있는데, 한 마디로 요약하면 다음과 같습니다.

- **BS-Roformer**는 보컬만 떼어냅니다.
- **Silero VAD**는 silence를 제거하고요.
- **Qwen3-ASR**는 받아쓰기를 합니다.
- **DiariZen**과 **ECAPA-TDNN**은 화자를 구분합니다. DiariZen은 일차 분리, ECAPA는 후처리로 over-detect를 합치고 outlier를 골라냅니다.
- **LightASD**는 영상에서 어느 얼굴이 말하고 있는지 감지합니다.
- **emotion2vec+**는 감정을 분류합니다.
- **MOS Evaluator**는 음성 품질을 평가해서 가장 깨끗한 reference를 골라냅니다.
- **VectorEngine LLM**은 영어를 한국어로 번역하면서 syllable 수까지 맞춥니다.
- **CosyVoice3**가 voice cloning으로 한국어를 합성하고요.
- 마지막으로 **LatentSync 1.6**과 **GFPGAN**이 입모양과 화질을 마무리합니다.

**(보조 설명, 질문 대응)**
- "왜 DiariZen에 ECAPA를 또 붙이나요?" → DiariZen이 같은 사람을 다른 화자로 over-detect하는 경우가 있어서, ECAPA centroid 거리로 자동 합쳐줍니다.
- "MOS는 학습된 모델인가요?" → 네, custom UTMOS-style 모델입니다. 1~5점으로 음성 품질 평가.

---

## 슬라이드 4 — Stage 1: 오디오 분리 (약 1분 30초)

**(말할 내용)**

첫 번째 스테이지인 오디오 분리부터 자세히 보겠습니다.

먼저 **ffmpeg**가 영상에서 오디오 트랙을 분리합니다. 입력은 mp4 영상이고 출력은 44.1kHz 스테레오 wav입니다.

이 wav를 **BS-Roformer**가 받아서 두 트랙으로 분리합니다. 하나는 사람 목소리만 있는 vocals.wav, 다른 하나는 배경음악과 효과음만 있는 bgm.wav입니다. BS-Roformer는 주파수 mask 기반 분리 모델인데, 노래에서 보컬을 분리할 때도 쓰이는 모델이라 정밀도가 좋습니다.

마지막으로 **Silero VAD v5**가 vocals.wav에서 실제 발화 구간만 골라냅니다. 말을 하지 않는 silence는 제거하고, 발화 구간 리스트도 함께 반환합니다.

여기서 **중요한 한계**가 하나 있습니다. BS-Roformer는 보컬과 배경을 분리할 뿐, **보컬 자체에 묻은 reverb는 제거하지 못합니다**. 드라마처럼 환경 변화가 큰 영상에서는 같은 화자가 컷마다 다른 reverb를 가지고 있어서, 뒤에 나올 ECAPA가 화자를 다르게 인식하는 원인이 됩니다. 이건 dereverb 단계를 추가하면 보완 가능합니다.

**(보조 설명)**
- BS-Roformer 대신 Demucs도 쓸 수 있지만 BS-Roformer가 노래 분리 task SOTA에 가까움
- vocals 분리 후에도 reverb는 voice에 묻어 있다는 점 강조

---

## 슬라이드 5 — Stage 2-1: ASR + 화자 분리 (약 1분 30초)

**(말할 내용)**

두 번째 스테이지는 가장 정보가 많이 만들어지는 단계입니다. 누가, 무슨 말을, 어떤 감정으로 했는지 한꺼번에 분석합니다.

먼저 **Qwen3-ASR-1.7B**가 받아쓰기를 합니다. 이 daemon은 8902 포트에 상주하고 있어서 매번 모델 로딩 시간을 절약합니다. 출력은 단어 하나하나에 대한 시작/끝 시각이 붙은 word list입니다. 단어 단위 timestamp가 있어야 나중에 화자 분리 결과랑 정확히 매칭할 수 있습니다.

다음은 **DiariZen**이 화자 분리를 합니다. 8903 포트의 daemon으로 동작하고, 음성 임베딩을 클러스터링해서 "누가 언제 말했는지"를 turn 단위로 반환합니다.

여기서 **ECAPA-TDNN**이 등장합니다. SpeechBrain의 192-dim speaker embedding 모델인데, DiariZen 결과를 검증하는 후처리 단계입니다. 같은 사람의 톤 변화로 분리된 화자들을 centroid 거리가 가까우면 자동으로 합치고, 짧은 outlier 발화는 가장 가까운 centroid로 재할당합니다.

여기 **한계**가 있습니다. **DiariZen은 1초 미만의 짧은 turn을 잘 못 잡습니다**. 빠른 화자 교차가 일어나는 드라마 같은 영상에서는 한 화자의 발화에 다른 화자의 짧은 끼어듦이 묻혀버립니다. 이걸 보완하기 위해 ECAPA sliding window split을 추가했습니다. segment 안에서 0.6초 윈도우로 sliding하면서 ECAPA 임베딩을 뽑고, 화자 변화 시점에서 추가 분할합니다.

**(보조 설명)**
- DiariZen은 pyannote 기반이지만 속도와 정확도 향상된 fork
- ECAPA centroid는 cosine similarity로 비교
- sliding window는 0.6초 길이, 0.2초 hop, min_consecutive=2 (1-window 깜빡임 무시)

---

## 슬라이드 6 — Stage 2-2: Visual + Emotion (약 1분 30초)

**(말할 내용)**

오디오만으로는 한계가 있어서, 영상 정보도 활용합니다.

**LightASD**는 active speaker detection 모델입니다. 영상의 매 프레임에서 어느 얼굴이 입을 움직이고 있는지, 그리고 그게 음성 트랙과 동기화되어 있는지 분석합니다. 출력은 face track별로 매 프레임의 score입니다. 양수면 발화 중, 음수면 입 닫힘.

**AV-Fusion**은 우리 시스템의 자체 알고리즘입니다. audio diarization과 visual ASD를 결합해서 세 가지 결과를 만듭니다. 첫째, **spurious 화자 제거** — 발화 시간이 짧고 face match도 없는 가짜 화자를 거릅니다. 둘째, **per-frame target** — 매 프레임에 lipsync를 적용할 face track을 결정합니다. 셋째, **speaker-face 매핑** — 어떤 audio 화자가 어느 face track과 매칭되는지 기록합니다.

**emotion2vec+ Large**는 audio 기반으로 감정을 분류합니다. segment 구간을 받아서 Neutral, Sad, Angry, Happy, Surprised, Scared 중 하나로 라벨링합니다. 이 결과는 두 가지 용도로 쓰입니다. 하나는 화자별 reference 선택 시 감정별로 따로 모으는 것, 다른 하나는 LLM 번역 prompt에 hint로 들어가서 감정에 맞는 한국어 표현이 나오도록 유도하는 것.

**한계**: 뒤를 돌아 말하거나 화면 밖에서 말하는 화자는 LightASD가 무력화됩니다. face가 안 보이니 visual 검증이 안 되죠. 이런 케이스는 ECAPA + 시간 인접성으로 보조하지만, 완벽하진 않습니다.

**(보조 설명)**
- LightASD 결과는 video hash 기반으로 캐싱 (재실행 시 1~2분 절감)
- speaker_face_map은 추후 화자 over-detect 자동 병합에도 활용 가능

---

## 슬라이드 7 — Stage 3: Reference + 번역 (약 1분 30초)

**(말할 내용)**

세 번째 스테이지에서는 음성 합성에 필요한 두 가지 재료를 만듭니다.

**MOS Evaluator**는 화자별 segment의 음성 품질을 1점부터 5점 사이로 평가하는 모델입니다. UTMOS-style의 custom 모델이고요. 같은 화자의 여러 segment 중에서 가장 깨끗한 것을 reference로 골라냅니다. 길이는 3~15초 범위에서 선택하는데, 이건 CosyVoice의 voice cloning에 적합한 길이입니다. 화자별 + 감정별로 best 한 개씩 추출합니다.

**VectorEngine LLM**은 영어 segment를 한국어로 번역하는데, 단순 번역이 아니라 **lip-sync에 맞는 한국어**를 만들어야 합니다. 그래서 prompt에 segment의 duration을 알려주고 syllable 수 가이드를 함께 줍니다. 예를 들어 영어 5초짜리 발화면 한국어로 약 27~33 syllable 정도가 적정인데, 이 범위를 벗어나면 합성 후 atempo로 늘이거나 줄여야 해서 음질이 손상됩니다.

batch=7로 한 번에 7개 segment를 번역합니다. 출력은 한국어 번역 텍스트와 영어 imperative 묘사가 같이 나옵니다. 영어 묘사는 CosyVoice에 emotion instruction으로 전달되는데, 너무 길면 모델이 그 영어를 그대로 음성으로 환각하는 문제가 있어서 짧은 카테고리 fallback만 사용합니다.

**핵심**: LLM이 한국어 syllable 수를 영어 발화 시간에 맞춰 정확히 만들 수 있다면, 뒤에 atempo 보정이 거의 필요 없어집니다. 이게 음질에 직접적인 영향을 줍니다.

**(보조 설명)**
- MOS 3.0 미만이면 robotic 위험 — 드라마처럼 reverb 강한 영상에서 자주 발생
- LLM은 cloud API 호출 (대용량 생성 모델)

---

## 슬라이드 8 — Stage 4: 한국어 합성 — CosyVoice3 (약 1분 30초)

**(말할 내용)**

네 번째 스테이지가 핵심 합성 단계입니다.

**CosyVoice3**, 정확히는 Fun-CosyVoice3-0.5B-2512 모델을 사용합니다. 이 모델은 **zero-shot voice cloning**이 가능합니다. 즉, 5초 정도의 reference 음성만 있으면 별도 학습 없이 즉시 그 사람 목소리를 흉내 낼 수 있습니다. **cross-lingual** 모드를 쓰면 영어 reference로도 한국어를 합성할 수 있고요.

입력은 세 가지입니다. tts_text는 표준 prefix와 한국어 텍스트를 합친 것이고, prompt_wav는 16kHz mono로 변환된 reference, speed는 발화 속도 조절 파라미터입니다. 출력은 24kHz wav입니다.

여기서 prefix가 중요한 역할을 합니다. `<|endofprompt|>` 토큰이 들어 있는데, 이게 모델이 instruction과 합성할 텍스트를 구분하는 마커입니다. 이전에는 LLM이 만든 긴 영어 묘사를 prefix에 넣었는데, 그게 한국어 합성 결과에 영어 단어로 누출되는 문제가 있어서 짧은 카테고리 fallback만 사용하도록 변경했습니다.

합성 후에는 ffmpeg atempo로 lip-sync 길이 보정을 합니다. 한국어가 영어 발화 시간보다 짧으면 0.85배까지 늘리고, 길면 1.25배까지 압축합니다. 이 보정이 자주 일어나면 pitch가 변형돼서 robotic해지므로, 앞 단계의 LLM 번역이 정확할수록 음질이 좋아집니다.

**한계**: 드라마처럼 reference 자체에 reverb나 background noise가 묻어 있으면 voice cloning이 robotic하게 나옵니다. TED 강연이나 인터뷰처럼 clean recording에 가장 적합합니다.

**(보조 설명)**
- daemon 8901 포트에 상주
- 합성 시간은 segment당 약 1~3초
- inference_cross_lingual / inference_zero_shot / inference_instruct2 등 여러 모드 있음

---

## 슬라이드 9 — Stage 5: 영상 결합 + 립싱크 (약 1분 30초)

**(말할 내용)**

마지막 다섯 번째 스테이지는 비디오 결합입니다.

먼저 **ffmpeg mix**가 한국어 더빙 wav와 원본 배경음 bgm을 결합합니다. 단순 합성이 아니라 EBU R128 표준에 따라 loudnorm을 적용해서 -23 LUFS의 표준 dialogue level로 정규화합니다. 이렇게 하면 더빙 음량이 너무 크거나 작아지는 현상을 방지할 수 있습니다.

다음이 **LatentSync 1.6**입니다. 입모양 동기화 모델인데, VAE 기반의 video diffusion 모델입니다. 한국어 음성에 맞춰서 입모양을 다시 그려냅니다. DDIM 20 steps로 추론하고요. 한국어 발음에 더 정확한 LoRA를 별도로 학습해서 사용할 수도 있습니다. 우리 시스템에는 AIHub 588개 영상으로 50k step 학습한 한국어 LoRA가 4.10GB 크기로 들어 있습니다.

마지막은 **GFPGAN v1.4**입니다. LatentSync 결과의 face 디테일을 복원하고 2배로 업스케일합니다. async I/O 파이프라인을 적용해서 GPU가 idle하지 않도록 했고, 처리 시간 22%를 절감했습니다.

**한계**: LatentSync는 RTX 5080의 sm_120 아키텍처에서 가속이 거의 불가능합니다. DeepCache, torch.compile, channels_last, SageAttention 등 14가지 가속 시도가 모두 실패했어요. 그래서 LatentSync 자체는 그대로 두고 후처리만 가속하고 있습니다.

**(보조 설명)**
- 한국어 LoRA는 PEFT 라이브러리 + LoRA+ (A/B 분리 lr) + 8-bit AdamW로 학습
- GFPGAN async는 prefetch=4 reader thread + write_pool=2

---

## 슬라이드 10 — Daemon 구조 (약 45초)

**(말할 내용)**

성능 최적화에서 중요한 부분이 daemon 구조입니다.

CosyVoice3, Qwen3-ASR, DiariZen 세 모델은 **로딩 시간이 60~90초**로 깁니다. 매 영상마다 새로 로딩하면 비효율적이죠. 그래서 이 세 모델을 각각 8901, 8902, 8903 포트에 daemon으로 띄워놓고 메모리 상주시킵니다.

orchestrator는 daemon이 살아 있는지 health check 후, 살아 있으면 HTTP로 호출하고 없으면 inline으로 fallback합니다.

이 구조 덕분에 100초 영상 처리 시 약 2~3분이 절감됩니다. 대량 처리에 큰 효과죠.

**(보조 설명)**
- daemon은 FastAPI + uvicorn 기반
- `--smart-daemon` 옵션 사용 시 자동 활용
- 모델 변경 시 daemon restart 필요

---

## 슬라이드 11 — 멀티 venv 구조 (약 45초)

**(말할 내용)**

또 하나 중요한 구조가 멀티 venv입니다.

**문제**: 각 모델의 의존성 버전이 충돌합니다. 예를 들어 transformers 버전이 ASR과 LatentSync 사이에서 다르고, pyannote와 basicsr도 충돌이 있습니다.

**해결**: Docker 컨테이너 안에 4개의 독립적인 venv를 운영합니다.
- `venv_lipsync`: 메인 — orchestrator, LatentSync, CosyVoice
- `venv_asr`: Qwen3-ASR 전용
- `venv_diarizen`: DiariZen 전용
- `venv_gfpgan`: GFPGAN과 후처리

orchestrator가 subprocess로 다른 venv를 호출하거나 HTTP로 daemon에 호출합니다. 의존성 충돌 없이 모든 모델을 한 컨테이너에서 운영할 수 있게 됩니다.

**(보조 설명)**
- 컨테이너 이미지 56GB (모든 의존성 포함)
- venv별 Python 3.12 통일

---

## 슬라이드 12 — 성능 측정 (약 1분)

**(말할 내용)**

15초 영상 기준으로 단계별 처리 시간을 측정한 결과입니다.

가장 비중이 큰 단계는 **LatentSync 1.6**의 30%입니다. 입모양 동기화는 GPU bound이고 가속이 어렵습니다. 그 다음은 GFPGAN async가 21%, DiariZen + AV-Fusion이 12%, CosyVoice TTS가 12%입니다.

전체 합계는 17분 15초입니다.

100초 영상은 화자 수와 발화 밀도에 따라 다르지만, 35분에서 80분 정도 소요됩니다. 신기하게도 multi-speaker 영상이 빠른데, 짧은 segment가 많아서 LatentSync 처리 단위가 잘게 쪼개지기 때문입니다.

이 시스템은 사전 처리 후 시청 use case에 적합하고, 실시간 처리는 불가능합니다.

**(보조 설명)**
- 시간 측정은 RTX 5080 기준
- daemon cold start 1:30은 별도

---

## 슬라이드 13 — 알려진 한계 (약 1분 30초)

**(말할 내용)**

마지막으로 시스템의 한계와 회피 방법을 정리하겠습니다.

첫째, **드라마 reverb 잔존**입니다. BS-Roformer는 vocal-BGM 분리만 하고 reverb는 제거하지 못합니다. 그래서 ECAPA가 같은 화자를 다른 사람으로 인식하기도 하고, voice cloning 결과가 robotic해지기도 합니다. 회피 방법은 CPU 기반 dereverb 단계를 추가하는 것입니다. DeepFilterNet 같은 모델이면 GPU 메모리를 추가로 쓰지 않고도 적용 가능합니다.

둘째, **DiariZen이 1초 미만 짧은 turn을 못 잡는** 문제입니다. 빠른 화자 교차에서 segment가 합쳐지는데, ECAPA sliding window split으로 보완하고 있습니다.

셋째, **등을 돌리거나 화면 밖에서 말하는 화자**는 LightASD가 무력화되어 AV-Fusion 보조가 안 됩니다. ECAPA와 시간 인접성으로 보조하지만 완벽하지는 않습니다.

넷째, **LatentSync 가속 불가**. sm_120 Blackwell GPU에서 모든 가속 시도가 실패했고, 후처리만 GFPGAN async로 가속 가능합니다.

다섯째, **LoRA cheek pink artifact**. 한국어 LoRA 학습 시 num_frames mismatch로 광대뼈 영역에 미세한 색감 시프트가 있습니다. nf=4 재학습이나 cloud A100에서 nf=16 학습으로 해결 가능합니다.

**(보조 설명, 질문 대응)**
- "dereverb 추가하면 GPU 메모리 더 필요한가요?" → DeepFilterNet은 CPU 가능, 메모리 0 추가
- "회상 씬의 동굴 reverb 같은 특수 효과는?" → 그건 voice cloning 후 ffmpeg aecho 등으로 별도 적용 가능

---

## 슬라이드 14 — 시스템 사양 + 결론 (약 1분)

**(말할 내용)**

마무리로 시스템 사양과 핵심 정리입니다.

개발 환경은 Windows 11, Docker Desktop, RTX 5080 16GB GPU, 63GB RAM입니다. AIHub 데이터 510GB까지 포함해서 1.9TB 스토리지를 사용했습니다.

**핵심 메시지** 여섯 가지입니다.

첫째, 5스테이지 12모델이 협력해서 완전 자동 더빙을 수행합니다.

둘째, Docker 멀티 venv로 의존성 충돌을 격리했습니다.

셋째, 3개 daemon이 메모리 상주해서 재실행 시 시간이 절감됩니다.

넷째, TED, 강의, 인터뷰처럼 clean recording 영상에 최적화되어 있습니다.

다섯째, 드라마는 reverb 한계로 robotic 경향이 있지만 dereverb로 보완할 수 있습니다.

여섯째, 한국어 LoRA로 입모양 정확도를 높였고, SyncNet 1.86점을 달성했습니다.

이상이 다국어 자동 더빙 시스템의 전체 구조입니다. 자세한 내용은 PIPELINE_OVERVIEW.md 문서에 정리되어 있습니다. 질문 부탁드립니다.

**(보조 설명, 예상 질문)**
- "다른 언어로도 확장 가능한가요?" → `latentsync_<lang>.pt` 자동 인식 인프라 있음. 일본어/스페인어 LoRA 학습 후 파일만 복사하면 됨
- "드라마용 모델은 따로 만들 계획?" → dereverb 통합 검토 중

---

## 발표 팁

| 팁 | 설명 |
|---|---|
| **눈맞춤** | 각 stage 슬라이드에서 input → process → output 흐름은 손으로 가리키며 |
| **속도** | 모델명은 천천히, 입출력은 약간 빠르게 (정보 밀도가 높으니) |
| **강조** | "한계" 슬라이드에서는 살짝 톤 낮춰 솔직하게 |
| **Q&A 대비** | 슬라이드 4~9 한계 박스 내용 다 외워둘 것 |
| **시간 안배** | Stage 4 (CosyVoice)와 Stage 5 (LatentSync)는 핵심이라 1분 30초 풀로 |

---

## 슬라이드별 시간 배분 요약

| # | 제목 | 분량 |
|---|---|---|
| 1 | 표지 | 0:30 |
| 2 | 5단계 한눈에 | 1:30 |
| 3 | 모델 인벤토리 | 1:00 |
| 4 | Stage 1 오디오 분리 | 1:30 |
| 5 | Stage 2-1 ASR + Diarization | 1:30 |
| 6 | Stage 2-2 Visual + Emotion | 1:30 |
| 7 | Stage 3 Reference + 번역 | 1:30 |
| 8 | Stage 4 CosyVoice3 | 1:30 |
| 9 | Stage 5 영상 결합 + 립싱크 | 1:30 |
| 10 | Daemon 구조 | 0:45 |
| 11 | 멀티 venv 구조 | 0:45 |
| 12 | 성능 측정 | 1:00 |
| 13 | 알려진 한계 | 1:30 |
| 14 | 결론 | 1:00 |
| **합계** | | **18:00** |

Q&A 포함 약 22~25분 권장.

---

**문서 끝.** PPT 슬라이드 진행과 같이 읽으면 자연스럽게 흐릅니다.
