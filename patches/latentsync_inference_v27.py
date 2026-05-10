# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# v27 LIP_SYNC_FIX: BF16 강제 (Blackwell sm_120 native), VAE chunk 적용 가능하게 patch
#
# 변경:
#  - dtype: fp16 → bfloat16 (Blackwell native, fp16보다 안정적)
#  - num_frames=16 + resolution=512 (LoRA 안 씀, base 모델만)
#  - VAE chunk encode/decode (lipsync_pipeline.py에서 patch — 별도 파일)
#  - DeepCache는 --enable_deepcache 사용자 명시 시에만 (default OFF, 품질 우선)
#  - xformers + attention_slicing + vae_slicing/tiling (이전 patch 유지)

import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper


def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # === v27.1 DTYPE_PATCH: FP16 강제 (BF16 후처리 회색 마스크 bug 회피) ===
    # BF16 시도 결과: GitHub Issue #297, #328 동일 증상 — affine_transform.restore_img의
    #   `.to(dtype=torch.uint8)` 와 kornia.warp_affine이 BF16 silent failure → 회색 마스크 visible.
    # LatentSync 1.6은 fp16으로 학습됨 → fp16이 가장 안정적.
    # 메모리 이득은 fp16==bf16 (둘 다 2 bytes/param), 속도 차이도 미미.
    # 환경변수 LATENTSYNC_DTYPE=bf16/fp16/fp32 로 강제 가능.
    forced = os.environ.get("LATENTSYNC_DTYPE", "").lower()
    if forced == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("[DTYPE_PATCH] forced bfloat16 (회색 마스크 bug 위험)")
    elif forced == "fp32":
        dtype = torch.float32
        print("[DTYPE_PATCH] forced fp32 (메모리 ↑↑)")
    elif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7:
        dtype = torch.float16
        print("[DTYPE_PATCH] dtype=float16 (LatentSync 학습 dtype, 안전)")
    else:
        dtype = torch.float32
        print("[DTYPE_PATCH] no GPU half precision → fp32")

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    # === v27.8 SCHEDULER_PATCH: DPMSolverMultistepScheduler 옵션 ===
    # LATENTSYNC_SCHEDULER=dpm 으로 같은 품질에 더 빠른 수렴 (steps 줄여도 품질 유지)
    scheduler_type = os.environ.get("LATENTSYNC_SCHEDULER", "ddim").lower()
    if scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_pretrained("configs")
        print("[SCHEDULER_PATCH] DPMSolverMultistep (faster convergence)")
    else:
        scheduler = DDIMScheduler.from_pretrained("configs")
        print("[SCHEDULER_PATCH] DDIM (default)")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    # === v27.2 VAE_FP32_PATCH: VAE만 fp32 (UNet은 fp16) ===
    # 가설: vae_slicing/tiling이 fp16에서 boundary precision 손실 → 회색 마스크.
    # VAE 가중치는 ~80MB (fp32 기준 320MB) → GPU 부담 미미.
    # UNet은 fp16 그대로 (메모리 절감 핵심).
    # 환경변수 LATENTSYNC_VAE_DTYPE=fp16/fp32 강제 가능.
    vae_dtype_str = os.environ.get("LATENTSYNC_VAE_DTYPE", "fp16").lower()  # default fp16 (UNet과 일치)
    if vae_dtype_str == "fp16":
        vae_dtype = torch.float16
    elif vae_dtype_str == "bf16":
        vae_dtype = torch.bfloat16
    else:
        vae_dtype = torch.float32
    # === VAE 선택 (LATENTSYNC_VAE_VARIANT=mse|ema) ===
    # mse (default, 학습 시 사용된 VAE) — 안전
    # ema — 더 sharp, 미세 디테일 ↑, 약간의 호환 위험
    vae_variant = os.environ.get("LATENTSYNC_VAE_VARIANT", "mse").lower()
    vae_repo = "stabilityai/sd-vae-ft-ema" if vae_variant == "ema" else "stabilityai/sd-vae-ft-mse"
    print(f"[VAE_VARIANT] {vae_repo}")
    vae = AutoencoderKL.from_pretrained(vae_repo, torch_dtype=vae_dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    print(f"[VAE_FP32_PATCH] vae_dtype={vae_dtype}, unet_dtype={dtype}")

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # === v27.10 CHANNELS_LAST_PATCH: UNet conv 가속 (cuDNN NHWC kernel 활용) ===
    # LATENTSYNC_CHANNELS_LAST=1 로 활성화. warmup 없음, 매 chunk 일정한 -5~15% 속도.
    # 위험: 일부 op가 channels_last 미지원 시 fallback (그냥 NCHW로 복귀, harmless).
    # sm_120 호환: cuDNN NHWC는 Ampere+ 부터 정식 지원, Blackwell 안전.
    if os.environ.get("LATENTSYNC_CHANNELS_LAST", "0") == "1":
        try:
            pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)
            print("[CHANNELS_LAST_PATCH] UNet → channels_last (NHWC)")
        except Exception as e:
            print(f"[CHANNELS_LAST_PATCH] failed: {e}")

    # === v27.9 COMPILE_PATCH: UNet torch.compile (첫 chunk warmup 60-90초, 이후 30-50% ↑) ===
    # LATENTSYNC_COMPILE=1 으로 활성화. PyTorch 2.0+ 필요. 실제 효과: 약 -2분 (15초 영상 기준)
    if os.environ.get("LATENTSYNC_COMPILE", "0") == "1":
        try:
            print("[COMPILE_PATCH] torch.compile UNet (첫 chunk warmup ~60-90초)")
            pipeline.unet = torch.compile(
                pipeline.unet,
                mode=os.environ.get("LATENTSYNC_COMPILE_MODE", "reduce-overhead"),
                fullgraph=False,  # LatentSync UNet 동적 shape (안전)
            )
            print("[COMPILE_PATCH] applied")
        except Exception as e:
            print(f"[COMPILE_PATCH] failed: {e}")

    # === v27.4 ATTENTION_PATCH: SDPA backend 환경변수로 컨트롤
    # LATENTSYNC_ATTN: xformers / mem_efficient / math / sageattention (default: mem_efficient)
    # sageattention = SageAttention 2.2.0 (sm_120 Blackwell 전용, ~30% diffusion 가속)
    import torch as _t
    import torch.nn.functional as _F
    _t.backends.cuda.enable_flash_sdp(False)            # sm_120 미지원
    attn_backend = os.environ.get("LATENTSYNC_ATTN", "mem_efficient").lower()
    if attn_backend == "sageattention":
        # === v27.11 SAGEATTENTION_PATCH (5/7): F.sdpa monkey-patch ===
        # 5/7 fix v3: SA가 head_dim=512 미지원 (LatentSync cross-attn) → 자동 fallback
        # 지원 head_dim (64/128)만 SA 사용, 나머지는 original SDPA
        try:
            from sageattention import sageattn_qk_int8_pv_fp16_triton
            _F._original_sdpa = _F.scaled_dot_product_attention
            _sa_calls = [0, 0]  # [SA_used, fallback_used]

            def _sageattn_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                                is_causal=False, scale=None, enable_gqa=False):
                """SDPA → SageAttention (Triton) with auto-fallback on unsupported head_dim."""
                if dropout_p != 0.0:
                    return _F._original_sdpa(query, key, value, attn_mask=attn_mask,
                                              dropout_p=dropout_p, is_causal=is_causal,
                                              scale=scale, enable_gqa=enable_gqa)
                # head_dim check: SA Triton supports 64/128 typically
                head_dim = query.shape[-1]
                if head_dim not in (32, 64, 96, 128):
                    _sa_calls[1] += 1
                    return _F._original_sdpa(query, key, value, attn_mask=attn_mask,
                                              dropout_p=dropout_p, is_causal=is_causal,
                                              scale=scale, enable_gqa=enable_gqa)
                try:
                    _sa_calls[0] += 1
                    return sageattn_qk_int8_pv_fp16_triton(
                        query, key, value,
                        tensor_layout="HND",
                        is_causal=is_causal,
                        attn_mask=attn_mask,
                        sm_scale=scale
                    )
                except (ValueError, RuntimeError, AssertionError):
                    _sa_calls[0] -= 1
                    _sa_calls[1] += 1
                    return _F._original_sdpa(query, key, value, attn_mask=attn_mask,
                                              dropout_p=dropout_p, is_causal=is_causal,
                                              scale=scale, enable_gqa=enable_gqa)

            _F.scaled_dot_product_attention = _sageattn_sdpa
            _t.backends.cuda.enable_mem_efficient_sdp(True)
            _t.backends.cuda.enable_math_sdp(True)
            print("[ATTENTION_PATCH] SageAttention 2.2.0 (Triton fp16) + auto fallback")
        except Exception as e:
            print(f"[ATTENTION_PATCH] sageattention failed: {e} — fallback mem_efficient")
            _t.backends.cuda.enable_mem_efficient_sdp(True)
            _t.backends.cuda.enable_math_sdp(True)
    elif attn_backend == "xformers":
        _t.backends.cuda.enable_mem_efficient_sdp(True)
        _t.backends.cuda.enable_math_sdp(True)
        try:
            pipeline.unet.enable_xformers_memory_efficient_attention()
            print("[ATTENTION_PATCH] xformers ON (회색 위험)")
        except Exception as e:
            print(f"[ATTENTION_PATCH] xformers failed: {e}")
    elif attn_backend == "math":
        _t.backends.cuda.enable_mem_efficient_sdp(False)  # mem_efficient도 OFF
        _t.backends.cuda.enable_math_sdp(True)            # math만 사용
        print("[ATTENTION_PATCH] math only (정확하지만 느림+메모리)")
    else:  # mem_efficient (default)
        _t.backends.cuda.enable_mem_efficient_sdp(True)
        _t.backends.cuda.enable_math_sdp(True)
        print("[ATTENTION_PATCH] SDPA mem_efficient (xformers 대체)")

    # === MEMORY_PATCH v27: 모든 옵션 적용 ===
    try:
        pipeline.unet.enable_attention_slicing(slice_size="auto")
        print("[MEMORY_PATCH] unet attention_slicing=auto")
    except Exception as e:
        print(f"[MEMORY_PATCH] attn_slicing failed: {e}")
    # v27.6 SLICING_OPT: 환경변수 LATENTSYNC_VAE_SLICING=0 으로 OFF 가능 (default ON)
    if os.environ.get("LATENTSYNC_VAE_SLICING", "1") == "1":
        try:
            pipeline.vae.enable_slicing()
            print("[MEMORY_PATCH] vae slicing (batch dim, 정확)")
        except Exception as e:
            print(f"[MEMORY_PATCH] vae_slicing failed: {e}")
    else:
        print("[MEMORY_PATCH] vae slicing OFF (OOM 위험, 진단용)")
    # === v27.5 TILING_OPT: vae_tiling은 spatial boundary 효과 의심 (회색 마스크 원인 후보)
    # 환경변수 LATENTSYNC_VAE_TILING=1로 명시 시에만 ON (default OFF)
    if os.environ.get("LATENTSYNC_VAE_TILING", "0") == "1":
        try:
            pipeline.vae.enable_tiling()
            print("[MEMORY_PATCH] vae tiling ON (spatial boundary 위험)")
        except Exception as e:
            print(f"[MEMORY_PATCH] vae_tiling failed: {e}")
    else:
        print("[MEMORY_PATCH] vae tiling OFF (boundary 정확도 우선)")

    # === VAE_CHUNK_PATCH v27: encode/decode chunking (16GB OOM 방지) ===
    # 기본 chunk_size=2 (num_frames=16 + CFG=2 → 32 frames를 2개씩 분할 = 16번 호출)
    # 메모리 < 1.5GB 추가 안정성, 속도 5-10% 느림
    chunk_size = int(os.environ.get("LATENTSYNC_VAE_CHUNK", "2"))
    if chunk_size > 0:
        try:
            from patches.latentsync_vae_chunk_patch import apply_vae_chunk_patch
            apply_vae_chunk_patch(pipeline, chunk_size=chunk_size)
            print(f"[VAE_CHUNK_PATCH] chunk_size={chunk_size}")
        except ImportError:
            # patches dir이 sys.path에 없으면 inline patch (자기 폴더에서 import)
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from patches.latentsync_vae_chunk_patch import apply_vae_chunk_patch
                apply_vae_chunk_patch(pipeline, chunk_size=chunk_size)
                print(f"[VAE_CHUNK_PATCH] chunk_size={chunk_size} (sys.path inserted)")
            except Exception as e:
                print(f"[VAE_CHUNK_PATCH] failed: {e}")
        except Exception as e:
            print(f"[VAE_CHUNK_PATCH] failed: {e}")

    # === DeepCache: 환경변수 LATENTSYNC_CACHE_INTERVAL로 컨트롤
    # cache_interval 3 = 30% 속도 향상 (메모리 +4GB)
    # cache_interval 5 = 20% 속도 향상 (메모리 +2GB, 16GB 안전)
    # cache_interval 7 = 10% 속도 향상 (메모리 +1GB)
    cache_interval = int(os.environ.get("LATENTSYNC_CACHE_INTERVAL", "7"))  # 16GB 안전 + 부분 속도
    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=cache_interval, cache_branch_id=0)
        helper.enable()
        print(f"[DeepCache] enabled (cache_interval={cache_interval})")
    else:
        print("[DeepCache] disabled (품질 우선, --enable_deepcache로 ON 가능)")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")
    print(f"[v27] num_frames={config.data.num_frames}, resolution={config.data.resolution}")

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
