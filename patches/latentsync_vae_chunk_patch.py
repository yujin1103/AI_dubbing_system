"""LatentSync VAE encode/decode chunking patch (v27).

문제: VAE encode/decode가 num_frames * 2 (CFG) 만큼을 한 번에 처리 →
      num_frames=16 + res=512 → 32 frames @ 512x512 → OOM (16GB)
해결: chunk_size 단위로 분할 처리. 수학적으로 동일, 메모리 -1~2GB.

사용:
  from patches.latentsync_vae_chunk_patch import apply_vae_chunk_patch
  apply_vae_chunk_patch(pipeline, chunk_size=2)
"""
import torch
from einops import rearrange


def apply_vae_chunk_patch(pipeline, chunk_size: int = 2):
    """LipsyncPipeline 인스턴스에 VAE chunking 적용.

    decode_latents, prepare_image_latents, prepare_mask_latents를 monkey-patch.
    """
    if not hasattr(pipeline, "vae"):
        raise ValueError("pipeline에 vae 속성 없음")

    vae = pipeline.vae
    cs = max(1, int(chunk_size))

    # === decode_latents chunk ===
    _orig_decode = pipeline.decode_latents

    def chunked_decode_latents(latents):
        """latents: [B, C, F, H, W] → [F, C, H, W] decoded.
        v27.5: decoded chunk을 fp32로 cast → fp16 VAE numerical instability 회피.
        SD VAE는 fp16에서 boundary precision 손실 (madebyollin sdxl-vae-fp16-fix와 동일 이슈).
        """
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # chunk decode + fp32 cast (회색 마스크 fix)
        outputs = []
        for i in range(0, latents.shape[0], cs):
            chunk = latents[i : i + cs]
            decoded = vae.decode(chunk).sample
            outputs.append(decoded.float())  # ← VAE fp16 → fp32 (mask boundary 정밀도 확보)
        return torch.cat(outputs, dim=0)

    pipeline.decode_latents = chunked_decode_latents

    # === prepare_image_latents chunk ===
    # 원본:
    #   image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
    #   image_latents = (image_latents - shift_factor) * scaling_factor
    #   image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
    #   image_latents = torch.cat([image_latents] * 2) if cfg else image_latents
    _orig_prep_img = pipeline.prepare_image_latents

    def chunked_prepare_image_latents(images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        # chunk encode
        chunks = []
        for i in range(0, images.shape[0], cs):
            chunk = images[i : i + cs]
            enc = vae.encode(chunk).latent_dist.sample(generator=generator)
            chunks.append(enc)
        image_latents = torch.cat(chunks, dim=0)
        image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
        return image_latents

    pipeline.prepare_image_latents = chunked_prepare_image_latents

    # === prepare_mask_latents chunk (masked_image vae.encode 부분만) ===
    # 원본은 mask + masked_image 둘 다 처리. masked_image의 vae.encode만 chunk
    _orig_prep_mask = pipeline.prepare_mask_latents

    def chunked_prepare_mask_latents(
        mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # mask interpolate (그대로)
        mask = torch.nn.functional.interpolate(
            mask, size=(height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)
        # chunk encode masked_image
        chunks = []
        for i in range(0, masked_image.shape[0], cs):
            chunk = masked_image[i : i + cs]
            enc = vae.encode(chunk).latent_dist.sample(generator=generator)
            chunks.append(enc)
        masked_image_latents = torch.cat(chunks, dim=0)
        masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    pipeline.prepare_mask_latents = chunked_prepare_mask_latents

    return pipeline
