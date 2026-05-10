"""LatentSync paste_back 양 볼 alpha=0 patch (3차 fallback).

mask_smaller.png로도 회색 패치 발생 시 사용.
inv_soft_mask 후처리에서 양 볼 위치(face crop 좌표 row 64~140) alpha=0 강제.
즉 paste back 시 양 볼 영역은 무조건 원본 face 그대로.

사용:
  from patches.latentsync_paste_cheek_zero import patch_restorer
  patch_restorer(pipeline.image_processor.restorer, cheek_top=64, cheek_bottom=140)
"""
import torch
import kornia
import cv2
import numpy as np
from einops import rearrange
import torchvision
from torchvision import transforms


def patch_restorer(restorer, cheek_top: int = 64, cheek_bottom: int = 140):
    """ImageRestorer.restore_img를 monkey-patch — inv_soft_mask 양 볼 영역 0 강제."""
    _orig_restore_img = restorer.restore_img
    resolution = getattr(restorer, "resolution", 256)

    def patched_restore_img(input_img, face, affine_matrix):
        h, w, _ = input_img.shape

        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix).to(
                device=restorer.device, dtype=restorer.dtype
            ).unsqueeze(0)

        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)
        face = face.to(dtype=restorer.dtype).unsqueeze(0)

        inv_face = kornia.geometry.transform.warp_affine(
            face, inv_affine_matrix, (h, w),
            mode="bilinear", padding_mode="fill", fill_value=restorer.fill_value
        ).squeeze(0)
        inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255

        input_img_tensor = rearrange(
            torch.from_numpy(input_img).to(device=restorer.device, dtype=restorer.dtype),
            "h w c -> c h w"
        )

        # === MASK 양 볼 영역 0 강제 ===
        # restorer.mask: (1, 1, H_mask, W_mask) face crop 좌표계 mask
        cheek_zeroed_mask = restorer.mask.clone()
        H = cheek_zeroed_mask.shape[-2]
        # cheek_top, cheek_bottom은 256 기준 → H에 맞게 scale
        ct = int(cheek_top * H / 256)
        cb = int(cheek_bottom * H / 256)
        cheek_zeroed_mask[:, :, ct:cb, :] = 0  # 양 볼 영역 = paste skip
        # 단, 가운데 코/입 위 영역 (x = W/3 ~ 2W/3)는 그대로 둠
        Wm = cheek_zeroed_mask.shape[-1]
        cheek_zeroed_mask[:, :, ct:cb, Wm // 3 : 2 * Wm // 3] = restorer.mask[:, :, ct:cb, Wm // 3 : 2 * Wm // 3]

        inv_mask = kornia.geometry.transform.warp_affine(
            cheek_zeroed_mask, inv_affine_matrix, (h, w), padding_mode="zeros"
        )

        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask,
            torch.ones(
                (int(2 * restorer.upscale_factor), int(2 * restorer.upscale_factor)),
                device=restorer.device, dtype=restorer.dtype
            ),
        )

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
        pasted_face = inv_mask_erosion_t * inv_face
        total_face_area = torch.sum(inv_mask_erosion.float())
        if total_face_area.item() < 1.0:
            # mask가 거의 비어있으면 원본 그대로 (paste skip)
            return input_img.copy()
        w_edge = int(total_face_area.item() ** 0.5) // 20
        erosion_radius = max(1, w_edge * 2)

        inv_mask_erosion = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        inv_mask_center = torch.from_numpy(inv_mask_center).to(
            device=restorer.device, dtype=restorer.dtype
        )[None, None, ...]

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0)
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
        img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img_tensor

        img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        img_back = img_back.cpu().numpy()
        return img_back

    restorer.restore_img = patched_restore_img
    print(f"[CHEEK_ZERO_PATCH] applied (cheek_top={cheek_top}, cheek_bottom={cheek_bottom})")
