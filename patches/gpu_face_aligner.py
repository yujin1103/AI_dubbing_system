"""GPU 기반 face alignment + paste-back (cv2.warpAffine 대체).

facexlib FaceRestoreHelper 의 align_warp_face / paste_faces_to_input_image
는 cv2.warpAffine 으로 CPU에서 도는데, 1080p 1프레임당 forward warp ~50ms +
inverse warp+mask warp ~50ms = ~100ms 이상이 깎인다. 1602프레임이면 ~160s.
이 시간을 grid_sample (CUDA bilinear) 로 옮긴다.

핵심 매핑:
    cv2.estimateAffinePartial2D(src_pts, dst_pts) -> M (2x3)
    cv2.warpAffine 은 default 로 M 을 input->output 매핑으로 받고 내부에서
    invert 해서 output->input 으로 sampling 한다.
    F.affine_grid(theta, ...) 는 normalized output->input 매핑을 받는다.
        theta 는 (N, 2, 3), align_corners=True 를 쓴다.

    cv2 픽셀좌표 (정수=픽셀중심) 를 normalized [-1, 1] 로 변환:
        x_norm = 2*x_pix/(W-1) - 1   (align_corners=True)

    derivation: input pixel = M_inv @ [output pixel; 1] 을 normalized 양변으로
    재정렬하면, theta_00 = a*(W_o-1)/(W_i-1), 등 (자세한 식은 _cv_to_theta 안에).

API:
    aligner = GPUFaceAligner(device, face_template, face_size=512)
    faces_512 = aligner.align(img_chw_bgr_01_gpu, landmarks_list_np)  # (N, 3, 512, 512) [-1, 1] RGB
    restored_img = aligner.paste(img_chw_bgr_01_gpu, restored_512, landmarks_list_np,
                                 parse_masks_or_None)  # (3, H, W) [0, 1] BGR

호환:
    landmarks_list_np: list of (5, 2) numpy array 좌표 (cv2 픽셀)
    face_template: (5, 2) numpy array — FFHQ 표준 5-point 좌표 (512 기준)
    img_chw_bgr_01_gpu: (3, H, W) FP32 [0, 1] BGR (cv2.imread 와 같은 채널순)
    restored_512: (N, 3, 512, 512) FP32 [-1, 1] RGB (GFPGAN TRT 출력)
"""
from __future__ import annotations

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Optional, Tuple


def _cv_to_theta(M_2x3: np.ndarray, src_size: Tuple[int, int],
                 dst_size: Tuple[int, int]) -> torch.Tensor:
    """cv2 affine matrix (output->input pixel) -> normalized theta (output->input norm).

    M_2x3:    cv2.invertAffineTransform 로 만든 output->input 매핑 (output 픽셀에서
              input 픽셀을 찾는). row-major: [[a,b,c],[d,e,f]] s.t.
                  x_in = a*x_out + b*y_out + c
                  y_in = d*x_out + e*y_out + f
    src_size: (H_src, W_src)  --  input image 크기 (sampling 대상)
    dst_size: (H_dst, W_dst)  --  output image 크기 (theta 가 만들 grid)

    align_corners=True 기준:
        x_pix = (x_norm + 1) * (W - 1) / 2
    위 식을 양변에 대입하면
        theta_00 = a * (W_dst - 1) / (W_src - 1)
        theta_01 = b * (H_dst - 1) / (W_src - 1)
        theta_02 = (a*(W_dst-1) + b*(H_dst-1) + 2c) / (W_src - 1) - 1
        theta_10 = d * (W_dst - 1) / (H_src - 1)
        theta_11 = e * (H_dst - 1) / (H_src - 1)
        theta_12 = (d*(W_dst-1) + e*(H_dst-1) + 2f) / (H_src - 1) - 1
    """
    H_src, W_src = src_size
    H_dst, W_dst = dst_size
    a, b, c = M_2x3[0]
    d, e, f = M_2x3[1]
    theta = np.zeros((2, 3), dtype=np.float32)
    theta[0, 0] = a * (W_dst - 1) / (W_src - 1)
    theta[0, 1] = b * (H_dst - 1) / (W_src - 1)
    theta[0, 2] = (a * (W_dst - 1) + b * (H_dst - 1) + 2 * c) / (W_src - 1) - 1
    theta[1, 0] = d * (W_dst - 1) / (H_src - 1)
    theta[1, 1] = e * (H_dst - 1) / (H_src - 1)
    theta[1, 2] = (d * (W_dst - 1) + e * (H_dst - 1) + 2 * f) / (H_src - 1) - 1
    return torch.from_numpy(theta)


def _gaussian_kernel_1d(ksize: int, sigma: float, dtype=torch.float32, device=None) -> torch.Tensor:
    """cv2.GaussianBlur 와 호환되는 1D Gaussian kernel (정규화됨).

    cv2.getGaussianKernel(ksize, sigma) 와 동일한 가중치를 만든다.
    sigma <= 0 이면 cv2 는 sigma = 0.3 * ((ksize-1)*0.5 - 1) + 0.8 로 자동.
    여기서는 사용자가 명시적으로 sigma 를 주는 케이스만 (mask blur 11) 다룬다.
    """
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    half = (ksize - 1) // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k


# OpenCV BGR <-> RGB 채널 swap.  이 모듈은 입력을 BGR FP32 [0, 1] 으로 받고
# face_parse / GFPGAN 은 RGB [-1, 1] 로 동작.

class GPUFaceAligner:
    """face alignment + paste-back 을 grid_sample 로.

    동시에 face_parse 로 segmentation mask 를 만들 수 있다 (use_parse=True).
    use_parse=False 면 square mask + erode + blur (cv2 와 동일) 을 GPU 로 한다.

    face_parse 모듈을 외부에서 주입한다 (facexlib parsing 모델). 무거운 객체라
    aligner 가 직접 만들지 않는다.
    """

    # GFPGANer 가 쓰는 mask colormap (19-class -> 0/255)
    # 0=background, 14=neck (둘 다 mask 에서 0)  - 그 외에 16/17/18 은 옷/모자 등.
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                     255, 255, 255, 0, 255, 0, 0, 0]

    def __init__(self, device: torch.device, face_template: np.ndarray,
                 face_size: int = 512, use_parse: bool = True,
                 face_parse_module: Optional[torch.nn.Module] = None):
        self.device = device
        self.face_size = face_size
        self.face_template = face_template.astype(np.float32)  # (5, 2)
        self.use_parse = use_parse
        self.face_parse = face_parse_module  # facexlib parsenet (eval/cuda)

        # Pre-built constants on GPU
        self._mask_lut = torch.tensor(
            [v / 255.0 for v in self.MASK_COLORMAP],
            dtype=torch.float32, device=device,
        )  # (19,)

        # Gaussian blur kernel (101, sigma=11) — separable depthwise conv
        kernel = _gaussian_kernel_1d(101, 11.0, dtype=torch.float32, device=device)
        # depthwise conv requires (out_ch, 1, 1, K) and (out_ch, 1, K, 1)
        self._gauss_kx = kernel.view(1, 1, 1, 101)
        self._gauss_ky = kernel.view(1, 1, 101, 1)
        self._gauss_pad_x = (0, 0, 50, 50)  # for 1xK kernel: pad H? no — pad W? F.pad uses (W_l, W_r, H_t, H_b)
        # We use F.pad with (left, right, top, bottom) => pad along W for horizontal kernel
        # We will pad explicitly inside _gaussian_blur.

        # RGB <-> BGR swap as fixed perm tensor
        self._bgr2rgb = torch.tensor([2, 1, 0], device=device, dtype=torch.long)

        # face_template tensor (kept on CPU — used by cv2.estimateAffinePartial2D)

        # cumulative diagnostics
        self.t_warp_forward_ms = 0.0
        self.t_warp_paste_ms = 0.0
        self.t_parse_ms = 0.0
        self.t_blend_ms = 0.0
        self.n_calls = 0

    # ------------------------------------------------------------------
    # Forward warp (extract face crops)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def align(self, img_chw_bgr_01: torch.Tensor,
              landmarks_list: List[np.ndarray]) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Extract aligned face crops on GPU.

        img_chw_bgr_01: (3, H, W) FP32 BGR [0, 1] on GPU
        landmarks_list: list of (5, 2) numpy arrays in image-pixel coords

        Returns
        -------
        faces_rgb_norm : (N, 3, 512, 512) FP32 RGB [-1, 1] on GPU
                        ready for GFPGAN TRT input
        affine_matrices : list of (2, 3) numpy arrays (cv2 input->output)
                          이 매트릭스를 paste 할 때 그대로 다시 쓴다.
        """
        if not landmarks_list:
            empty = torch.empty((0, 3, self.face_size, self.face_size),
                                dtype=torch.float32, device=self.device)
            return empty, []

        N = len(landmarks_list)
        H, W = img_chw_bgr_01.shape[-2], img_chw_bgr_01.shape[-1]

        # build per-face affine matrices on CPU (cv2 LMEDS, 5-point)
        affines = []
        thetas = torch.zeros((N, 2, 3), dtype=torch.float32)
        for i, lm in enumerate(landmarks_list):
            M = cv2.estimateAffinePartial2D(lm, self.face_template, method=cv2.LMEDS)[0]
            if M is None:
                # Fallback: identity (very rare). Skip.
                M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            affines.append(M.astype(np.float32))
            # for forward warp, sampling is from image -> face_512
            # cv2 internally inverts M to get output->input map
            M_inv = cv2.invertAffineTransform(M)
            theta = _cv_to_theta(M_inv, src_size=(H, W),
                                 dst_size=(self.face_size, self.face_size))
            thetas[i] = theta

        thetas = thetas.to(self.device, non_blocking=True)
        # affine_grid wants (N, 2, 3) and returns (N, H_out, W_out, 2)
        grid = F.affine_grid(thetas,
                             size=(N, 3, self.face_size, self.face_size),
                             align_corners=True)
        # broadcast image to N
        img_n = img_chw_bgr_01.unsqueeze(0).expand(N, -1, -1, -1)
        # padding_mode='zeros' to mirror cv2 BORDER_CONSTANT (gray=132 in cv2,
        # but for in-frame faces this rarely hits the border — acceptable diff).
        faces_bgr = F.grid_sample(img_n, grid, mode='bilinear',
                                  padding_mode='zeros', align_corners=True)
        # BGR -> RGB and normalize to [-1, 1]
        faces_rgb = faces_bgr.index_select(1, self._bgr2rgb)
        faces_rgb_norm = faces_rgb * 2.0 - 1.0

        return faces_rgb_norm, affines

    # ------------------------------------------------------------------
    # Parse mask on GPU
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _parse_mask(self, faces_rgb_norm: torch.Tensor) -> torch.Tensor:
        """faces_rgb_norm: (N, 3, 512, 512) RGB [-1, 1].
        Returns (N, 1, 512, 512) FP32 [0, 1] soft mask aligned with face_512.

        Steps reproduce facexlib FaceRestoreHelper.paste_faces_to_input_image
        when self.use_parse=True:
            - face_parse(face_input) -> 19-class logits
            - argmax -> (512, 512) class map
            - lookup table -> 0 or 1 mask
            - GaussianBlur 101x101 sigma=11, twice
            - zero-out 10-pixel borders
        """
        # face_parse expects RGB in [-1, 1] (img/255 then mean=std=0.5 normalize)
        # which is exactly what faces_rgb_norm already is.
        logits = self.face_parse(faces_rgb_norm)[0]  # (N, 19, 512, 512)
        labels = logits.argmax(dim=1)  # (N, 512, 512) int64
        # gather mask values via lookup table
        mask = self._mask_lut[labels].unsqueeze(1)  # (N, 1, 512, 512) FP32

        # 2x Gaussian blur (101x101, sigma=11) — depthwise separable
        mask = self._gaussian_blur(mask)
        mask = self._gaussian_blur(mask)

        # zero out 10-pixel borders
        thres = 10
        mask[..., :thres, :] = 0
        mask[..., -thres:, :] = 0
        mask[..., :, :thres] = 0
        mask[..., :, -thres:] = 0
        return mask

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Separable 1D Gaussian, kernel=101, sigma=11, padding=50 (replicate)."""
        N = x.shape[0]
        # horizontal: (1, 1, 1, 101) applied per channel — depthwise
        kx = self._gauss_kx.expand(N, 1, 1, 101) if False else self._gauss_kx
        ky = self._gauss_ky if False else self._gauss_ky
        # x is (N, 1, H, W) for masks, depthwise with groups=N? Better: just use
        # per-batch single-channel conv (groups=1 since C=1) and rely on conv2d
        # doing the broadcast over N.
        # F.conv2d with kernel (1, 1, 1, 101): treats the single-channel input
        # uniformly — works on any (N, 1, H, W).
        x = F.pad(x, (50, 50, 0, 0), mode='replicate')
        x = F.conv2d(x, kx, padding=0)
        x = F.pad(x, (0, 0, 50, 50), mode='replicate')
        x = F.conv2d(x, ky, padding=0)
        return x

    # ------------------------------------------------------------------
    # Paste back
    # ------------------------------------------------------------------
    @torch.no_grad()
    def paste(self, img_chw_bgr_01: torch.Tensor,
              restored_faces_rgb_norm: torch.Tensor,
              affine_matrices: List[np.ndarray],
              parse_input_faces_rgb_norm: Optional[torch.Tensor] = None,
              upscale_factor: float = 1.0) -> torch.Tensor:
        """Paste restored faces back to the original image.

        img_chw_bgr_01: (3, H, W) FP32 BGR [0, 1] on GPU
        restored_faces_rgb_norm: (N, 3, 512, 512) FP32 RGB [-1, 1] on GPU
                                  (GFPGAN TRT 출력)
        affine_matrices: per-face cv2 affine M (input -> face_512).
                         GPUFaceAligner.align 반환값을 그대로 넘기면 된다.
        parse_input_faces_rgb_norm: face_parse 입력으로 쓸 face crop. None 이면
            restored 자체를 입력으로 사용 (facexlib 가 그렇게 한다).
        upscale_factor: GFPGANer.upscale (보통 1).

        Returns: (3, H_out, W_out) FP32 BGR [0, 1] on GPU,
                 H_out = H * upscale_factor, W_out = W * upscale_factor.
        """
        N = restored_faces_rgb_norm.shape[0]
        if N == 0:
            return img_chw_bgr_01

        H, W = img_chw_bgr_01.shape[-2], img_chw_bgr_01.shape[-1]
        H_up = int(round(H * upscale_factor))
        W_up = int(round(W * upscale_factor))

        # restored RGB [-1, 1] -> BGR [0, 1]
        restored_bgr = restored_faces_rgb_norm.index_select(1, self._bgr2rgb)
        restored_bgr_01 = (restored_bgr + 1.0) * 0.5
        restored_bgr_01 = restored_bgr_01.clamp_(0, 1)

        # mask: (N, 1, 512, 512) FP32 [0, 1]
        if self.use_parse:
            parse_in = parse_input_faces_rgb_norm if parse_input_faces_rgb_norm is not None \
                else restored_faces_rgb_norm
            mask_512 = self._parse_mask(parse_in)
        else:
            mask_512 = self._square_mask(N, upscale_factor)

        # background canvas: upscale input image with bilinear (cv2 LANCZOS4 차이는 작음)
        if upscale_factor != 1.0:
            bg = F.interpolate(img_chw_bgr_01.unsqueeze(0),
                               size=(H_up, W_up), mode='bilinear',
                               align_corners=False, antialias=True).squeeze(0)
        else:
            bg = img_chw_bgr_01

        # build per-face inverse-warp theta: face_512 -> H_up x W_up
        thetas = torch.zeros((N, 2, 3), dtype=torch.float32)
        for i, M in enumerate(affine_matrices):
            # cv2 inverse_affine = invert(M); then add extra_offset on translation
            # for upscale > 1.  affine_matrix is input->face_512.
            inv = cv2.invertAffineTransform(M).astype(np.float32)
            inv = inv * 1.0  # copy
            # cv2 rule: inverse_affine *= upscale_factor (col 0..2)
            inv = inv * upscale_factor
            if upscale_factor > 1:
                inv[:, 2] += 0.5 * upscale_factor
            # for paste-back, sampling is face_512 -> H_up x W_up.
            # cv2.warpAffine(face, inv, (W_up, H_up)) treats inv as input->output
            # and inverts internally to get output->input = M (modified).
            # output here is (H_up, W_up); input is face_512.
            # So output->input pixel map = invertAffineTransform(inv) (post-scale).
            M_out_to_in = cv2.invertAffineTransform(inv)
            theta = _cv_to_theta(M_out_to_in,
                                 src_size=(self.face_size, self.face_size),
                                 dst_size=(H_up, W_up))
            thetas[i] = theta

        thetas = thetas.to(self.device, non_blocking=True)
        # combine face + mask into a single 4-channel sample to halve grid_sample cost
        face_plus_mask = torch.cat([restored_bgr_01, mask_512], dim=1)  # (N, 4, 512, 512)
        grid = F.affine_grid(thetas, size=(N, 4, H_up, W_up),
                             align_corners=True)
        sampled = F.grid_sample(face_plus_mask, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        # split back
        inv_restored = sampled[:, :3]    # (N, 3, H_up, W_up)
        inv_soft_mask = sampled[:, 3:4]  # (N, 1, H_up, W_up)
        inv_soft_mask = inv_soft_mask.clamp_(0, 1)

        # composite: out = mask * inv_restored + (1 - mask) * bg, applied
        # cumulatively per face.
        out = bg
        if N == 1:
            out = inv_soft_mask[0] * inv_restored[0] + (1 - inv_soft_mask[0]) * out
        else:
            for i in range(N):
                out = inv_soft_mask[i] * inv_restored[i] + (1 - inv_soft_mask[i]) * out

        return out.clamp_(0, 1)

    # ------------------------------------------------------------------
    # Square mask path (use_parse=False)
    # ------------------------------------------------------------------
    def _square_mask(self, N: int, upscale_factor: float) -> torch.Tensor:
        """Pre-erosion of a 512x512 ones mask (cv2 use_parse=False branch).

        cv2 코드는 inv_mask = cv2.warpAffine(ones, inverse_affine, ...) 후
        erode + Gaussian blur 한다 (W_up 크기에서). GPU 에서 그대로 따라
        하기엔 비싸서, 단순 face_size 안쪽에 약간 erode 한 ones mask 를
        쓰고 paste 할 때 grid_sample 로 함께 warp 한다.
        """
        # erode amount in pixels (cv2 uses int(2*upscale)) — after warp.
        # 가장 단순한 근사: 가장자리 4 픽셀 zero.
        m = torch.ones((1, 1, self.face_size, self.face_size),
                       dtype=torch.float32, device=self.device)
        e = max(int(round(2 * upscale_factor)), 1)
        m[..., :e, :] = 0
        m[..., -e:, :] = 0
        m[..., :, :e] = 0
        m[..., :, -e:] = 0
        # one round of Gaussian blur for soft edges
        m = self._gaussian_blur(m)
        return m.expand(N, 1, self.face_size, self.face_size).contiguous()
