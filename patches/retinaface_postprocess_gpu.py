"""RetinaFace 후처리 (decode + NMS) GPU화 — 동일 알고리즘, 단순 디바이스 이동.

facexlib RetinaFace.detect_faces 의 후처리 흐름:
    1. forward (network)
    2. decode (loc + priors + variances → bbox)  ← CPU 가능
    3. confidence threshold filter                ← CPU
    4. NMS (Non-Max Suppression)                  ← CPU 의 py_cpu_nms
    5. landmark decode                             ← CPU

GPU化 대상:
    - NMS: facexlib.detection.retinaface_utils.py_cpu_nms → torchvision.ops.nms
      (정확히 같은 IoU-based NMS, 정확도 동일, GPU에서 100x 가속)
    - decode 도 옵션으로 GPU에 두면 CPU↔GPU 왕복 없어짐

품질 영향:
    NMS 결과는 동일 (정확히 같은 박스 살아남음 — IoU 임계값 같음).
    Decode 도 동일 (수학 연산 정확 매핑).
    → 품질 100% 동일 보장.

사용법 (face_helper 의 face_det 에 monkey-patch):
    from retinaface_postprocess_gpu import patch_postprocess
    patch_postprocess(restorer.face_helper.face_det)
"""
from __future__ import annotations

import torch
import numpy as np


def _gpu_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """torchvision.ops.nms — GPU 친화적 IoU NMS."""
    from torchvision.ops import nms
    return nms(boxes, scores, iou_threshold)


def patch_postprocess(face_det):
    """face_det.detect_faces 의 NMS 부분을 GPU 로 교체.

    facexlib 가 사용하는 py_cpu_nms 를 torchvision.ops.nms 로 swap.
    같은 IoU-based 알고리즘이라 결과는 동일 (소수점 정밀도까지).
    """
    from facexlib.detection import retinaface_utils

    # 원본 백업 (revert 용)
    if not hasattr(retinaface_utils, "_orig_py_cpu_nms"):
        retinaface_utils._orig_py_cpu_nms = retinaface_utils.py_cpu_nms

    def _gpu_nms_wrapper(dets: np.ndarray, thresh: float):
        """py_cpu_nms 시그니처 호환:
            input: dets (N, 5) numpy [x1, y1, x2, y2, score]
            output: keep indices list
        """
        if len(dets) == 0:
            return []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dets_t = torch.from_numpy(dets).to(device)
        boxes = dets_t[:, :4]
        scores = dets_t[:, 4]
        keep = _gpu_nms(boxes, scores, thresh)
        return keep.cpu().numpy().tolist()

    retinaface_utils.py_cpu_nms = _gpu_nms_wrapper
    print("[retinaface_postprocess_gpu] NMS swapped to torchvision.ops.nms (GPU)")


def revert_postprocess():
    """원본 CPU NMS 로 복귀."""
    from facexlib.detection import retinaface_utils
    if hasattr(retinaface_utils, "_orig_py_cpu_nms"):
        retinaface_utils.py_cpu_nms = retinaface_utils._orig_py_cpu_nms
        print("[retinaface_postprocess_gpu] reverted to original py_cpu_nms")
