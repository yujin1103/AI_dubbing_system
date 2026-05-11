"""RetinaFace TRT wrapper — 입력 해상도에 따라 적절한 엔진 자동 선택.

지원 해상도 (정적 엔진):
    HD   720x1280   (37,840 anchors)
    FHD  1080x1920  (85,200 anchors)
    QHD  1440x2560  (151,200 anchors)
    UHD  2160x3840  (340,320 anchors)

수입력이 위 4 종류와 다른 경우:
    가까운 큰 해상도 엔진을 골라서 입력을 그 크기로 letterbox / resize.
    좌표 후처리는 facexlib RetinaFace 의 detect_faces 가 자동으로 priors 를
    input shape 기준으로 다시 만들기 때문에 wrapper 가 입력 shape 를 그대로
    유지하면 안 됨 — 항상 엔진과 같은 shape 으로 보내야 한다.

사용:
    from retinaface_trt_wrapper import wrap_face_helper_detector
    wrap_face_helper_detector(restorer.face_helper, engine_dir="/workspace/trt_work/engines")
    # → input shape 보고 자동 선택
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import tensorrt as trt
from typing import Optional


# (name, H, W, anchors) — anchors = (H*W*42)//1024, 단 약간 차이가 있어 실측값 사용
RESOLUTIONS = [
    ("hd",  720,  1280,  37840),
    ("fhd", 1080, 1920,  85200),
    ("qhd", 1440, 2560, 151200),
    ("uhd", 2160, 3840, 340320),
]


def _engine_path(engine_dir: str, name: str) -> str:
    """BF16 우선 (conf precision 안전), 없으면 FP16 fallback."""
    bf16_path = os.path.join(engine_dir, f"retinaface_r50_{name}_bf16.trt")
    fp16_path = os.path.join(engine_dir, f"retinaface_r50_{name}_fp16.trt")
    if os.path.isfile(bf16_path):
        return bf16_path
    return fp16_path


def _pick_resolution(h: int, w: int):
    """입력 (h, w) 에 가장 잘 맞는 정적 엔진 선택.

    같으면 그대로, 아니면 더 큰 해상도 엔진을 쓰고 입력을 그 크기로 resize.
    너무 큰 입력 (>UHD) 은 UHD 엔진으로 강제 다운스케일.
    """
    for name, eh, ew, anchors in RESOLUTIONS:
        if h <= eh and w <= ew:
            return name, eh, ew, anchors
    # 입력 > UHD → UHD 로 다운스케일
    return RESOLUTIONS[-1]


class _TRTSession:
    """엔진 + execution context + persistent output buffers (해상도 1개분)."""

    def __init__(self, engine_path: str, h: int, w: int, anchors: int):
        self.h, self.w, self.anchors = h, w, anchors
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self._ctx = self._engine.create_execution_context()
        self._ctx.set_input_shape("images", (1, 3, h, w))
        self.out_bbox = torch.empty((1, anchors, 4), dtype=torch.float32, device="cuda")
        self.out_conf = torch.empty((1, anchors, 2), dtype=torch.float32, device="cuda")
        self.out_lmk  = torch.empty((1, anchors, 10), dtype=torch.float32, device="cuda")
        self._ctx.set_tensor_address("bbox",      self.out_bbox.data_ptr())
        self._ctx.set_tensor_address("conf",      self.out_conf.data_ptr())
        self._ctx.set_tensor_address("landmarks", self.out_lmk.data_ptr())

    def infer(self, x: torch.Tensor):
        """x: (1, 3, H, W) FP32 cuda. self.h/w 와 일치해야 함."""
        assert x.shape == (1, 3, self.h, self.w), \
            f"shape mismatch: got {tuple(x.shape)}, expected (1, 3, {self.h}, {self.w})"
        x = x.contiguous().float()
        self._ctx.set_tensor_address("images", x.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not self._ctx.execute_async_v3(stream):
            raise RuntimeError("TRT enqueueV3 failed")
        torch.cuda.synchronize()
        return self.out_bbox, self.out_conf, self.out_lmk


class RetinaFaceTRT(nn.Module):
    """다중 해상도 엔진을 lazy-load + 입력에 맞춰 dispatch."""

    def __init__(self, engine_dir: str, original: Optional[nn.Module] = None,
                 preload: Optional[list] = None):
        super().__init__()
        self.engine_dir = engine_dir
        # diffusers/facexlib 가 .device 읽을 때
        self._device_marker = nn.Parameter(
            torch.zeros(1, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )
        # facexlib RetinaFace attribute borrow (안전)
        if original is not None:
            for attr in ("variance", "cfg", "phase", "mean_tensor", "scale", "scale1"):
                if hasattr(original, attr):
                    setattr(self, attr, getattr(original, attr))
        # session cache: name -> _TRTSession
        self._sessions: dict[str, _TRTSession] = {}
        # 사전 로드 (선택)
        if preload:
            for name in preload:
                self._get_session(name)

    def _get_session(self, name: str) -> _TRTSession:
        if name in self._sessions:
            return self._sessions[name]
        for n, h, w, anchors in RESOLUTIONS:
            if n == name:
                path = _engine_path(self.engine_dir, name)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"engine not found: {path}")
                sess = _TRTSession(path, h, w, anchors)
                self._sessions[name] = sess
                print(f"[RetinaFaceTRT] loaded {name} engine ({h}x{w}) — "
                      f"{os.path.getsize(path)/1e6:.1f} MB")
                return sess
        raise ValueError(f"unknown resolution name: {name}")

    def forward(self, x: torch.Tensor):
        """facexlib RetinaFace.forward 와 동일한 인터페이스 — (loc, conf, lmk) tuple.

        입력 shape 보고 적절한 엔진 자동 선택. 해상도가 등록된 4개와 다르면
        가장 가까운 큰 해상도 엔진을 사용하고 letterbox resize.
        """
        _, _, h, w = x.shape
        # 등록된 해상도와 정확히 일치하는지
        exact = next((r for r in RESOLUTIONS if r[1] == h and r[2] == w), None)
        if exact is not None:
            name, eh, ew, anchors = exact
            sess = self._get_session(name)
            return sess.infer(x)
        # 정확히 일치 안 함 → 가까운 큰 엔진 + resize
        name, eh, ew, anchors = _pick_resolution(h, w)
        sess = self._get_session(name)
        x_resized = torch.nn.functional.interpolate(
            x, size=(eh, ew), mode="bilinear", align_corners=False
        )
        return sess.infer(x_resized)


def wrap_face_helper_detector(face_helper, engine_dir: str = "/workspace/trt_work/engines",
                              preload: Optional[list] = None):
    """face_helper.face_det.forward 를 TRT 로 swap (multi-resolution).

    facexlib FaceRestoreHelper 의 .face_det 인스턴스를 그대로 두고 forward 만 교체:
    detect_faces() 의 preprocess (resize, mean subtract) / postprocess (decode, NMS)
    는 PyTorch 원본 그대로 사용 → priors 는 input shape 기준으로 자동 생성됨.
    """
    if not hasattr(face_helper, "face_det"):
        raise AttributeError("face_helper has no .face_det")
    original = face_helper.face_det
    trt_module = RetinaFaceTRT(engine_dir, original=original, preload=preload).cuda().eval()

    import types
    def _trt_forward(self, x):
        return trt_module(x)
    original.forward = types.MethodType(_trt_forward, original)
    original._trt_module = trt_module  # 가비지 컬렉트 방지
    print(f"[RetinaFaceTRT] swapped face_det.forward (engine_dir={engine_dir})")
    return trt_module
