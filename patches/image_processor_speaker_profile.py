# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        # === ASD_BBOX_MATCH_PATCH ===
        # Stash original-frame bbox so the lipsync pipeline can ask ASD
        # whether the detected face matches the speaker track.
        self.last_face_bbox = bbox
        # === ASD_BBOX_MATCH_PATCH end ===
        # === SPEAKER_PROFILE_PATCH: face embedding ===
        # Stash face embedding (if recognition module enabled) so the lipsync
        # pipeline can verify the detected face matches the current speaker.
        self.last_face_embedding = getattr(self.face_detector, "last_embedding", None)
        # === SPEAKER_PROFILE_PATCH end ===
        if bbox is None:
            return None, None, None  # FACE_SKIP_PATCH

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        # === FACE_PROFILE_SKIP_PATCH ===
        # 측면 face (yaw > threshold) 는 LatentSync 가 입 위치를 잘못 잡아서
        # "입이 떠다니는" artifact 발생 → 검출만 하고 원본 frame 유지.
        import os as _os_pf
        _profile_thresh = float(_os_pf.environ.get("LATENTSYNC_PROFILE_THRESHOLD", "0"))
        if _profile_thresh > 0:
            eye_dist = float(np.linalg.norm(pt_right_eye - pt_left_eye))
            if eye_dist > 1.0:  # eye_dist=0 이면 div by zero 회피
                eye_midpoint_x = float((pt_left_eye[0] + pt_right_eye[0]) / 2)
                nose_offset_x = abs(float(pt_nose[0]) - eye_midpoint_x)
                yaw_ratio = nose_offset_x / eye_dist
                if yaw_ratio > _profile_thresh:
                    return None, None, None  # profile face → skip (원본 유지)
        # === FACE_PROFILE_SKIP_PATCH end ===

        # === FACE_DISTANCE_SKIP_PATCH ===
        # 작은 (멀리 있는) face 는 over-zoom 으로 입 위치 부정확 → 원본 유지
        import os as _os_dist
        _dist_min = float(_os_dist.environ.get("LATENTSYNC_FACE_DIAG_MIN_RATIO", "0"))
        if _dist_min > 0 and bbox is not None:
            try:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                bbox_diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                # image is torch tensor (H, W, C) — get frame diagonal
                if hasattr(image, "shape") and len(image.shape) >= 2:
                    H, W = int(image.shape[0]), int(image.shape[1])
                    frame_diag = (H ** 2 + W ** 2) ** 0.5
                    if frame_diag > 0:
                        dist_ratio = bbox_diag / frame_diag
                        if dist_ratio < _dist_min:
                            return None, None, None  # 너무 작은 face → skip
            except Exception:
                pass  # 측정 실패 시 패스
        # === FACE_DISTANCE_SKIP_PATCH end ===

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
