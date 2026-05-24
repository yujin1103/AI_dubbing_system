# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        # === ASD_FILTER_PATCH:lazy-init ===
        # Initialize filter on first call (subprocess env var read here).
        if not hasattr(self, "_asd_filter"):
            try:
                from latentsync.utils.asd_filter import maybe_load_filter
                self._asd_filter = maybe_load_filter()
            except Exception as _e:
                import traceback as _tb
                print(f"[ASD-Filter] init failed (continuing without filter): {_e}")
                _tb.print_exc()
                self._asd_filter = None
            self._asd_global_frame_offset = 0
            self._asd_skip_count = 0
        # === ASD_FILTER_PATCH:lazy-init end ===
        # === SPEAKER_PROFILE_PATCH:lazy-init ===
        if not hasattr(self, "_speaker_profiles"):
            try:
                from latentsync.utils.asd_filter import (
                    maybe_load_speaker_profiles, maybe_load_audio_gender,
                )
                self._speaker_profiles = maybe_load_speaker_profiles()
                self._audio_gender = maybe_load_audio_gender()
            except Exception as _e:
                import traceback as _tb
                print(f"[SpeakerProfile] init failed (continuing without profiles): {_e}")
                _tb.print_exc()
                self._speaker_profiles = None
                self._audio_gender = None
            self._sp_mismatch_count = 0
            self._sp_gender_mismatch_count = 0
        # === SPEAKER_PROFILE_PATCH:lazy-init end ===
        # FACE_KEEP_ORIG_PATCH + STABILITY_CHECK: face 위치/크기 outlier 제거
        faces = []
        boxes = []
        affine_matrices = []
        valid_mask = []  # True = face 있음 (paste OK), False = invalid (원본 유지)
        skipped = 0
        placeholder_face = None
        placeholder_box = None
        placeholder_affine = None
        print(f"Affine transforming {len(video_frames)} faces...")
        dark_skip = 0
        # === SPEAKER_PROFILE_PATCH:loop-optimized ===
        # Cache references for the inner loop (avoid 2656× repeat getattr).
        _asd_flt = self._asd_filter
        _profiles = self._speaker_profiles
        _audio_gender = self._audio_gender
        _offset = self._asd_global_frame_offset
        _ip = self.image_processor
        _has_asd_scene = _asd_flt is not None
        _has_profile = _profiles is not None
        _has_gender = _audio_gender is not None and _profiles is not None
        _n_dark = 0
        _n_asd_scene_skip = 0
        _n_bbox_mm = 0
        _n_sp_mm = 0
        _n_sp_gender_mm = 0
        # === SPEAKER_PROFILE_PATCH:loop-optimized end ===
        for fi, frame in enumerate(tqdm.tqdm(video_frames)):
            # frame is already an ndarray — .mean() works directly (no copy).
            if float(frame.mean()) < 25.0:
                # 거의 black frame → face가 있어도 무시 (title card 등)
                valid_mask.append(False)
                skipped += 1
                _n_dark += 1
                faces.append(None); boxes.append(None); affine_matrices.append(None)
                continue

            _g = _offset + fi
            # === ASD_FILTER_PATCH (per-scene) ===
            if _has_asd_scene and _asd_flt.should_skip(_g):
                valid_mask.append(False)
                skipped += 1
                _n_asd_scene_skip += 1
                faces.append(None); boxes.append(None); affine_matrices.append(None)
                continue
            # === ASD_FILTER_PATCH end ===

            face, box, affine_matrix = _ip.affine_transform(frame)
            # === ASD_BBOX_MATCH_PATCH ===
            if face is not None and _has_asd_scene:
                _det_bbox = _ip.last_face_bbox
                if _det_bbox is not None:
                    if _asd_flt.is_detected_face_speaker(_g, list(_det_bbox)) is False:
                        _n_bbox_mm += 1
                        face = None
            # === ASD_BBOX_MATCH_PATCH end ===
            # === SPEAKER_PROFILE_PATCH: profile + gender ===
            if face is not None and _has_profile:
                _det_emb = _ip.last_face_embedding
                if _det_emb is not None:
                    if _profiles.is_detected_face_correct_speaker(_g, _det_emb) is False:
                        _n_sp_mm += 1
                        face = None
            if face is not None and _has_gender:
                _aud_g = _audio_gender.gender_at_frame(_g)
                if _aud_g in ("male", "female"):
                    _spk = _profiles.speaker_at_frame(_g)
                    if _spk:
                        _spk_g = _profiles.gender_hint(_spk)
                        if _spk_g in ("male", "female") and _aud_g != _spk_g:
                            _n_sp_gender_mm += 1
                            face = None
            # === SPEAKER_PROFILE_PATCH end ===
            if face is None:
                valid_mask.append(False)
                skipped += 1
                faces.append(None); boxes.append(None); affine_matrices.append(None)
            else:
                valid_mask.append(True)
                if placeholder_face is None:
                    placeholder_face = face
                    placeholder_box = box
                    placeholder_affine = affine_matrix
                faces.append(face); boxes.append(box); affine_matrices.append(affine_matrix)
        # Bulk-update self counters at end (one attribute set vs N).
        dark_skip = _n_dark
        self._asd_skip_count = getattr(self, "_asd_skip_count", 0) + _n_asd_scene_skip
        self._asd_bbox_mismatch_count = getattr(self, "_asd_bbox_mismatch_count", 0) + _n_bbox_mm
        self._sp_mismatch_count = getattr(self, "_sp_mismatch_count", 0) + _n_sp_mm
        self._sp_gender_mismatch_count = getattr(self, "_sp_gender_mismatch_count", 0) + _n_sp_gender_mm
        if dark_skip > 0:
            print(f"[Brightness] {dark_skip} 어두운 frame skip (mean<25)")

        if placeholder_face is None:
            # === CHUNKED_NO_FACE_PASSTHROUGH ===
            # chunk 전체 face 없음 (drama 어두운/풍경 chunk) → 모든 frame 원본 유지.
            # restore_video 가 valid_mask=False 인 frame 은 원본 그대로 두므로
            # 빈 placeholder (검은 face) 를 set 하고 valid_mask 모두 False 로.
            print(f"[FACE_PASSTHROUGH] chunk 전체에 face 없음 → 원본 그대로 출력")
            import torch as _torch_pf
            import numpy as _np_pf
            placeholder_face = _torch_pf.zeros(3, 512, 512, dtype=_torch_pf.uint8)
            placeholder_box = [0, 0, 512, 512]
            placeholder_affine = _np_pf.eye(2, 3, dtype=_np_pf.float32)
            # 모든 frame invalid → restore_video 가 원본 frame 유지
            valid_mask = [False] * len(video_frames)
            # faces 빈 자리 placeholder 채움 (inference batch 통과용)
            faces = [placeholder_face] * len(video_frames)
            boxes = [placeholder_box] * len(video_frames)
            affine_matrices = [placeholder_affine] * len(video_frames)
            self._valid_face_mask = valid_mask
            faces_stacked = _torch_pf.stack(faces)
            return faces_stacked, boxes, affine_matrices

        # === STABILITY_CHECK: face size/position outlier 제거 ===
        # title card 등 false positive는 일반적으로 size/position이 normal range 벗어남
        import numpy as _np
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        if len(valid_indices) >= 5:  # 통계 가능한 frame 수
            sizes = []
            positions = []
            for i in valid_indices:
                x1, y1, x2, y2 = boxes[i]
                w = x2 - x1; h = y2 - y1
                sizes.append((w, h))
                positions.append(((x1 + x2) / 2, (y1 + y2) / 2))

            sizes_arr = _np.array(sizes, dtype=_np.float32)
            positions_arr = _np.array(positions, dtype=_np.float32)

            # median size로 reference 잡기
            med_w = float(_np.median(sizes_arr[:, 0]))
            med_h = float(_np.median(sizes_arr[:, 1]))
            med_cx = float(_np.median(positions_arr[:, 0]))
            med_cy = float(_np.median(positions_arr[:, 1]))

            # 허용 범위: size는 ±50%, position은 face width × 2 이내
            min_w, max_w = med_w * 0.5, med_w * 2.0
            min_h, max_h = med_h * 0.5, med_h * 2.0
            max_pos_dist = max(med_w, med_h) * 2.0

            outliers = 0
            for i in valid_indices:
                x1, y1, x2, y2 = boxes[i]
                w = x2 - x1; h = y2 - y1
                cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                # size outlier
                size_ok = (min_w <= w <= max_w) and (min_h <= h <= max_h)
                # position outlier (median 대비 거리)
                dist = ((cx - med_cx) ** 2 + (cy - med_cy) ** 2) ** 0.5
                pos_ok = dist <= max_pos_dist
                if not (size_ok and pos_ok):
                    valid_mask[i] = False
                    outliers += 1

            if outliers > 0:
                print(f"[Face Stability] {outliers}/{len(valid_indices)} frames outlier 감지 → invalid 처리")

        # None 자리에 placeholder 채움 (inference batch 통과용)
        for i in range(len(faces)):
            if faces[i] is None:
                faces[i] = placeholder_face
                boxes[i] = placeholder_box
                affine_matrices[i] = placeholder_affine

        if skipped > 0:
            print(f"[Face Skip] {skipped}/{len(video_frames)} frames face 미감지 → 원본 유지 (paste skip)")
        # === ASD_FILTER_PATCH end:summary ===
        _asd_n = getattr(self, "_asd_skip_count", 0)
        if _asd_n > 0:
            print(f"[ASD-Filter] {_asd_n}/{len(video_frames)} frames non-speaker face skipped (lipsync bypass)")
            self._asd_skip_count = 0
        _asd_bbox_mm = getattr(self, "_asd_bbox_mismatch_count", 0)
        if _asd_bbox_mm > 0:
            print(f"[ASD-BBox] {_asd_bbox_mm}/{len(video_frames)} frames detected face != speaker track → skipped")
            self._asd_bbox_mismatch_count = 0
        _sp_mm = getattr(self, "_sp_mismatch_count", 0)
        if _sp_mm > 0:
            print(f"[SpeakerProfile] {_sp_mm}/{len(video_frames)} frames detected face != diarized speaker profile → skipped")
            self._sp_mismatch_count = 0
        _sp_gmm = getattr(self, "_sp_gender_mismatch_count", 0)
        if _sp_gmm > 0:
            print(f"[SpeakerProfile-Gender] {_sp_gmm}/{len(video_frames)} frames audio gender != face gender → skipped")
            self._sp_gender_mismatch_count = 0

        faces = torch.stack(faces)
        self._valid_face_mask = valid_mask
        return faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        # FACE_KEEP_ORIG_PATCH: valid_mask가 False면 원본 frame 유지 (잘못된 paste 방지)
        video_frames = video_frames[: len(faces)]
        valid_mask = getattr(self, "_valid_face_mask", None)
        if valid_mask is None or len(valid_mask) != len(faces):
            valid_mask = [True] * len(faces)
        out_frames = []
        kept_original = 0
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            if not valid_mask[index]:
                # face=None이었던 frame → 원본 video_frame 그대로 (lipsync 적용 X)
                out_frames.append(video_frames[index])
                kept_original += 1
                continue
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        if kept_original > 0:
            print(f"[Restore] {kept_original}/{len(faces)} frames 원본 유지 (face 미감지)")
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # LOOP_VALID_MASK_FIX: loop 연장 시 valid_mask도 같이 ping-pong 해야 함
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            base_valid_mask = list(getattr(self, "_valid_face_mask", [True] * len(faces)))
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_valid_mask = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_valid_mask += base_valid_mask
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_valid_mask += base_valid_mask[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            # 연장된 valid_mask 갱신
            self._valid_face_mask = loop_valid_mask[: len(whisper_chunks)]
            print(f"[Loop] valid_mask {len(base_valid_mask)} → {len(self._valid_face_mask)} 확장")
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices



    def _chunked_call(
        self, video_path, audio_path, video_out_path,
        whisper_chunks, audio_samples, chunk_seconds,
        num_frames, video_fps, audio_sample_rate, height, width,
        num_inference_steps, guidance_scale, weight_dtype, eta,
        mask_image_path, temp_dir, generator, callback, callback_steps,
        do_classifier_free_guidance, timesteps, extra_step_kwargs,
    ):
        """CHUNKED_INFERENCE_PATCH: chunk 단위 처리로 메모리 절약."""
        import shutil, subprocess, tempfile

        device = torch.device("cuda")  # FIX: _execution_device flaky after TRT swap
        num_channels_latents = self.vae.config.latent_channels

        # ffmpeg로 video를 25fps mp4로 변환 (이미 read_video 내부에서 함)
        norm_temp = "temp"
        if os.path.exists(norm_temp):
            shutil.rmtree(norm_temp)
        os.makedirs(norm_temp, exist_ok=True)
        normalized_video = os.path.join(norm_temp, "video.mp4")
        subprocess.run(
            f"ffmpeg -loglevel error -y -nostdin -i {video_path} -r {video_fps} -crf 18 {normalized_video}",
            shell=True
        )

        # 영상 길이 + 총 frames
        import cv2
        cap = cv2.VideoCapture(normalized_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frames_per_chunk = chunk_seconds * video_fps
        num_video_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

        # whisper chunks도 같이 분할
        whisper_per_chunk = chunk_seconds * video_fps  # whisper_chunks도 fps 단위

        chunked_temp = tempfile.mkdtemp(prefix="latentsync_chunks_")
        chunk_video_paths = []

        print(f"[CHUNKED] total_frames={total_frames}, num_chunks={num_video_chunks}, frames/chunk={frames_per_chunk}", flush=True)

        for chunk_i in range(num_video_chunks):
            chunk_start_frame = chunk_i * frames_per_chunk
            chunk_end_frame = min(chunk_start_frame + frames_per_chunk, total_frames)
            chunk_actual_frames = chunk_end_frame - chunk_start_frame

            print(f"[CHUNKED] chunk {chunk_i+1}/{num_video_chunks}: frames {chunk_start_frame}~{chunk_end_frame}", flush=True)

            # 1. 이 chunk의 video frames만 read
            chunk_video_path = os.path.join(chunked_temp, f"video_chunk_{chunk_i}.mp4")
            start_sec = chunk_start_frame / video_fps
            duration_sec = chunk_actual_frames / video_fps
            subprocess.run(
                f"ffmpeg -loglevel error -y -nostdin -ss {start_sec} -i {normalized_video} -t {duration_sec} -c copy {chunk_video_path}",
                shell=True
            )
            cap = cv2.VideoCapture(chunk_video_path)
            video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            video_frames = np.array(video_frames)

            # 2. 이 chunk의 whisper_chunks subset
            chunk_whisper = whisper_chunks[chunk_start_frame:chunk_end_frame]

            # 3. loop_video + face transform
            # === ASD_FILTER_PATCH:chunk-offset ===
            self._asd_global_frame_offset = chunk_start_frame
            # === ASD_FILTER_PATCH:chunk-offset end ===
            video_frames, faces, boxes, affine_matrices = self.loop_video(chunk_whisper, video_frames)

            # 4. inference loop (기존과 동일)
            synced_video_frames = []
            all_latents = self.prepare_latents(
                len(chunk_whisper), num_channels_latents,
                height, width, weight_dtype, device, generator,
            )
            num_inferences = math.ceil(len(chunk_whisper) / num_frames)
            for i in tqdm.tqdm(range(num_inferences), desc=f"Chunk {chunk_i+1} inference"):
                if self.unet.add_audio_layer:
                    audio_embeds = torch.stack(chunk_whisper[i * num_frames : (i + 1) * num_frames])
                    audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                    if do_classifier_free_guidance:
                        null_audio_embeds = torch.zeros_like(audio_embeds)
                        audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
                else:
                    audio_embeds = None
                inference_faces = faces[i * num_frames : (i + 1) * num_frames]
                latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
                ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    inference_faces, affine_transform=False
                )
                mask_latents, masked_image_latents = self.prepare_mask_latents(
                    masks, masked_pixel_values, height, width, weight_dtype, device, generator, do_classifier_free_guidance,
                )
                ref_latents = self.prepare_image_latents(
                    ref_pixel_values, device, weight_dtype, generator, do_classifier_free_guidance,
                )
                # === DPM_CHUNKED_RESET_PATCH: chunked 경로에서도 chunk 마다 scheduler reset ===
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                if hasattr(self.unet, "reset_cache"):
                    self.unet.reset_cache()
                # === DPM_CHUNKED_RESET_PATCH end ===
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        unet_input = self.scheduler.scale_model_input(unet_input, t)
                        unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                        if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                decoded_latents = self.decode_latents(latents)
                # MEMORY_LEAK_PATCH: decode_latents 후 GPU cleanup
                import torch as _t, gc as _gc
                _t.cuda.empty_cache(); _gc.collect()
                decoded_latents = self.paste_surrounding_pixels_back(
                    decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                )
                synced_video_frames.append(decoded_latents)

            # 5. restore + chunk 결과 저장
            synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

            # chunk audio 추출
            chunk_audio_path = os.path.join(chunked_temp, f"audio_chunk_{chunk_i}.wav")
            subprocess.run(
                f"ffmpeg -loglevel error -y -nostdin -ss {start_sec} -i {audio_path} -t {duration_sec} -c copy {chunk_audio_path}",
                shell=True
            )

            # chunk video 저장 (write_video 함수)
            chunk_out_path = os.path.join(chunked_temp, f"out_chunk_{chunk_i}.mp4")
            from ..utils.util import write_video
            write_video(chunk_out_path, synced_video_frames, fps=video_fps)

            # audio merge
            chunk_final_path = os.path.join(chunked_temp, f"final_chunk_{chunk_i}.mp4")
            subprocess.run(
                f"ffmpeg -loglevel error -y -nostdin -i {chunk_out_path} -i {chunk_audio_path} -c:v copy -c:a aac -shortest {chunk_final_path}",
                shell=True
            )
            chunk_video_paths.append(chunk_final_path)

            # 메모리 해제
            del video_frames, faces, boxes, affine_matrices, synced_video_frames, all_latents
            torch.cuda.empty_cache()

        # 6. 모든 chunks concat
        concat_list = os.path.join(chunked_temp, "concat.txt")
        with open(concat_list, "w") as f:
            for cp in chunk_video_paths:
                f.write(f"file '{cp}'\n")
        subprocess.run(
            f"ffmpeg -loglevel error -y -nostdin -f concat -safe 0 -i {concat_list} -c copy {video_out_path}",
            shell=True
        )

        shutil.rmtree(chunked_temp, ignore_errors=True)
        print(f"[CHUNKED] 완료: {video_out_path}", flush=True)
        if is_train := False:
            self.unet.train()
        return

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = torch.device("cuda")  # FIX: _execution_device flaky after TRT swap
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)

        # CHUNKED_INFERENCE_PATCH: 환경변수로 chunk 단위 처리 활성화 (메모리 절약)
        chunk_seconds = int(os.environ.get("LATENTSYNC_CHUNK_SECONDS", "0"))
        if chunk_seconds > 0:
            print(f"[CHUNKED_INFERENCE_PATCH] chunk_seconds={chunk_seconds} (메모리 절약 모드)", flush=True)
            return self._chunked_call(
                video_path, audio_path, video_out_path,
                whisper_chunks, audio_samples, chunk_seconds,
                num_frames, video_fps, audio_sample_rate, height, width,
                num_inference_steps, guidance_scale, weight_dtype, eta,
                mask_image_path, temp_dir, generator, callback, callback_steps,
                do_classifier_free_guidance, timesteps, extra_step_kwargs,
            )

        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # === DPM_RESET_PATCH: chunk 마다 scheduler state 재설정 ===
            # DPMSolver 는 _step_index 가 chunk 간 누적 → IndexError 방지
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            # === DPM_RESET_PATCH end ===
            # === TEACACHE_RESET_PATCH ===
            # chunk 마다 cache 무효화 (입력 frame 이 완전히 다름)
            if hasattr(self.unet, "reset_cache"):
                self.unet.reset_cache()
            # === TEACACHE_RESET_PATCH end ===
            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
