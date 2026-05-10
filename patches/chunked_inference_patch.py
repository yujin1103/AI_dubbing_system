"""LatentSync pipeline에 Chunked Inference 모드 추가.

기존 동작:
  pipeline.__call__()
    → video_frames 전체 메모리 (18 GB for 2분 1080p)
    → OOM (50 GB 컨테이너 한도)

패치 후 동작:
  환경변수 LATENTSYNC_CHUNK_SECONDS=30 설정 시
  → 30초씩 chunk 처리
  → 각 chunk: video_frames(4 GB) + face crops + 결과 → disk write → 메모리 해제
  → 마지막에 ffmpeg concat
  → 메모리 사용: 4-6 GB 안정 (1080p 유지)
"""
from pathlib import Path

p = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")
src = p.read_text(encoding="utf-8")

if "CHUNKED_INFERENCE_PATCH" in src:
    print("이미 적용됨")
    raise SystemExit(0)

# pipeline __call__의 video_frames 로드 부분 찾기
old = """        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)"""

new = """        whisper_feature = self.audio_encoder.audio2feat(audio_path)
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

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)"""

if old in src:
    src = src.replace(old, new)
else:
    print("[CHUNKED_PATCH] 원본 패턴 못 찾음")
    raise SystemExit(1)

# import os 확인 (이미 있을 수도)
if "import os\n" not in src:
    src = src.replace("import math", "import math\nimport os")

# _chunked_call 메서드를 클래스에 추가 (restore_video 메서드 다음)
chunked_method = '''

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

        device = self._execution_device
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
                f.write(f"file '{cp}'\\n")
        subprocess.run(
            f"ffmpeg -loglevel error -y -nostdin -f concat -safe 0 -i {concat_list} -c copy {video_out_path}",
            shell=True
        )

        shutil.rmtree(chunked_temp, ignore_errors=True)
        print(f"[CHUNKED] 완료: {video_out_path}", flush=True)
        if is_train := False:
            self.unet.train()
        return
'''

# class 끝 직전에 추가 (마지막 메서드 다음)
# restore_video 메서드 끝 위치 찾기 (def __call__ 직전)
class_marker = "    @torch.no_grad()\n    def __call__("
if class_marker in src:
    src = src.replace(class_marker, chunked_method + "\n" + class_marker)
else:
    print("[CHUNKED_PATCH] _chunked_call 삽입 위치 못 찾음")
    raise SystemExit(1)

p.write_text(src, encoding="utf-8")
print("[Chunked Inference] 패치 적용 완료 ✅")
print("사용: LATENTSYNC_CHUNK_SECONDS=30 환경변수로 활성화")
