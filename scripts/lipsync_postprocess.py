"""LatentSync 출력 영상 후처리 — 마스크 경계 + 디테일 개선.

적용 효과:
  1. GFPGAN face restoration  : 얼굴 영역만 detail 복원
                                  → 마스크 경계 자동 부드럽게 + 색상 일관성
  2. Color matching (선택)     : face crop 영역의 색감을 원본 분포에 맞춤
                                  → 옆모습/회전 시 face 색 차이 감소
  3. Optional: re-encode HQ    : 원본 오디오 보존, libx264 crf 18

사용법:
  /opt/venv_lipsync/bin/python /workspace/scripts/lipsync_postprocess.py \
      --input /workspace/media/output/test_lipsync.mp4 \
      --output /workspace/media/output/test_lipsync_post.mp4 \
      --gfpgan-model /workspace/media/model_cache/musetalk/models/gfpgan/GFPGANv1.4.pth \
      --color-match            # 선택: 원본과 색감 일치 (--reference 필요)
      --reference /workspace/media/output/test.mp4   # 원본 영상 (color match 시)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def load_gfpgan(model_path: str, use_fp16: bool = True):
    """GFPGAN restorer 로드 (fp16 + face_det/face_parse 모두)."""
    from gfpgan import GFPGANer

    print(f"[GFPGAN] 로드: {model_path} (fp16={use_fp16})")
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,                # 해상도 유지
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
    )
    if use_fp16:
        try:
            if hasattr(restorer, "gfpgan") and restorer.gfpgan is not None:
                restorer.gfpgan = restorer.gfpgan.half()
            if hasattr(restorer, "face_helper"):
                fh = restorer.face_helper
                if getattr(fh, "face_det", None) is not None:
                    fh.face_det = fh.face_det.half()
                if getattr(fh, "face_parse", None) is not None:
                    fh.face_parse = fh.face_parse.half()
            print("[GFPGAN] fp16 변환 완료")
        except Exception as e:
            print(f"[GFPGAN] fp16 변환 실패 ({e}) — fp32 사용")
            use_fp16 = False
    return restorer, use_fp16


def color_match_at_boundary(generated_frame, reference_frame, blend: float = 0.5):
    """원본 reference_frame과 generated_frame의 색상 분포 매칭.

    얼굴 영역 boundary에서 색상 차이가 크면 마스크가 보임.
    Histogram matching 대신 단순 mean/std shift (속도 + 자연스러움).
    """
    import cv2
    import numpy as np

    if generated_frame.shape != reference_frame.shape:
        return generated_frame

    # LAB 색공간에서 통계 매칭 (RGB보다 자연스러움)
    gen_lab = cv2.cvtColor(generated_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    # L 채널만 약하게 보정 (color는 baby step)
    matched = gen_lab.copy()
    for c in range(3):
        mu_g, std_g = gen_lab[..., c].mean(), gen_lab[..., c].std() + 1e-6
        mu_r, std_r = ref_lab[..., c].mean(), ref_lab[..., c].std() + 1e-6
        # blend: 0.0=원래 그대로, 1.0=완전 매칭
        scale = 1 + blend * (std_r / std_g - 1)
        shift = blend * (mu_r - mu_g * scale)
        matched[..., c] = matched[..., c] * scale + shift

    matched = np.clip(matched, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)


def postprocess_video(
    input_video: str,
    output_video: str,
    gfpgan_model: Optional[str] = None,
    reference_video: Optional[str] = None,
    color_match_blend: float = 0.0,
    use_fp16: bool = True,
    weight: float = 0.5,
) -> bool:
    """입력 영상에 GFPGAN + (선택) color matching 적용 후 출력.

    Args:
      input_video       : LatentSync 출력 mp4 (lipsync 적용된 것)
      output_video      : 후처리 결과 mp4
      gfpgan_model      : GFPGANv1.4.pth 경로. None이면 GFPGAN 스킵
      reference_video   : 원본 영상 (color matching 시 비교 대상)
      color_match_blend : 0.0~1.0. 0.5=원본과 절반 정도 색감 일치
      weight            : GFPGAN 강도. 0.5=원본과 복원의 균형
    """
    import cv2

    if not Path(input_video).is_file():
        print(f"[ERR] 입력 없음: {input_video}")
        return False

    # GFPGAN 로드
    restorer = None
    if gfpgan_model and Path(gfpgan_model).is_file():
        restorer, use_fp16 = load_gfpgan(gfpgan_model, use_fp16)
    else:
        print("[Info] GFPGAN 비활성화 (model 없음 또는 미지정)")

    # Reference video 로드 (color match 시)
    ref_cap = None
    if color_match_blend > 0 and reference_video and Path(reference_video).is_file():
        ref_cap = cv2.VideoCapture(reference_video)
        print(f"[ColorMatch] reference 로드: {reference_video}, blend={color_match_blend}")
    elif color_match_blend > 0:
        print(f"[Info] color match 요청됐으나 reference 없음 — 비활성화")
        color_match_blend = 0

    # 입력 영상 처리
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_dir = tempfile.mkdtemp(prefix="lipsync_post_")
    tmp_silent = str(Path(tmp_dir) / "out_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_silent, fourcc, fps, (w, h))

    print(f"[Postprocess] 시작: {n_frames} frames @ {fps}fps, {w}x{h}")
    t0 = time.time()
    fp16_failed = False
    n_face_failed = 0

    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Reference frame (color match)
        ref_frame = None
        if ref_cap is not None:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_r, ref_frame = ref_cap.read()
            if not ret_r:
                ref_frame = None

        # 1. Color matching (GFPGAN 적용 전, 색감 미리 맞춤)
        if color_match_blend > 0 and ref_frame is not None:
            try:
                # ref와 같은 크기로 맞춤 (다르면 skip)
                if ref_frame.shape == frame.shape:
                    frame = color_match_at_boundary(frame, ref_frame, blend=color_match_blend)
            except Exception as e:
                if idx == 0:
                    print(f"[ColorMatch] 첫 frame 실패 ({e}) — 비활성화")
                    color_match_blend = 0

        # 2. GFPGAN face restoration
        if restorer is not None:
            try:
                _, _, restored = restorer.enhance(
                    frame, has_aligned=False, only_center_face=False,
                    paste_back=True, weight=weight,
                )
                if restored is not None:
                    frame = restored
            except RuntimeError as e:
                err_str = str(e)
                # fp16 dtype 충돌 자동 fallback
                is_dtype_err = (
                    use_fp16 and not fp16_failed and (
                        "FloatTensor" in err_str or "HalfTensor" in err_str
                        or "dtype" in err_str.lower() or "should be the same" in err_str
                    )
                )
                if is_dtype_err:
                    print(f"[GFPGAN] fp16 dtype 충돌 (frame {idx}) → fp32 fallback")
                    try:
                        if hasattr(restorer, "gfpgan"):
                            restorer.gfpgan = restorer.gfpgan.float()
                        if hasattr(restorer, "face_helper"):
                            if getattr(restorer.face_helper, "face_det", None) is not None:
                                restorer.face_helper.face_det = restorer.face_helper.face_det.float()
                            if getattr(restorer.face_helper, "face_parse", None) is not None:
                                restorer.face_helper.face_parse = restorer.face_helper.face_parse.float()
                        use_fp16 = False
                        fp16_failed = True
                        # 같은 frame 다시
                        _, _, restored = restorer.enhance(
                            frame, has_aligned=False, only_center_face=False,
                            paste_back=True, weight=weight,
                        )
                        if restored is not None:
                            frame = restored
                    except Exception:
                        n_face_failed += 1
                else:
                    n_face_failed += 1
            except Exception:
                n_face_failed += 1

        writer.write(frame)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = (n_frames - idx - 1) * elapsed / (idx + 1)
            print(f"[Postprocess] {idx+1}/{n_frames} ({100*(idx+1)/n_frames:.1f}%) "
                  f"— ETA {eta:.0f}s, avg {elapsed/(idx+1)*1000:.0f}ms/frame",
                  flush=True)

    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"[Postprocess] 프레임 처리 완료 ({elapsed:.1f}s)")
    if n_face_failed:
        print(f"[Postprocess] 얼굴 실패 frame: {n_face_failed} (원본 유지)")

    # ffmpeg로 오디오 + H.264 재인코딩
    print(f"[Postprocess] ffmpeg 인코딩...")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_silent,
        "-i", input_video,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        output_video,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:500] if e.stderr else str(e)
        print(f"[Postprocess] ffmpeg 실패: {err}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out_size = Path(output_video).stat().st_size / (1024 * 1024)
    print(f"[Postprocess] ✅ 완료: {output_video} ({out_size:.1f}MB)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="LatentSync 출력 mp4")
    parser.add_argument("--output", required=True, help="후처리 결과 mp4")
    parser.add_argument("--gfpgan-model",
                        default="/workspace/media/model_cache/musetalk/models/gfpgan/GFPGANv1.4.pth",
                        help="GFPGAN 가중치 경로 (없으면 자동 다운로드 영역 확인)")
    parser.add_argument("--no-gfpgan", action="store_true", help="GFPGAN 비활성화")
    parser.add_argument("--reference", default=None, help="원본 영상 (color match용)")
    parser.add_argument("--color-match", type=float, default=0.0,
                        help="0.0~1.0. 0.0=비활성화, 0.5=권장, 1.0=완전 매칭")
    parser.add_argument("--weight", type=float, default=0.5,
                        help="GFPGAN 강도. 0.0=원본 / 0.5=균형 / 1.0=완전 복원")
    parser.add_argument("--no-fp16", action="store_true",
                        help="GFPGAN fp16 비활성화 (호환성 문제 시)")
    args = parser.parse_args()

    # GFPGAN model 자동 탐색 (없으면 None으로)
    gfpgan_model = None if args.no_gfpgan else args.gfpgan_model
    if gfpgan_model and not Path(gfpgan_model).is_file():
        # 대체 경로들 탐색
        alts = [
            "/workspace/media/model_cache/gfpgan/GFPGANv1.4.pth",
            "/workspace/media/model_cache/musetalk/models/gfpgan/GFPGANv1.4.pth",
        ]
        for alt in alts:
            if Path(alt).is_file():
                gfpgan_model = alt
                print(f"[Postprocess] GFPGAN 자동 탐지: {gfpgan_model}")
                break
        else:
            print(f"[Postprocess] GFPGAN 모델 없음 — 비활성화 (다운로드: "
                  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)")
            gfpgan_model = None

    ok = postprocess_video(
        input_video=args.input,
        output_video=args.output,
        gfpgan_model=gfpgan_model,
        reference_video=args.reference,
        color_match_blend=args.color_match,
        use_fp16=not args.no_fp16,
        weight=args.weight,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
