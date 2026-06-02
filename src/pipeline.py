from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import soundfile as sf

from build_master_timeline import build_master_timeline
from common import build_dub_runtime_settings, deep_get, ensure_project_layout, get_logger, load_config, require_value, resolve_project_path, run_command, select_audio_path
from compose_audio import compose_audio
from cut_chunks import cut_chunks
from diarize import diarize_audio
from extract_emotion import extract_chunk_emotions
from extract_audio import extract_audio
from generate_tts_instructions import generate_tts_instructions
from merge_speaker_chunks import merge_speaker_chunks
from redirect_nonspeech import redirect_nonspeech_to_bgm
from rttm_to_json import convert_rttm_to_json
from run_asr import transcribe_chunks
from stabilize_diarization import stabilize_diarization_file
from run_tts import synthesize_dub_chunks
from separate_audio import separate_audio
from translate_chunks import build_translation_entries
from validate_tts_output import validate_tts_output

logger = get_logger("pipeline")


def _audio_format_matches(path: str, *, sample_rate: int, channels: int) -> bool:
    audio_path = resolve_project_path(path)
    if not audio_path.exists():
        return False
    with sf.SoundFile(str(audio_path)) as handle:
        return int(handle.samplerate) == sample_rate and int(handle.channels) == channels


def _final_audio_sample_rate(config: dict) -> int:
    default_rate = int(deep_get(config, ("audio", "source_sample_rate"), 44100)) if bool(deep_get(config, ("pipeline", "use_separator"), False)) else int(deep_get(config, ("audio", "sample_rate"), 16000))
    return int(deep_get(config, ("audio", "final_sample_rate"), default_rate))


def _final_audio_channels(config: dict) -> int:
    default_channels = int(deep_get(config, ("audio", "source_channels"), 2)) if bool(deep_get(config, ("pipeline", "use_separator"), False)) else int(deep_get(config, ("audio", "channels"), 1))
    return int(deep_get(config, ("audio", "final_channels"), default_channels))


def step_extract_audio(config: dict) -> None:
    use_separator = bool(deep_get(config, ("pipeline", "use_separator"), False))
    sample_rate = int(deep_get(config, ("audio", "sample_rate"), 16000))
    channels = int(deep_get(config, ("audio", "channels"), 1))
    if use_separator:
        sample_rate = int(deep_get(config, ("audio", "source_sample_rate"), 44100))
        channels = int(deep_get(config, ("audio", "source_channels"), 2))
    extract_audio(
        require_value(config, ("input_video",)),
        require_value(config, ("paths", "raw_audio")),
        sample_rate=sample_rate,
        channels=channels,
    )


def step_separate_audio(config: dict) -> None:
    raw_audio = require_value(config, ("paths", "raw_audio"))
    use_separator = bool(deep_get(config, ("pipeline", "use_separator"), False))
    if use_separator:
        source_sample_rate = int(deep_get(config, ("audio", "source_sample_rate"), 44100))
        source_channels = int(deep_get(config, ("audio", "source_channels"), 2))
        if not _audio_format_matches(raw_audio, sample_rate=source_sample_rate, channels=source_channels):
            logger.info(
                "Re-extracting raw audio for separator at %s Hz / %s ch",
                source_sample_rate,
                source_channels,
            )
            extract_audio(
                require_value(config, ("input_video",)),
                raw_audio,
                sample_rate=source_sample_rate,
                channels=source_channels,
            )
    ensemble_models = list(deep_get(config, ("audio", "subtractive_ensemble", "models"), []) or [])
    ensemble_weights = list(deep_get(config, ("audio", "subtractive_ensemble", "weights"), []) or [])
    ensemble_model_dir = str(deep_get(config, ("audio", "subtractive_ensemble", "model_dir"), "models/separation"))
    separate_audio_kwargs = {
        "use_separator": use_separator,
        "bgm_wav": deep_get(config, ("paths", "bgm_audio")),
        "ensemble_model_dir": ensemble_model_dir,
        "device": str(deep_get(config, ("runtime", "device"), "cuda:0")),
    }
    if ensemble_models:
        separate_audio_kwargs["ensemble_models"] = ensemble_models
    if ensemble_weights:
        separate_audio_kwargs["ensemble_weights"] = ensemble_weights
    separate_audio(
        raw_audio,
        require_value(config, ("paths", "dialogue_audio")),
        **separate_audio_kwargs,
    )


def step_redirect_nonspeech(config: dict) -> None:
    if not bool(deep_get(config, ("pipeline", "redirect_nonspeech"), False)):
        logger.info("Skipping redirect_nonspeech because pipeline.redirect_nonspeech=false")
        return
    bgm_audio = deep_get(config, ("paths", "bgm_audio"))
    if not bgm_audio:
        logger.info("Skipping redirect_nonspeech because paths.bgm_audio is not configured")
        return
    redirect_nonspeech_to_bgm(
        require_value(config, ("paths", "dialogue_audio")),
        bgm_audio,
        model_path=str(deep_get(config, ("redirect_nonspeech", "model_path"), "models/vad/silero_vad.jit")),
        threshold=float(deep_get(config, ("redirect_nonspeech", "threshold"), 0.4)),
        min_speech_ms=int(deep_get(config, ("redirect_nonspeech", "min_speech_ms"), 250)),
        pad_ms=int(deep_get(config, ("redirect_nonspeech", "pad_ms"), 100)),
        ramp_ms=int(deep_get(config, ("redirect_nonspeech", "ramp_ms"), 15)),
    )


def step_diarize(config: dict) -> None:
    # fusion_4way/team engines 는 자체 데몬/모델 사용 — diarizen 전용 model 경로는 lazy(deep_get).
    diarization_kwargs = {
        "model_dir": deep_get(config, ("models", "diarization")),
        "embedding_model_dir": deep_get(config, ("models", "diarization_embedding")),
        "device": str(deep_get(config, ("runtime", "device"), "cuda:0")),
        "max_speakers": deep_get(config, ("diarization", "max_speakers")),
        "min_speakers": deep_get(config, ("diarization", "min_speakers")),
        "ahc_threshold": deep_get(config, ("diarization", "ahc_threshold")),
        "ahc_criterion": deep_get(config, ("diarization", "ahc_criterion")),
        "fa": deep_get(config, ("diarization", "fa")),
        "fb": deep_get(config, ("diarization", "fb")),
        "lda_dim": deep_get(config, ("diarization", "lda_dim")),
        "max_iters": deep_get(config, ("diarization", "max_iters")),
        "method": deep_get(config, ("diarization", "method")),
        "min_cluster_size": deep_get(config, ("diarization", "min_cluster_size")),
        "seg_duration": deep_get(config, ("diarization", "seg_duration")),
        "segmentation_step": deep_get(config, ("diarization", "segmentation_step")),
        "batch_size": deep_get(config, ("diarization", "batch_size")),
        "apply_median_filtering": deep_get(config, ("diarization", "apply_median_filtering")),
    }
    input_audio = select_audio_path(config, "dialogue_audio")
    output_rttm = require_value(config, ("paths", "diarization_rttm"))
    engine = str(deep_get(config, ("diarization", "engine"), "diarizen")).strip().lower()
    if engine in {"team_refiner", "team_v195", "v195"}:
        from team_diarization import diarize_with_team_refiner

        diarize_with_team_refiner(
            input_audio,
            output_rttm,
            report_json=deep_get(config, ("paths", "diarization_refiner_report")),
            refiner_config=deep_get(config, ("diarization", "team_refiner"), {}) or {},
            **diarization_kwargs,
        )
        return
    if engine in {"fusion_4way", "preserved_fusion"}:
        # 검증된 4-way fusion daemon (port 8903) 호출. test4/test5 sweep best 적용.
        from preserved_fusion import diarize_with_4way_fusion

        diarize_with_4way_fusion(
            input_audio,
            output_rttm,
            num_speakers=deep_get(config, ("diarization", "num_speakers")),
            min_duration=float(deep_get(config, ("diarization", "min_duration"), 0.3)),
            fusion_url=deep_get(config, ("diarization", "fusion_url")),
            timeout=int(deep_get(config, ("diarization", "fusion_timeout"), 600)),
        )
        return
    if engine != "diarizen":
        raise ValueError(f"Unsupported diarization.engine: {engine}")
    diarize_audio(input_audio, output_rttm, **diarization_kwargs)


def _diarization_stabilization_enabled(config: dict) -> bool:
    return bool(deep_get(config, ("diarization", "stabilization", "enabled"), False))


def _diarization_json_for_merge(config: dict) -> str:
    if _diarization_stabilization_enabled(config):
        return require_value(config, ("paths", "diarization_stabilized_json"))
    return require_value(config, ("paths", "diarization_json"))


def step_rttm_to_json(config: dict) -> None:
    convert_rttm_to_json(
        require_value(config, ("paths", "diarization_rttm")),
        require_value(config, ("paths", "diarization_json")),
    )
    if _diarization_stabilization_enabled(config):
        stabilize_diarization_file(
            require_value(config, ("paths", "diarization_json")),
            require_value(config, ("paths", "diarization_stabilized_json")),
            same_speaker_gap_sec=float(deep_get(config, ("diarization", "stabilization", "same_speaker_gap_sec"), 0.1)),
            bridge_max_sec=float(deep_get(config, ("diarization", "stabilization", "bridge_max_sec"), 0.4)),
            bridge_gap_sec=float(deep_get(config, ("diarization", "stabilization", "bridge_gap_sec"), 0.25)),
            tiny_segment_sec=float(deep_get(config, ("diarization", "stabilization", "tiny_segment_sec"), 0.12)),
            absorb_gap_sec=float(deep_get(config, ("diarization", "stabilization", "absorb_gap_sec"), 0.25)),
            max_passes=int(deep_get(config, ("diarization", "stabilization", "max_passes"), 5)),
        )


def step_merge_chunks(config: dict) -> None:
    merge_speaker_chunks(
        _diarization_json_for_merge(config),
        require_value(config, ("paths", "speaker_chunks_json")),
        gap_threshold=float(deep_get(config, ("chunking", "speaker_merge_gap_sec"), 0.5)),
        min_chunk_sec=float(deep_get(config, ("chunking", "min_chunk_sec"), 0.0)),
        max_chunk_sec=float(deep_get(config, ("chunking", "max_chunk_sec"), 0.0)),
    )


def step_cut_chunks(config: dict) -> None:
    cut_chunks(
        deep_get(config, ("audio", "chunk_source"), require_value(config, ("paths", "dialogue_audio"))),
        require_value(config, ("paths", "speaker_chunks_json")),
        chunks_dir=require_value(config, ("paths", "chunks_dir")),
    )


def step_run_asr(config: dict) -> None:
    # 검증된 boost subchunk 옵션 (test5: 141 → 156 words, +15 fresh, Adam x2 detect)
    boost_cfg = deep_get(config, ("asr", "boost_subchunk")) or None
    transcribe_chunks(
        require_value(config, ("paths", "speaker_chunks_json")),
        require_value(config, ("paths", "asr_json")),
        model_dir=require_value(config, ("models", "asr")),
        device=str(deep_get(config, ("runtime", "device"), "cuda:0")),
        dtype=str(deep_get(config, ("runtime", "dtype"), "float16")),
        boost_subchunk=boost_cfg,
    )


def step_face_clustering(config: dict) -> None:
    # 신규 face service 에서 LightASD + face cluster + SPK remap. config 로 on/off.
    fc_cfg = deep_get(config, ("pipeline", "face_clustering"), {}) or {}
    if not bool(fc_cfg.get("enabled", False)):
        logger.info("Skipping face_clustering (pipeline.face_clustering.enabled=false)")
        return
    from face_clustering import cluster_faces_in_run
    import shutil as _shutil

    # face detection 은 전체 소스 비디오(단일 chunk) 기준. 모듈 cut_chunks 는 화자-turn 오디오라
    # 비디오 chunk 가 없음 → input_video 를 chunks_dir 에 단일 mp4 로 제공(없을 때만).
    stem = Path(str(config.get("input_video", "input"))).stem or "input"
    chunks_dir_p = resolve_project_path(require_value(config, ("paths", "chunks_dir")))
    chunks_dir_p.mkdir(parents=True, exist_ok=True)
    if not any(not v.stem.endswith("_final") for v in chunks_dir_p.glob("*.mp4")):
        src_video = resolve_project_path(require_value(config, ("input_video",)))
        _shutil.copyfile(src_video, chunks_dir_p / f"{stem}_full.mp4")
        logger.info("face_clustering: source video -> %s/%s_full.mp4", chunks_dir_p, stem)
    # stabilization 비활성 시 stabilized.json(미생성) 대신 diarization_json 사용.
    diar_for_face = (require_value(config, ("paths", "diarization_stabilized_json"))
                     if _diarization_stabilization_enabled(config)
                     else require_value(config, ("paths", "diarization_json")))
    # {input_stem} 치환(config 에 명시 경로 없을 때 default 가 리터럴로 남는 것 방지 → 브리지와 경로 일치).
    fc_json = str(deep_get(config, ("paths", "face_clusters_json"),
                           "meta/{input_stem}/face_clusters.json")).replace("{input_stem}", stem)
    fm_json = str(deep_get(config, ("paths", "diarization_face_matched_json"),
                           "meta/{input_stem}/diarization_face_matched.json")).replace("{input_stem}", stem)
    cluster_faces_in_run(
        str(chunks_dir_p),
        diar_for_face,
        fc_json,
        fm_json,
        light_asd_dir=str(fc_cfg.get("light_asd_dir", "/opt/Light-ASD")),
        venv_python=str(fc_cfg.get("venv_python", "/usr/bin/python")),
        face_sim_threshold=float(fc_cfg.get("face_sim_threshold", 0.4)),
        min_speak_score=float(fc_cfg.get("min_speak_score", 0.5)),
        dominant_ratio=float(fc_cfg.get("dominant_ratio", 0.5)),
        min_evidence_frames=int(fc_cfg.get("min_evidence_frames", 5)),
    )


def step_build_repair_inputs(config: dict) -> None:
    # 모듈 stage 출력(fusion diar + vocals + faces.json + 전체영상 ASR)을 preserved_repair.run_dir
    # 형식으로 채움 → 검증된 4-way + 8 repair patch 가 모듈 pipeline 안에서 그대로 동작(webUI 소비 가능).
    repair_cfg = deep_get(config, ("preserved_repair",)) or {}
    if not bool(repair_cfg.get("enabled", False)):
        logger.info("Skipping build_repair_inputs (preserved_repair.enabled=false)")
        return
    run_dir = repair_cfg.get("run_dir")
    if not run_dir:
        logger.warning("preserved_repair.run_dir not set; skipping build_repair_inputs")
        return
    from build_repair_inputs import build_repair_inputs

    stem = Path(str(config.get("input_video", "input"))).stem or "input"
    fc_json = str(deep_get(config, ("paths", "face_clusters_json"),
                           "meta/{input_stem}/face_clusters.json")).replace("{input_stem}", stem)
    faces_src = str(resolve_project_path(fc_json).parent / "faces.json")
    build_repair_inputs(
        str(resolve_project_path(run_dir)),
        stem=stem,
        vocals_src=str(select_audio_path(config, "dialogue_audio")),
        diarization_json=str(resolve_project_path(require_value(config, ("paths", "diarization_json")))),
        faces_src=faces_src,
        video_src=str(resolve_project_path(require_value(config, ("input_video",)))),
        asr_url=str(deep_get(config, ("asr", "daemon_url"), "http://127.0.0.1:8902")),
        asr_language=str(deep_get(config, ("asr", "language"),
                                  deep_get(config, ("translation", "source_language"), "English"))),
    )


def step_apply_preserved_repair(config: dict) -> None:
    # E:\TTS_capstone 검증된 8 repair patches 일괄 호출.
    # config.preserved_repair.run_dir = run 디렉토리 (meta/ + vocals/ 필요).
    # config.preserved_repair.gap_fill = {main_merge, bg_merge, sim_match, pad}.
    # config.preserved_repair.skip = ['word_level_split', ...] (optional).
    repair_cfg = deep_get(config, ("preserved_repair",)) or {}
    if not bool(repair_cfg.get("enabled", False)):
        logger.info("Skipping preserved_repair (preserved_repair.enabled=false)")
        return
    run_dir = repair_cfg.get("run_dir")
    if not run_dir:
        logger.warning("preserved_repair.run_dir not set; skipping")
        return
    from apply_repair_patches import apply_all
    gf = repair_cfg.get("gap_fill") or {}
    apply_all(
        str(resolve_project_path(run_dir)),
        gap_fill_args={
            "main_merge": float(gf.get("main_merge", 0.45)),
            "bg_merge": float(gf.get("bg_merge", 0.30)),
            "sim_match": float(gf.get("sim_match", 0.45)),
            "pad": float(gf.get("pad", 0.5)),
        },
        skip=repair_cfg.get("skip") or [],
        venv_python=repair_cfg.get("venv_python") or "/opt/venv_diarizen/bin/python",
    )


def step_apply_gapfilled(config: dict) -> None:
    # 검증된 gapfilled 화자분리(run_dir/meta/*_segments_gapfilled.json, BG화자·gap회수 포함)를
    # 청크 단계 입력(diarization_json)으로 export → merge_speaker_chunks 가 검증 화자 turn 을 청크로 묶음.
    repair_cfg = deep_get(config, ("preserved_repair",)) or {}
    if not bool(repair_cfg.get("enabled", False)):
        logger.info("Skipping apply_gapfilled (preserved_repair.enabled=false)")
        return
    run_dir = repair_cfg.get("run_dir")
    if not run_dir:
        logger.warning("preserved_repair.run_dir not set; skipping apply_gapfilled")
        return
    from apply_gapfilled_diarization import apply_gapfilled_to_diarization

    stabilized = (
        require_value(config, ("paths", "diarization_stabilized_json"))
        if _diarization_stabilization_enabled(config)
        else None
    )
    apply_gapfilled_to_diarization(
        str(resolve_project_path(run_dir)),
        require_value(config, ("paths", "diarization_json")),
        stabilized_json=stabilized,
    )


def step_extract_emotion(config: dict) -> None:
    extract_chunk_emotions(
        require_value(config, ("paths", "speaker_chunks_json")),
        require_value(config, ("paths", "emotion_json")),
        model_ref=str(deep_get(config, ("models", "emotion"), "iic/emotion2vec_plus_large")),
        backend=str(deep_get(config, ("emotion", "backend"), "funasr")),
        device=str(deep_get(config, ("runtime", "device"), "cuda:0")),
        skip_existing=bool(deep_get(config, ("emotion", "skip_existing"), True)),
        funasr_output_dir=deep_get(config, ("emotion", "funasr_output_dir")),
    )


def step_translate(config: dict) -> None:
    build_translation_entries(
        require_value(config, ("paths", "asr_json")),
        require_value(config, ("paths", "translated_json")),
        mode=str(deep_get(config, ("translation", "mode"), "copy_source")),
        source_language=str(deep_get(config, ("translation", "source_language"), "en")),
        target_language=str(deep_get(config, ("translation", "target_language"), "ko")),
        env_file=str(deep_get(config, ("translation", "env_file"), ".env")),
        timeout_sec=int(deep_get(config, ("translation", "timeout_sec"), 60)),
        context_refine=bool(deep_get(config, ("translation", "context_refine"), True)),
        context_batch_size=int(deep_get(config, ("translation", "context_batch_size"), 12)),
        duration_control=bool(deep_get(config, ("translation", "duration_control"), True)),
        max_budget_rewrites=int(deep_get(config, ("translation", "max_budget_rewrites"), 2)),
        scene_context=str(deep_get(config, ("translation", "scene_context"), "")),
        register_override=str(deep_get(config, ("translation", "register"), "")),
        auto_scene_context=bool(deep_get(config, ("translation", "auto_scene_context"), False)),
    )


def _build_dub_runtime_from_config(config: dict) -> dict:
    translation_mode = str(deep_get(config, ("translation", "mode"), "copy_source"))
    source_language = str(deep_get(config, ("translation", "source_language"), ""))
    target_language = str(deep_get(config, ("translation", "target_language"), ""))
    tts_engine = str(deep_get(config, ("tts", "engine"), "cosyvoice")).strip().lower()
    passthrough_source_audio = bool(deep_get(config, ("tts", "passthrough_on_copy_source"), True)) and translation_mode == "copy_source"
    use_cross_lingual = bool(source_language.strip()) and bool(target_language.strip()) and source_language.strip().lower() != target_language.strip().lower()
    runtime_kwargs = {
        "engine": tts_engine,
        "model_dir": str(require_value(config, ("models", "tts"))),
        "target_language": target_language,
        "reference_mode": str(deep_get(config, ("tts", "reference_mode"), "auto")),
        "min_prompt_sec": float(deep_get(config, ("tts", "min_prompt_sec"), 1.2)),
        "fit_to_duration": bool(deep_get(config, ("tts", "fit_to_duration"), False)),
        "passthrough_source_audio": passthrough_source_audio,
        "use_cross_lingual": use_cross_lingual,
        "system_prompt": str(deep_get(config, ("tts", "system_prompt"), "")),
        "speed": float(deep_get(config, ("tts", "speed"), 1.0)),
        "trim_silence": bool(deep_get(config, ("tts", "trim_silence"), False)),
        "silence_trim_threshold_dbfs": float(deep_get(config, ("tts", "silence_trim_threshold_dbfs"), -45.0)),
        "max_leading_silence_sec": float(deep_get(config, ("tts", "max_leading_silence_sec"), 0.1)),
        "max_trailing_silence_sec": float(deep_get(config, ("tts", "max_trailing_silence_sec"), 0.2)),
        "style_priority": str(deep_get(config, ("tts", "style_priority"), "instruction")),
    }
    return build_dub_runtime_settings(**runtime_kwargs)


def step_build_timeline(config: dict) -> None:
    build_master_timeline(
        require_value(config, ("paths", "speaker_chunks_json")),
        require_value(config, ("paths", "asr_json")),
        require_value(config, ("paths", "translated_json")),
        require_value(config, ("paths", "master_timeline_json")),
        dub_dir=require_value(config, ("paths", "dub_dir")),
        emotion_json=deep_get(config, ("paths", "emotion_json")),
        chunk_overrides_json=deep_get(config, ("paths", "chunk_overrides_json")),
        dub_runtime=_build_dub_runtime_from_config(config),
    )


def step_generate_tts_instructions(config: dict) -> None:
    # style_priority='voice' 는 instruct 자체를 무시하는 별도 모드 — instruction 생성 스킵
    style_priority = str(deep_get(config, ("tts", "style_priority"), "")).strip().lower()
    if style_priority == "voice":
        logger.info("Skipping TTS instruction generation because tts.style_priority=voice")
        return
    instruction_config = deep_get(config, ("tts", "instruction"), {}) or {}
    if isinstance(instruction_config, dict) and not bool(instruction_config.get("enabled", True)):
        logger.info("Skipping TTS instruction generation because tts.instruction.enabled=false")
        return
    generate_tts_instructions(
        require_value(config, ("paths", "master_timeline_json")),
        mode=str(deep_get(config, ("tts", "instruction", "mode"), "vectorengine_gpt")),
        env_file=str(deep_get(config, ("tts", "instruction", "env_file"), deep_get(config, ("translation", "env_file"), ".env"))),
        timeout_sec=int(deep_get(config, ("tts", "instruction", "timeout_sec"), deep_get(config, ("translation", "timeout_sec"), 60))),
        skip_existing=bool(deep_get(config, ("tts", "instruction", "skip_existing"), True)),
        fallback_on_error=bool(deep_get(config, ("tts", "instruction", "fallback_on_error"), True)),
        batch_size=int(deep_get(config, ("tts", "instruction", "batch_size"), 6)),
    )


def step_run_tts(config: dict) -> None:
    translation_mode = str(deep_get(config, ("translation", "mode"), "copy_source"))
    source_language = str(deep_get(config, ("translation", "source_language"), ""))
    target_language = str(deep_get(config, ("translation", "target_language"), ""))
    tts_engine = str(deep_get(config, ("tts", "engine"), "cosyvoice")).strip().lower()
    if tts_engine != "cosyvoice":
        raise ValueError(f"Unsupported TTS engine after cleanup: {tts_engine}")
    passthrough_source_audio = bool(deep_get(config, ("tts", "passthrough_on_copy_source"), True)) and translation_mode == "copy_source"
    use_cross_lingual = bool(source_language.strip()) and bool(target_language.strip()) and source_language.strip().lower() != target_language.strip().lower()
    synthesize_dub_chunks(
        require_value(config, ("paths", "master_timeline_json")),
        model_dir=require_value(config, ("models", "tts")),
        cosyvoice_repo=deep_get(config, ("models", "cosyvoice_repo")),
        system_prompt=str(deep_get(config, ("tts", "system_prompt"), "")),
        stream=bool(deep_get(config, ("tts", "stream"), False)),
        skip_existing=bool(deep_get(config, ("tts", "skip_existing"), True)),
        min_prompt_sec=float(deep_get(config, ("tts", "min_prompt_sec"), 1.2)),
        passthrough_source_audio=passthrough_source_audio,
        use_cross_lingual=use_cross_lingual,
        target_language=target_language,
        fit_to_duration=bool(deep_get(config, ("tts", "fit_to_duration"), False)),
        engine=tts_engine,
        device=str(deep_get(config, ("runtime", "device"), "cuda:0")),
        speed=float(deep_get(config, ("tts", "speed"), 1.0)),
        reference_mode=str(deep_get(config, ("tts", "reference_mode"), "auto")),
        compact_timeline=bool(deep_get(config, ("tts", "compact_timeline"), False)),
        duration_fit_min_tempo=float(deep_get(config, ("tts", "duration_fit_min_tempo"), 0.85)),
        duration_fit_max_tempo=float(deep_get(config, ("tts", "duration_fit_max_tempo"), 1.2)),
        duration_fit_trim_overlong=bool(deep_get(config, ("tts", "duration_fit_trim_overlong"), False)),
        trim_silence=bool(deep_get(config, ("tts", "trim_silence"), False)),
        silence_trim_threshold_dbfs=float(deep_get(config, ("tts", "silence_trim_threshold_dbfs"), -45.0)),
        max_leading_silence_sec=float(deep_get(config, ("tts", "max_leading_silence_sec"), 0.1)),
        max_trailing_silence_sec=float(deep_get(config, ("tts", "max_trailing_silence_sec"), 0.2)),
        cap_risky_self_reference=bool(deep_get(config, ("tts", "cap_risky_self_reference"), True)),
        prompt_cap_max_sec=float(deep_get(config, ("tts", "prompt_cap_max_sec"), 4.5)),
        style_priority=str(deep_get(config, ("tts", "style_priority"), "instruction")),
    )


def step_validate_tts(config: dict) -> None:
    validation_config = deep_get(config, ("tts", "validation"), {}) or {}
    if not bool(deep_get(config, ("tts", "validation", "enabled"), True)):
        logger.info("Skipping TTS ASR validation because tts.validation.enabled=false")
        return
    master_timeline_json = str(require_value(config, ("paths", "master_timeline_json")))
    output_json = deep_get(config, ("paths", "tts_validation_json"))
    if not output_json:
        master_path = Path(master_timeline_json)
        tts_engine = str(deep_get(config, ("tts", "engine"), "cosyvoice")).strip().lower() or "cosyvoice"
        output_json = str(master_path.with_name(f"tts_validation_{tts_engine}.json"))
    validate_tts_output(
        master_timeline_json,
        output_json=output_json,
        model_dir=require_value(config, ("models", "asr")),
        device=str(deep_get(config, ("runtime", "device"), "cuda:0")),
        dtype=str(deep_get(config, ("runtime", "dtype"), "float16")),
        language=str(deep_get(config, ("tts", "validation", "language"), deep_get(config, ("translation", "target_language"), "Korean"))),
        mark_stale_on_fail=bool(validation_config.get("mark_stale_on_fail", True)),
        fail_on_error=bool(validation_config.get("fail_on_error", False)),
    )


def step_compose_audio(config: dict) -> None:
    background_audio = deep_get(config, ("paths", "bgm_audio"))
    compose_audio(
        require_value(config, ("paths", "master_timeline_json")),
        require_value(config, ("paths", "final_dub_audio")),
        sample_rate=_final_audio_sample_rate(config),
        channels=_final_audio_channels(config),
        background_wav=background_audio if background_audio and resolve_project_path(background_audio).exists() else None,
        background_gain=float(deep_get(config, ("audio", "background_gain"), 1.0)),
        dub_gain=float(deep_get(config, ("audio", "dub_gain"), 1.0)),
        target_peak_dbfs=float(deep_get(config, ("audio", "target_peak_dbfs"), -1.0)),
        per_chunk_peak_dbfs=(
            float(deep_get(config, ("audio", "per_chunk_peak_dbfs")))
            if deep_get(config, ("audio", "per_chunk_peak_dbfs")) is not None
            else None
        ),
    )


def step_mux(config: dict) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            resolve_project_path(require_value(config, ("input_video",))),
            "-i",
            resolve_project_path(require_value(config, ("paths", "final_dub_audio"))),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:a",
            str(deep_get(config, ("audio", "final_codec"), "aac")),
            "-b:a",
            str(deep_get(config, ("audio", "final_bitrate"), "192k")),
            "-ar",
            str(_final_audio_sample_rate(config)),
            "-ac",
            str(_final_audio_channels(config)),
            resolve_project_path(require_value(config, ("paths", "output_video"))),
        ]
    )


STEP_FUNCTIONS: list[tuple[str, Callable[[dict], None]]] = [
    ("extract_audio", step_extract_audio),
    ("separate_audio", step_separate_audio),
    ("redirect_nonspeech", step_redirect_nonspeech),
    ("diarize", step_diarize),
    ("rttm_to_json", step_rttm_to_json),
    ("face_clustering", step_face_clustering),
    ("build_repair_inputs", step_build_repair_inputs),
    ("apply_preserved_repair", step_apply_preserved_repair),
    ("apply_gapfilled", step_apply_gapfilled),
    ("merge_chunks", step_merge_chunks),
    ("cut_chunks", step_cut_chunks),
    ("extract_emotion", step_extract_emotion),
    ("run_asr", step_run_asr),
    ("translate", step_translate),
    ("build_timeline", step_build_timeline),
    ("generate_tts_instructions", step_generate_tts_instructions),
    ("run_tts", step_run_tts),
    ("validate_tts", step_validate_tts),
    ("compose_audio", step_compose_audio),
    ("mux", step_mux),
]


def select_steps(
    *,
    only: str | None,
    from_step: str | None,
    to_step: str | None,
) -> list[tuple[str, Callable[[dict], None]]]:
    names = [name for name, _ in STEP_FUNCTIONS]
    if only:
        if only not in names:
            raise ValueError(f"Unknown step: {only}")
        return [item for item in STEP_FUNCTIONS if item[0] == only]

    start_index = names.index(from_step) if from_step else 0
    end_index = names.index(to_step) if to_step else len(names) - 1
    if end_index < start_index:
        raise ValueError("to_step must be after from_step")
    return STEP_FUNCTIONS[start_index:end_index + 1]


def run_pipeline(
    config: dict,
    *,
    only: str | None = None,
    from_step: str | None = None,
    to_step: str | None = None,
) -> None:
    ensure_project_layout(config)
    stop_on_error = bool(deep_get(config, ("pipeline", "stop_on_error"), True))
    for step_name, handler in select_steps(only=only, from_step=from_step, to_step=to_step):
        logger.info("Starting step: %s", step_name)
        try:
            handler(config)
        except Exception as exc:
            logger.error("Step failed: %s | %s", step_name, exc)
            if stop_on_error:
                raise
            logger.exception("Continuing after failed step: %s", step_name)
        else:
            logger.info("Completed step: %s", step_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the movie dubbing pipeline.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--input-video")
    parser.add_argument("--only", choices=[name for name, _ in STEP_FUNCTIONS])
    parser.add_argument("--from-step", choices=[name for name, _ in STEP_FUNCTIONS])
    parser.add_argument("--to-step", choices=[name for name, _ in STEP_FUNCTIONS])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config, input_video=args.input_video)
    run_pipeline(
        config,
        only=args.only,
        from_step=args.from_step,
        to_step=args.to_step,
    )


if __name__ == "__main__":
    main()
