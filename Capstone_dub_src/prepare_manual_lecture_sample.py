from __future__ import annotations

import json
import subprocess
from pathlib import Path


SEGMENTS = [
    (
        "chunk_0001",
        0.031,
        9.397,
        "When I was 27 years old, I left a very demanding job in management consulting for a job that was even more demanding: teaching.",
        "스물일곱 살 때, 저는 매우 힘든 경영 컨설팅 일을 그만두고, 더 힘든 일인 교직을 선택했습니다.",
    ),
    (
        "chunk_0002",
        10.780,
        20.300,
        "I went to teach seventh graders math in the New York City public schools, and like any teacher, I made quizzes and tests.",
        "뉴욕시 공립학교에서 중학교 1학년 학생들에게 수학을 가르쳤고, 다른 교사들처럼 퀴즈와 시험을 만들었습니다.",
    ),
    (
        "chunk_0003",
        20.300,
        27.300,
        "I gave out homework assignments. When the work came back, I calculated grades.",
        "숙제도 내주었습니다. 과제가 돌아오면 저는 점수를 계산했습니다.",
    ),
    (
        "chunk_0004",
        27.300,
        38.500,
        "What struck me was that IQ was not the only difference between my best and my worst students.",
        "그때 인상 깊었던 것은, 가장 잘하는 학생과 가장 힘들어하는 학생의 차이가 IQ만은 아니었다는 점입니다.",
    ),
    (
        "chunk_0005",
        38.500,
        48.800,
        "Some of my strongest performers did not have stratospheric IQ scores. Some of my smartest kids weren't doing so well, and that got me thinking.",
        "성적이 아주 좋은 학생들 중에도 IQ가 엄청나게 높은 것은 아닌 아이들이 있었습니다. 반대로 아주 똑똑한 아이들 중 일부는 잘 해내지 못했습니다. 그래서 저는 생각하게 되었습니다.",
    ),
    (
        "chunk_0006",
        48.800,
        59.971,
        "The kinds of things you need to learn in seventh grade math - sure, they're hard: ratios, decimals, the area of a parallelogram - but these concepts are not impossible. And I was firmly convinced that every one of my students could learn.",
        "중학교 1학년 수학에서 배워야 하는 것들, 예를 들면 비율, 소수, 평행사변형의 넓이는 분명 어렵습니다. 하지만 불가능한 개념은 아닙니다. 저는 제 학생 모두가 배울 수 있다고 굳게 믿었습니다.",
    ),
]


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    chunks_dir = Path("chunks/강의샘플_scene_emotion")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    source_wav = "audio/강의샘플/dialogue.wav"

    speaker_rows = []
    asr_rows = []
    translated_rows = []
    for chunk_id, start, end, text_src, text_ko in SEGMENTS:
        duration = round(end - start, 3)
        wav = chunks_dir / f"{chunk_id}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                source_wav,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = {
            "chunk_id": chunk_id,
            "speaker": "SPEAKER_00",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": duration,
            "wav": str(wav).replace("\\", "/"),
        }
        speaker_rows.append({**base, "source_segment_count": 1})
        asr_rows.append({**base, "language": "English", "text_src": text_src})
        translated_rows.append(
            {
                "chunk_id": chunk_id,
                "speaker": "SPEAKER_00",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": duration,
                "text_src": text_src,
                "text_translated": text_ko,
                "text_tts": text_ko,
            }
        )

    save_json(Path("meta/강의샘플/speaker_chunks_scene_emotion.json"), speaker_rows)
    save_json(Path("meta/강의샘플/asr_scene_emotion.json"), asr_rows)
    save_json(Path("meta/강의샘플/translated_scene_emotion.json"), translated_rows)
    print(f"prepared {len(SEGMENTS)} lecture sample chunks")


if __name__ == "__main__":
    main()
