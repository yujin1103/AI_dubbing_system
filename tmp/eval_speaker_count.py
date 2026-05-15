#!/usr/bin/env python3
"""v20+ 화자 분리 자동 검증 도구.

ground truth (사용자 지정 발화 횟수)와 비교해 발화 횟수 미스매치 리포트.

사용법:
    python3 eval_speaker_count.py <segments_json> [<gt_json>]

기본 GT (test4.mp4): SPEAKER_05는 1회만 발화 (사용자 명시)
"""
import json
import sys
from pathlib import Path


# test4 영상 기본 ground truth (사용자 피드백 누적)
DEFAULT_GT = {
    "expected_speaker_count": 6,
    "speaker_constraints": {
        # 사용자 명시: SPEAKER_05는 영상 중 1번만 발화
        # 그 외 화자는 자유 (단 모두 1+ 발화)
        # "SPEAKER_05_max_groups": 1,  # 1번 발화 → 그룹 1개 정도 (split 고려 2 허용)
    },
    "_note": "test4 영상의 추정 화자 분포. 사용자가 새 GT를 주면 갱신할 것."
}


def evaluate(segments_json_path: str, gt_path: str = None):
    data = json.loads(Path(segments_json_path).read_text())
    groups = data.get("groups", [])

    if gt_path and Path(gt_path).exists():
        gt = json.loads(Path(gt_path).read_text())
    else:
        gt = DEFAULT_GT

    # 화자별 group 수
    from collections import Counter
    counts = Counter(g["speaker"] for g in groups)
    speakers = sorted(counts.keys())
    n_speakers = len(speakers)

    print(f"=== Speaker Diarization Evaluation ===")
    print(f"File: {segments_json_path}")
    print(f"Total groups: {len(groups)}")
    print(f"Speakers detected: {n_speakers}")
    print()
    print(f"{'Speaker':<14} {'Groups':>8} {'%':>6}")
    for spk in speakers:
        c = counts[spk]
        pct = c * 100 // max(1, len(groups))
        print(f"  {spk:<12} {c:>8} {pct:>5}%")
    print()

    # 평가
    issues = []
    expected_n = gt.get("expected_speaker_count", n_speakers)
    if n_speakers != expected_n:
        issues.append(f"❌ 화자 수: detected={n_speakers} vs expected={expected_n}")
    else:
        print(f"✅ 화자 수 일치: {n_speakers}")

    constraints = gt.get("speaker_constraints", {})
    for spk in speakers:
        max_key = f"{spk}_max_groups"
        min_key = f"{spk}_min_groups"
        if max_key in constraints and counts[spk] > constraints[max_key]:
            issues.append(f"❌ {spk}: {counts[spk]} groups > max={constraints[max_key]} (over-detect 의심)")
        if min_key in constraints and counts[spk] < constraints[min_key]:
            issues.append(f"❌ {spk}: {counts[spk]} groups < min={constraints[min_key]}")

    # 휴리스틱 검증: 1번만 발화한 SPEAKER (사용자 case)
    # SPEAKER_X가 영상 어딘가에 1회만 등장하는데 group이 3+ 라면 over-detect 강한 신호
    suspect = []
    for spk, c in counts.items():
        if c >= 3:
            # 모든 group의 group_start, group_end 확인 — 시간적으로 분산되어 있으면 일단 OK
            spk_groups = [g for g in groups if g["speaker"] == spk]
            time_span = max(g["group_end"] for g in spk_groups) - min(g["group_start"] for g in spk_groups)
            if time_span < 5.0:
                suspect.append(f"⚠️  {spk}: {c} groups, 5초 미만에 집중 (over-split 의심)")

    print()
    if issues:
        print("=== 발견된 이슈 ===")
        for i in issues:
            print(f"  {i}")
    else:
        print("✅ 제약사항 모두 통과")
    if suspect:
        print()
        print("=== 휴리스틱 의심 ===")
        for s in suspect:
            print(f"  {s}")

    return {"counts": dict(counts), "issues": issues, "suspect": suspect}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: eval_speaker_count.py <segments_json> [<gt_json>]")
        sys.exit(1)
    seg_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else None
    evaluate(seg_path, gt_path)
