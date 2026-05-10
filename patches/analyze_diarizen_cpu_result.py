"""DiariZen CPU 결과를 우리 segment 정답과 비교."""

# DiariZen CPU 출력
diarizen_turns = [
    (5.49, 9.37, 2),
    (19.23, 20.35, 2),
    (20.35, 20.37, 1),
    (20.57, 22.41, 2),
    (22.41, 23.99, 1),
    (25.31, 26.71, 1),
    (27.31, 32.65, 1),
    (33.41, 37.19, 1),
    (37.67, 39.21, 1),
    (39.57, 40.85, 1),
    (42.13, 43.61, 1),
    (43.61, 43.65, 2),
    (44.15, 44.17, 2),
    (44.17, 44.79, 1),
    (44.79, 45.17, 2),
    (45.59, 46.71, 2),
    (47.17, 48.11, 2),
    (48.99, 50.21, 2),
    (50.81, 52.53, 2),
    (54.35, 57.13, 2),
    (61.51, 62.25, 2),
    (64.57, 65.25, 2),
    (73.71, 75.05, 2),
]

# 우리 segment + ground truth
segments = [
    (0, 5.52, 8.96, "M", "Welcome to first day..."),
    (1, 19.28, 23.6, "M", "You gonna help..."),
    (2, 25.36, 32.48, "M", "You are smart hard as hell..."),
    (3, 33.44, 35.52, "M", "No one gonna wanna watch..."),
    (4, 35.52, 43.04, "M", "I don't wanna see that..."),
    (5, 44.24, 47.36, "M", "Maybe couple Does"),
    (6, 49.12, 52.4, "M", "But not you not most..."),
    (7, 54.4, 65.12, "M", "We are gonna help solve..."),
    (8, 73.76, 74.96, "F", "What you need from me"),  # 여자!
]

print("=== Segment vs DiariZen (CPU) 비교 ===")
print(f"{'ID':>3} | {'Truth':>5} | {'Time':>15} | {'DiariZen Speakers':<25} | Text")
print("-" * 90)

for sid, ss, se, truth, text in segments:
    # 이 segment 시간대에 겹치는 DiariZen turns 찾기
    overlapping = []
    for ts, te, spk in diarizen_turns:
        # 겹치는지
        if te < ss or ts > se:
            continue
        # 겹친 시간 길이
        overlap = min(te, se) - max(ts, ss)
        if overlap > 0:
            overlapping.append((spk, overlap))
    if overlapping:
        # 가장 길게 겹친 화자
        from collections import defaultdict
        spk_total = defaultdict(float)
        for spk, ov in overlapping:
            spk_total[spk] += ov
        spk_str = " ".join(f"SPK{s}({d:.1f}s)" for s, d in sorted(spk_total.items()))
    else:
        spk_str = "(no detect)"
    print(f"  {sid} | {truth:>5} | {ss:5.1f}-{se:5.1f}s | {spk_str:<25} | {text[:35]}")

# 정확도 분석
print("\n=== 정확도 분석 ===")
print("Ground truth: 남자 8개 (ID 0-7), 여자 1개 (ID 8)")
print()

# 각 segment의 best_speaker 결정 (가장 많이 겹친 SPK)
seg_assignments = []
for sid, ss, se, truth, _ in segments:
    overlapping = []
    for ts, te, spk in diarizen_turns:
        overlap = min(te, se) - max(ts, ss)
        if overlap > 0:
            overlapping.append((spk, overlap))
    if not overlapping:
        seg_assignments.append((sid, truth, None))
        continue
    spk_total = {}
    for spk, ov in overlapping:
        spk_total[spk] = spk_total.get(spk, 0) + ov
    best = max(spk_total, key=spk_total.get)
    seg_assignments.append((sid, truth, best))

# 화자별 분포
spk_truth_map = {}
for sid, truth, spk in seg_assignments:
    if spk is None: continue
    spk_truth_map.setdefault(spk, []).append((sid, truth))

print("DiariZen이 분류한 화자 별:")
for spk in sorted(spk_truth_map.keys()):
    seg_list = spk_truth_map[spk]
    truths = [t for _, t in seg_list]
    print(f"  SPEAKER_{spk}: {len(seg_list)}개 segment, ground truth: {' '.join(truths)} (ID: {[s for s,_ in seg_list]})")

# 결론
print("\n결론:")
n_male_truth = 8
n_female_truth = 1
total = 9
correct_male_in_one_spk = max(
    (sum(1 for sid, truth in spk_truth_map.get(spk, []) if truth == "M"))
    for spk in spk_truth_map
)
print(f"  - 정답: 남자 {n_male_truth}, 여자 {n_female_truth}")
print(f"  - DiariZen detect: {len(spk_truth_map)} 화자")
print(f"  - 한 화자에 가장 많이 모인 남자: {correct_male_in_one_spk}/{n_male_truth}")
print(f"  - 여자 (ID 8) 분류:")
for sid, truth, spk in seg_assignments:
    if sid == 8:
        print(f"    ID 8 (여자) → SPEAKER_{spk}")
        break
