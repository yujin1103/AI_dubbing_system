# docs/ — 보조 문서 모음

루트 `README.md`와 `PIPELINE_OVERVIEW.md` 외 추가 문서를 카테고리별로 정리.

## 📂 구조

```
docs/
├── reports/             # 진행 상황 보고서
│   ├── PROJECT_STATE.md          # 5/4 LoRA 학습 완료 시점 상태
│   ├── PROJECT_STATUS_5_7.md     # 5/7 작업 내역
│   └── FINAL_REPORT_v15.md       # v15 시점 최종 보고
│
├── presentations/       # 발표 자료
│   ├── PIPELINE_PRESENTATION_SCRIPT.md   # 14 슬라이드 발표 대본 (18~22분)
│   ├── 중간발표_v2_대본.md
│   └── 최종발표_대본.md
│
├── diagrams_source/     # 다이어그램 재생성 소스
│   ├── pipeline_v3.html          # 메인 파이프라인 (SVG 직각 라우팅)
│   ├── screenshot_v3.js          # puppeteer 캡쳐 스크립트
│   └── poster_a4.html            # A4 포스터
│
└── old_script/          # 영문 자막 raw 파일 (참고용)
    ├── old_script.docx
    ├── old_script.txt
    └── old_script_raw.xml
```

## 다이어그램 재생성 방법

```bash
cd docs/diagrams_source

# 의존성 설치 (한 번만)
npm install puppeteer

# pipeline_v3.png 재생성
node screenshot_v3.js
# → 결과는 루트의 pipeline_v3.png 갱신

# poster_a4.png 재생성
# (screenshot_poster.js 필요 — archive/temp_scripts/ 참조)
```
