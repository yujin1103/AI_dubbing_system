# -*- coding: utf-8 -*-
"""Light-ASD crop_video 속도 patch — 정확도 무영향 (bbox·ASD score 100% 동일).

병목 (crop_video, Columbia_test.py):
  - track 마다 `glob.glob(*.jpg)` + sort 를 반복 (test4=61 tracks → 61회 중복 glob).
  - track 들이 시간적으로 겹치면 같은 JPG 프레임을 여러 번 cv2.imread (중복 디스크 I/O).

수정 (영상 무관, 캐싱과 다름 — 같은 run 안에서만 재사용):
  1) flist(JPG 목록) 를 모듈 전역에 1회만 glob+sort → track 반복 제거.
  2) cv2.imread 를 LRU 캐시(maxsize=400)로 감싸 중복 프레임 read 제거.
     (컨테이너 RAM 7GB → 720p*400 ≈ 1.1GB 안전. 전체 2732장 캐시는 7.5GB라 불가.)
  결과 bbox/crop/ASD score 는 동일 (numpy.pad 는 복사본 반환 → 캐시 원본 불변).

사용: docker exec dubbing_pipeline /usr/bin/python /workspace/scripts/patch_lightasd_crop.py
환경변수 LIGHTASD_FRAME_CACHE (기본 400) 로 캐시 프레임 수 조정.
백업: /opt/Light-ASD/Columbia_test.py.bak_crop
"""
import shutil

COL = '/opt/Light-ASD/Columbia_test.py'
src = open(COL).read()

if '_lightasd_read_frame' in src:
    print('[crop] 이미 패치됨')
    raise SystemExit(0)

shutil.copy(COL, COL + '.bak_crop')

# ---- 1) 모듈 전역 헬퍼를 crop_video 정의 "앞에" 순수 삽입 (중복 def 방지) ----
helper = '''import os as _os_crop
from functools import lru_cache as _lru_crop
_LIGHTASD_FLIST = None

@_lru_crop(maxsize=int(_os_crop.environ.get('LIGHTASD_FRAME_CACHE', '400')))
def _lightasd_read_frame(path):
    # 같은 프레임을 여러 track 이 읽을 때 디스크 I/O 중복 제거 (결과 동일).
    return cv2.imread(path)


'''

anchor = 'def crop_video(args, track, cropFile):'
if anchor not in src:
    raise SystemExit('crop_video 정의 못 찾음')
# anchor 앞에 helper 삽입 (anchor 는 그대로 유지 → def 1개)
src = src.replace(anchor, helper + anchor, 1)

# ---- 2) crop_video 안: track 마다 glob → 전역 1회 ----
old_glob = "\tflist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames\n\tflist.sort()"
new_glob = ("\tglobal _LIGHTASD_FLIST\n"
            "\tif _LIGHTASD_FLIST is None:\n"
            "\t\t_LIGHTASD_FLIST = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))\n"
            "\tflist = _LIGHTASD_FLIST")
if old_glob not in src:
    raise SystemExit('crop_video 의 flist glob 패턴 못 찾음')
src = src.replace(old_glob, new_glob, 1)

# ---- 3) crop_video 안: imread → LRU 캐시 헬퍼 ----
old_read = '\t\timage = cv2.imread(flist[frame])'
new_read = '\t\timage = _lightasd_read_frame(flist[frame])'
if old_read not in src:
    raise SystemExit('crop_video 의 imread 패턴 못 찾음')
src = src.replace(old_read, new_read, 1)

open(COL, 'w').write(src)
print('[crop] crop_video 패치 완료 — flist 1회 glob + imread LRU 캐시')
print('done — 정확도 무영향. 검증: face_clustering 1회 → crop 단계 시간 측정 + score 동일 확인')
