# -*- coding: utf-8 -*-
"""Light-ASD 검출 속도 patch — 병목(inference_video) 직격.

병목 진단 (test4 108s, facedetScale 0.25):
  영상변환 ~15s + 프레임추출 ~10s + **얼굴검출 ~5.5분(병목)** + crop ~1분 + ASD.
  검출이 느린 이유: 2732개 JPG를 1장씩 cv2.imread + S3FD를 1프레임씩 호출
  → GPU util 3~17%(미포화), disk I/O + Python 루프 오버헤드.

수정 (영상 무관, 모든 영상 가속 — 캐싱과 다름):
  1) S3FD.detect_faces_batch 추가 — N프레임 동시 GPU forward (전처리는 detect_faces verbatim).
  2) inference_video 가 JPG 대신 **video.avi 에서 프레임을 순차 디코드로 직접 읽어** 배치 검출.
     (pyframes JPG 추출은 crop 단계가 쓰므로 유지 — 추출은 ~10s 로 빠름. 검출의 JPG read 만 제거.)
  검출 결과(bbox)는 동일. conf_th/facedetScale 은 기존값 유지(검출 민감도 튜닝은 별도).

사용: docker exec dubbing_pipeline /usr/bin/python /workspace/scripts/patch_lightasd_speed.py
환경변수 LIGHTASD_DET_BATCH (기본 16) 로 배치 크기 조정.
백업: /opt/Light-ASD/*.bak_speed
"""
import shutil, re

S3FD = '/opt/Light-ASD/model/faceDetector/s3fd/__init__.py'
COL = '/opt/Light-ASD/Columbia_test.py'

# ---- 1) S3FD.detect_faces_batch ----
src = open(S3FD).read()
if 'def detect_faces_batch' not in src:
    shutil.copy(S3FD, S3FD + '.bak_speed')
    method = '''
    def detect_faces_batch(self, images, conf_th=0.9, scale=1.0, batch_size=16):
        # images: list of RGB HxWx3 (동일 크기). 단일 scale. per-image bbox 리스트 반환.
        results = []
        with torch.no_grad():
            for b0 in range(0, len(images), batch_size):
                chunk = images[b0:b0 + batch_size]
                tensors = []
                for image in chunk:
                    scaled_img = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    scaled_img = np.swapaxes(scaled_img, 1, 2)
                    scaled_img = np.swapaxes(scaled_img, 1, 0)
                    scaled_img = scaled_img[[2, 1, 0], :, :]
                    scaled_img = scaled_img.astype('float32')
                    scaled_img -= img_mean
                    scaled_img = scaled_img[[2, 1, 0], :, :]
                    tensors.append(torch.from_numpy(scaled_img))
                x = torch.stack(tensors, dim=0).to(self.device)
                detections = self.net(x).data
                for n in range(detections.size(0)):
                    w, h = chunk[n].shape[1], chunk[n].shape[0]
                    scale_t = torch.Tensor([w, h, w, h])
                    bboxes = np.empty(shape=(0, 5))
                    for i in range(detections.size(1)):
                        j = 0
                        while detections[n, i, j, 0] > conf_th:
                            score = detections[n, i, j, 0]
                            pt = (detections[n, i, j, 1:] * scale_t).cpu().numpy()
                            bboxes = np.vstack((bboxes, (pt[0], pt[1], pt[2], pt[3], score)))
                            j += 1
                    keep = nms_(bboxes, 0.1)
                    results.append(bboxes[keep])
        return results
'''
    idx = src.rindex('return bboxes')
    end = src.index('\n', idx) + 1
    src = src[:end] + method + src[end:]
    open(S3FD, 'w').write(src)
    print('[s3fd] detect_faces_batch 추가')
else:
    print('[s3fd] 이미 패치됨')

# ---- 2) inference_video: video.avi 순차 읽기 + 배치 ----
col = open(COL).read()
if 'detect_faces_batch' not in col:
    shutil.copy(COL, COL + '.bak_speed')
    new_func = '''def inference_video(args):
\t# GPU: video.avi 에서 프레임을 순차 디코드로 읽어 배치 검출 (JPG 2732개 open 제거 → 병목 해소).
\timport os
\tDET = S3FD(device='cuda')
\tcap = cv2.VideoCapture(args.videoFilePath)
\tdets = []
\tB = int(os.environ.get('LIGHTASD_DET_BATCH', '16'))
\tfidx = 0
\twhile True:
\t\tframes = []
\t\tfor _ in range(B):
\t\t\tok, fr = cap.read()
\t\t\tif not ok:
\t\t\t\tbreak
\t\t\tframes.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
\t\tif not frames:
\t\t\tbreak
\t\tbatch_bboxes = DET.detect_faces_batch(frames, conf_th=0.9, scale=args.facedetScale, batch_size=B)
\t\tfor bboxes in batch_bboxes:
\t\t\tdets.append([])
\t\t\tfor bbox in bboxes:
\t\t\t\tdets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
\t\t\tfidx += 1
\t\tsys.stderr.write('%s det batch @%d\\r' % (args.videoFilePath, fidx))
\tcap.release()
\tsavePath = os.path.join(args.pyworkPath,'faces.pckl')
\twith open(savePath, 'wb') as fil:
\t\tpickle.dump(dets, fil)
\treturn dets'''
    m = re.search(r'def inference_video\(args\):.*?(?=\ndef bb_intersection_over_union)', col, re.S)
    if not m:
        raise SystemExit('inference_video 패턴 못 찾음')
    col = col[:m.start()] + new_func + '\n\n' + col[m.end():]
    open(COL, 'w').write(col)
    print('[col] inference_video → video.avi 순차+배치')
else:
    print('[col] 이미 패치됨')
print('done — 검출 속도 patch 적용 (미실행). 검증: face_clustering 1회 실행해 검출시간 5.5분→? 측정')
