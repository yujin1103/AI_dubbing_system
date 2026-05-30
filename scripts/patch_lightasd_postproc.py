# -*- coding: utf-8 -*-
"""Light-ASD S3FD.detect_faces_batch 후처리 벡터화 patch — 정확도 무영향.

병목 프로파일 (test6 3102프레임, 검출 139.6s):
  decode 8.8s(6%) + GPU forward 7.7s(6%) + prep 1.2s(1%) + **post 121.9s(87%)**.
  post 가 87% — 원인: detections 텐서를 픽셀마다 .cpu().numpy() 동기 복사 +
  np.vstack 루프 재할당. GPU↔CPU 작은 전송 반복이 극도로 느림.

수정 (bbox 결과 100% 동일):
  - detections[n] 을 프레임당 1회 .cpu() 로 이동 (픽셀마다 복사 제거)
  - conf_th 필터를 텐서 마스크로 (while 루프 제거)
  - np.vstack 반복 → 1회 stack
  conf_th/scale/nms 파라미터·임계값 동일 → 검출 bbox·NMS 결과 불변.

사용: docker exec dubbing_pipeline /usr/bin/python /workspace/scripts/patch_lightasd_postproc.py
백업: /opt/Light-ASD/model/faceDetector/s3fd/__init__.py.bak_postproc
"""
import shutil

S3FD = '/opt/Light-ASD/model/faceDetector/s3fd/__init__.py'
src = open(S3FD).read()

MARKER = '# POSTPROC_VECTORIZED'
if MARKER in src:
    print('[postproc] 이미 패치됨')
    raise SystemExit(0)

# 기존 detect_faces_batch 후처리 inner loop (정확히 매칭)
OLD = '''                for n in range(detections.size(0)):
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
                    results.append(bboxes[keep])'''

NEW = '''                # POSTPROC_VECTORIZED — detections 프레임당 1회 .cpu(), conf 마스크 추출 (bbox 동일)
                det_cpu = detections.cpu()
                for n in range(det_cpu.size(0)):
                    w, h = chunk[n].shape[1], chunk[n].shape[0]
                    scale_t = torch.Tensor([w, h, w, h])
                    dn = det_cpu[n]  # (num_classes, top_k, 5)
                    rows = []
                    for i in range(dn.size(1)):
                        conf = dn[i, :, 0]
                        mask = conf > conf_th
                        if not bool(mask.any()):
                            continue
                        sel = dn[i, mask]            # (k, 5): [score, x1,y1,x2,y2]
                        pts = sel[:, 1:] * scale_t   # (k, 4)
                        scores = sel[:, 0:1]         # (k, 1)
                        rows.append(torch.cat([pts, scores], dim=1).numpy())
                    if rows:
                        bboxes = np.vstack(rows)
                    else:
                        bboxes = np.empty(shape=(0, 5))
                    keep = nms_(bboxes, 0.1)
                    results.append(bboxes[keep])'''

if OLD not in src:
    raise SystemExit('detect_faces_batch 후처리 패턴 못 찾음 — 수동 확인 필요')

shutil.copy(S3FD, S3FD + '.bak_postproc')
src = src.replace(OLD, NEW, 1)
open(S3FD, 'w').write(src)
print('[postproc] detect_faces_batch 후처리 벡터화 완료')
print('done — 정확도 무영향. 검증: 검출시간 측정 + bbox 동일 확인')
