try:
    from audio_separator.separator import Separator
    print("audio_separator OK")
except Exception as e:
    print(f"FAIL: {e}")

try:
    from model.faceDetector.s3fd import S3FD
    print("S3FD OK")
except Exception as e:
    print(f"S3FD FAIL: {e}")

try:
    import sys
    sys.path.insert(0, "/opt/Light-ASD")
    from ASD import ASD
    print("ASD OK")
except Exception as e:
    print(f"ASD FAIL: {e}")
