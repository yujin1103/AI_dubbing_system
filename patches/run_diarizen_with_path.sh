#!/bin/bash
export LD_LIBRARY_PATH=/opt/venv_diarizen/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/venv_diarizen/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/venv_diarizen/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
/opt/venv_diarizen/bin/python /tmp/td.py "$@"
