"""
MOS 추론 모듈
=============
학습된 MOS 모델을 사용하여 더빙 음성의 품질을 1~5점으로 평가.
orchestrator.py에서 import하여 사용.

사용법:
  from mos_evaluator import MOSEvaluator

  evaluator = MOSEvaluator("/workspace/media/model_cache/mos_model/best.pt")
  score = evaluator.evaluate("dubbed_segment.wav")
  print(f"MOS: {score:.2f}")

  # 배치 평가
  scores = evaluator.evaluate_batch(["seg1.wav", "seg2.wav", "seg3.wav"])
"""
import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np


class MOSPredictor(nn.Module):
    """wav2vec2-base + MLP head로 MOS 예측."""
    
    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    
    def forward(self, waveform, attention_mask=None):
        outputs = self.backbone(waveform, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.head(pooled)


class MOSEvaluator:
    """
    MOS 평가기 — 오디오 파일의 자연스러움을 1~5점으로 평가.
    
    Args:
        checkpoint_path: 학습된 모델 체크포인트 경로
        device: "cuda" 또는 "cpu"
        threshold: 이 점수 미만이면 "재합성 필요"로 판단
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        threshold: float = 3.5
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        
        # 모델 로드
        self.model = MOSPredictor().to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"[MOS] 모델 로드 완료 (SRCC={checkpoint.get('dev_srcc', 'N/A')})")
    
    def _load_audio(self, audio_path: str, max_duration: float = 10.0) -> torch.Tensor:
        """오디오 파일을 16kHz mono로 로드."""
        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)
        
        max_samples = int(max_duration * 16000)
        if waveform.size(0) > max_samples:
            waveform = waveform[:max_samples]
        
        return waveform
    
    @torch.no_grad()
    def evaluate(self, audio_path: str) -> float:
        """
        오디오 파일 1개의 MOS 점수를 반환.
        
        Args:
            audio_path: wav 파일 경로
            
        Returns:
            MOS 점수 (1.0 ~ 5.0)
        """
        waveform = self._load_audio(audio_path)
        waveform = waveform.unsqueeze(0).to(self.device)  # (1, seq_len)
        
        score = self.model(waveform).item()
        
        # 1~5 범위로 클리핑
        score = max(1.0, min(5.0, score))
        return round(score, 2)
    
    @torch.no_grad()
    def evaluate_batch(self, audio_paths: list) -> list:
        """
        여러 오디오 파일의 MOS 점수를 반환.
        
        Args:
            audio_paths: wav 파일 경로 리스트
            
        Returns:
            MOS 점수 리스트
        """
        scores = []
        for path in audio_paths:
            try:
                score = self.evaluate(path)
                scores.append(score)
            except Exception as e:
                print(f"[MOS] 평가 실패 ({path}): {e}")
                scores.append(0.0)
        return scores
    
    def should_retry(self, score: float) -> bool:
        """점수가 기준 미만이면 재합성 필요."""
        return score < self.threshold
    
    def diagnose(self, audio_path: str, original_audio: str = None) -> dict:
        """
        품질 진단 — 어떤 부분이 문제인지 분석.
        
        Args:
            audio_path: 더빙 오디오 경로
            original_audio: 원본 오디오 경로 (선택)
            
        Returns:
            {"score": 3.2, "pass": False, "suggestion": "속도를 줄여보세요"}
        """
        score = self.evaluate(audio_path)
        
        result = {
            "score": score,
            "pass": score >= self.threshold,
            "suggestion": "",
        }
        
        if score >= 4.0:
            result["suggestion"] = "품질 우수"
        elif score >= 3.5:
            result["suggestion"] = "통과, 약간의 개선 여지"
        elif score >= 3.0:
            result["suggestion"] = "속도 조절 또는 다른 레퍼런스 시도"
        elif score >= 2.0:
            result["suggestion"] = "레퍼런스 교체 및 텍스트 재번역 권장"
        else:
            result["suggestion"] = "세그먼트 재구성 필요"
        
        return result
    
    def unload(self):
        """GPU 메모리 해제."""
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        print("[MOS] 모델 해제 완료")


# ─── CLI 테스트 ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MOS 평가")
    parser.add_argument("audio", nargs="+", help="평가할 오디오 파일")
    parser.add_argument("--checkpoint", required=True, help="모델 체크포인트 경로")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args()
    
    evaluator = MOSEvaluator(args.checkpoint, threshold=args.threshold)
    
    for path in args.audio:
        result = evaluator.diagnose(path)
        status = "✅" if result["pass"] else "❌"
        print(f"{status} {os.path.basename(path)}: {result['score']:.2f} — {result['suggestion']}")
