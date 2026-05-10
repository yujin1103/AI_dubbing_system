"""
MOS 예측 모델 학습 스크립트
============================
wav2vec2-base를 백본으로 사용하여 TTS 음성의 자연스러움(MOS)을 1~5점으로 예측.
BVCC 데이터셋으로 학습.

사용법:
  python train_mos.py --datadir /workspace/media/datasets/bvcc/phase1-main/DATA
  python train_mos.py --datadir /workspace/media/datasets/bvcc/phase1-main/DATA --epochs 20 --lr 1e-5
  python train_mos.py --datadir /workspace/media/datasets/combined --epochs 15

모델 저장: /workspace/media/model_cache/mos_model/best.pt
"""
import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from scipy.stats import pearsonr, spearmanr


# ─── 모델 정의 ────────────────────────────────────────────────

class MOSPredictor(nn.Module):
    """
    wav2vec2-base → mean pooling → MLP → MOS 점수 (1.0~5.0)
    
    파라미터:
      - wav2vec2-base: 95M (frozen 또는 fine-tune)
      - MLP head: ~400K
      - 총 VRAM: ~2GB (batch_size=8 기준)
    """
    def __init__(self, freeze_backbone=False):
        super().__init__()
        
        # wav2vec2-base from HuggingFace
        from transformers import Wav2Vec2Model
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        hidden_size = self.backbone.config.hidden_size  # 768
        
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
        """
        Args:
            waveform: (batch, seq_len) — 16kHz raw audio
            attention_mask: (batch, seq_len) — padding mask
        Returns:
            mos: (batch, 1) — predicted MOS score
        """
        outputs = self.backbone(waveform, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, time, 768)
        
        # mean pooling (mask-aware)
        if attention_mask is not None:
            # attention_mask를 hidden states 시간 축에 맞게 조정
            # wav2vec2는 입력을 ~320x 다운샘플하므로 마스크도 조정
            mask_len = hidden.size(1)
            mask = attention_mask[:, :mask_len * 320:320]  # 대략적 다운샘플
            if mask.size(1) < mask_len:
                mask = torch.nn.functional.pad(mask, (0, mask_len - mask.size(1)), value=0)
            elif mask.size(1) > mask_len:
                mask = mask[:, :mask_len]
            mask = mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)  # (batch, 768)
        
        mos = self.head(pooled)  # (batch, 1)
        return mos


# ─── 데이터셋 ─────────────────────────────────────────────────

class BVCCDataset(Dataset):
    """
    BVCC 데이터셋 로더.
    
    디렉토리 구조:
      DATA/
        wav/          — 오디오 파일 (.wav, 16kHz)
        sets/
          TRAINSET    — train 파일 목록
          DEVSET      — dev 파일 목록
          TESTSET     — test 파일 목록 (없을 수 있음)
        mydata_system.csv  — 시스템별 MOS
    
    각 줄 형식: filename,mos_score
    """
    def __init__(self, datadir: str, split: str = "train", max_duration: float = 10.0):
        self.datadir = Path(datadir)
        self.wav_dir = self.datadir / "wav"
        self.max_samples = int(max_duration * 16000)  # 16kHz 기준
        
        # 레이블 파일 로드
        self.items = []
        
        # BVCC split별 레이블 파일 매핑
        label_map = {
            "train": "train_mos_list.txt",
            "dev": "val_mos_list.txt",
            "test": "test_mos_list.txt",
        }
        label_file = self.datadir / "sets" / label_map.get(split, "train_mos_list.txt")
        
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        fname = parts[0].strip()
                        try:
                            mos = float(parts[1].strip())
                        except ValueError:
                            continue
                        
                        wav_path = self.wav_dir / fname
                        if wav_path.exists():
                            self.items.append((str(wav_path), mos))
        
        if not self.items:
            print(f"[WARN] 레이블 파일을 찾을 수 없습니다: {label_file}")
            print(f"[WARN] 데이터 디렉토리 구조를 확인하세요: {self.datadir}")
        
        print(f"[Dataset] {split}: {len(self.items)}개 샘플 로드")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        wav_path, mos = self.items[idx]
        
        waveform, sr = torchaudio.load(wav_path)
        
        # 모노로 변환
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 16kHz로 리샘플
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)  # (seq_len,)
        
        # 최대 길이 제한
        if waveform.size(0) > self.max_samples:
            waveform = waveform[:self.max_samples]
        
        return waveform, torch.tensor(mos, dtype=torch.float32)


def collate_fn(batch):
    """가변 길이 오디오를 패딩하여 배치 구성."""
    waveforms, scores = zip(*batch)
    
    max_len = max(w.size(0) for w in waveforms)
    
    padded = torch.zeros(len(waveforms), max_len)
    masks = torch.zeros(len(waveforms), max_len, dtype=torch.long)
    
    for i, w in enumerate(waveforms):
        padded[i, :w.size(0)] = w
        masks[i, :w.size(0)] = 1
    
    scores = torch.stack(scores)
    return padded, masks, scores


# ─── 학습 루프 ─────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count = 0
    
    for batch_idx, (waveforms, masks, scores) in enumerate(loader):
        waveforms = waveforms.to(device)
        masks = masks.to(device)
        scores = scores.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        predictions = model(waveforms, attention_mask=masks)
        loss = criterion(predictions, scores)
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * waveforms.size(0)
        count += waveforms.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx+1}: loss={loss.item():.4f}")
    
    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    count = 0
    criterion = nn.MSELoss()
    
    for waveforms, masks, scores in loader:
        waveforms = waveforms.to(device)
        masks = masks.to(device)
        scores = scores.to(device).unsqueeze(1)
        
        predictions = model(waveforms, attention_mask=masks)
        loss = criterion(predictions, scores)
        
        total_loss += loss.item() * waveforms.size(0)
        count += waveforms.size(0)
        
        all_preds.extend(predictions.squeeze().cpu().numpy().tolist())
        all_labels.extend(scores.squeeze().cpu().numpy().tolist())
    
    # Pearson / Spearman 상관계수
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    lcc, _ = pearsonr(all_preds, all_labels)
    srcc, _ = spearmanr(all_preds, all_labels)
    mse = total_loss / count
    
    return mse, lcc, srcc


# ─── 메인 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MOS 예측 모델 학습")
    parser.add_argument("--datadir", required=True, help="BVCC DATA 디렉토리")
    parser.add_argument("--outdir", default="/workspace/media/model_cache/mos_model",
                        help="모델 저장 경로")
    parser.add_argument("--epochs", type=int, default=15, help="학습 에포크")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-5, help="학습률")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="wav2vec2 백본 동결 (빠른 학습)")
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="오디오 최대 길이 (초)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device: {device}")
    
    # 출력 디렉토리
    os.makedirs(args.outdir, exist_ok=True)
    
    # 데이터셋
    print("[Dataset] 로딩 중...")
    train_dataset = BVCCDataset(args.datadir, split="train", max_duration=args.max_duration)
    dev_dataset = BVCCDataset(args.datadir, split="dev", max_duration=args.max_duration)
    
    if len(train_dataset) == 0:
        print("[ERROR] 학습 데이터가 없습니다. 데이터 경로를 확인하세요.")
        print(f"  datadir: {args.datadir}")
        print(f"  wav 폴더: {Path(args.datadir) / 'wav'}")
        print(f"  sets 폴더: {Path(args.datadir) / 'sets'}")
        
        # 디렉토리 구조 출력
        datadir = Path(args.datadir)
        if datadir.exists():
            print(f"\n  디렉토리 내용:")
            for item in sorted(datadir.iterdir()):
                print(f"    {item.name}")
        sys.exit(1)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    
    # 모델
    print("[Model] MOSPredictor 초기화...")
    model = MOSPredictor(freeze_backbone=args.freeze_backbone).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 총 파라미터: {total_params:,}")
    print(f"[Model] 학습 파라미터: {trainable_params:,}")
    
    # 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 학습
    best_srcc = -1
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"[Train] 시작: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_mse, dev_lcc, dev_srcc = evaluate(model, dev_loader, device)
        scheduler.step()
        
        print(f"  train_loss: {train_loss:.4f}")
        print(f"  dev_mse: {dev_mse:.4f}, LCC: {dev_lcc:.4f}, SRCC: {dev_srcc:.4f}")
        
        # Best 모델 저장
        if dev_srcc > best_srcc:
            best_srcc = dev_srcc
            best_epoch = epoch
            save_path = os.path.join(args.outdir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_mse": dev_mse,
                "dev_lcc": dev_lcc,
                "dev_srcc": dev_srcc,
            }, save_path)
            print(f"  ★ Best 모델 저장 (SRCC={dev_srcc:.4f})")
        
        print()
    
    print(f"{'='*60}")
    print(f"[Train] 완료!")
    print(f"[Train] Best epoch: {best_epoch}, SRCC: {best_srcc:.4f}")
    print(f"[Train] 모델 저장: {os.path.join(args.outdir, 'best.pt')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
