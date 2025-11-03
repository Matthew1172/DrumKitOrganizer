import os
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from model import DrumCNNv2

# ----------------------
# Dataset
# ----------------------
class DrumDataset(Dataset):
    def __init__(
        self,
        csv_path,
        sample_rate=16000,
        duration=1.0,
        n_mels=64,
        label_to_idx=None,
    ):
        """
        csv_path: path to CSV with columns: "File Path", "Tag"
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

        self.filepaths = []
        self.labels = []

        # Read CSV
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["File Path"]
                tag = row["Tag"]
                if not os.path.isfile(path):
                    # Skip missing files
                    continue
                self.filepaths.append(path)
                self.labels.append(tag)

        # Build label mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(set(self.labels))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Audio transforms (log-mel)
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
        )
        self.amplitude_to_db = AmplitudeToDB()

        # Optional resampler (in case file SR != target)
        self.resampler = None

    def __len__(self):
        return len(self.filepaths)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)  # (channels, samples)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, samples)

        # Resample if needed
        if sr != self.sample_rate:
            if self.resampler is None:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate,
                )
            waveform = self.resampler(waveform)

        # Pad/trim to fixed length
        if waveform.size(1) < self.n_samples:
            pad_amount = self.n_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, : self.n_samples]

        return waveform

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label_str = self.labels[idx]
        label_idx = self.label_to_idx[label_str]

        waveform = self._load_audio(path)  # (1, n_samples)

        # Create log-mel spectrogram
        mel = self.mel_spectrogram(waveform)  # (1, n_mels, time)
        mel_db = self.amplitude_to_db(mel)    # (1, n_mels, time)

        # Normalize (simple mean/std)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        return mel_db, label_idx

# ----------------------
# Training & Evaluation
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Train simple drum classifier from WAV files.")
    parser.add_argument("--csv", required=True, help="Path to CSV: File Path,Tag")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds of audio per sample")
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--model_out", default="drum_cnn.pt")
    parser.add_argument("--val_split", type=float, default=0.2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = DrumDataset(
        csv_path=args.csv,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
    )
    num_classes = len(dataset.label_to_idx)
    print(f"Found {len(dataset)} samples, {num_classes} classes: {dataset.label_to_idx}")

    if len(dataset) == 0:
        raise RuntimeError("No valid audio files found. Check paths in the CSV.")

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # keep 0 for Windows simplicity
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Build model
    model = DrumCNNv2(n_mels=args.n_mels, n_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )
        
        # in training loop, after val_loss/val_acc:
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # Save model and label mapping
    save_obj = {
        "model_state_dict": model.state_dict(),
        "label_to_idx": dataset.label_to_idx,
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "n_mels": args.n_mels,
        # no flatten_dim needed for DrumCNNv2
    }
    torch.save(save_obj, args.model_out)

    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
