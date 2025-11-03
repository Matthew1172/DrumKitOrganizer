import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from model import DrumCNNv2

# ----------------------
# Audio preprocessing (same as training)
# ----------------------
def build_transforms(sample_rate, n_mels, n_fft=1024, hop_length=256):
    mel_spectrogram = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    amplitude_to_db = AmplitudeToDB()
    return mel_spectrogram, amplitude_to_db


def preprocess_wav(
    path,
    sample_rate,
    duration,
    n_mels,
    mel_spectrogram,
    amplitude_to_db,
):
    # Load
    waveform, sr = torchaudio.load(path)  # (channels, samples)

    # Mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Pad / trim
    n_samples = int(sample_rate * duration)
    if waveform.size(1) < n_samples:
        pad_amount = n_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        waveform = waveform[:, :n_samples]

    # Log-mel
    mel = mel_spectrogram(waveform)       # (1, n_mels, time)
    mel_db = amplitude_to_db(mel)         # (1, n_mels, time)

    # Normalize per sample
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return mel_db  # (1, n_mels, time)


# ----------------------
# Prediction
# ----------------------
def predict_label(
    model,
    path,
    device,
    sample_rate,
    duration,
    n_mels,
    mel_spectrogram,
    amplitude_to_db,
    idx_to_label,
):
    mel_db = preprocess_wav(
        path,
        sample_rate,
        duration,
        n_mels,
        mel_spectrogram,
        amplitude_to_db,
    )
    # Add batch dimension
    x = mel_db.unsqueeze(0).to(device)  # (1, 1, n_mels, time)

    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()

    return idx_to_label[pred_idx]


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Classify WAV files in a directory and copy into folders by predicted tag."
    )
    parser.add_argument("--model", required=True, help="Path to trained model .pt (e.g. drum_cnn.pt)")
    parser.add_argument("--input_dir", required=True, help="Directory of .wav files (nested allowed)")
    parser.add_argument(
        "--example_name",
        default="example1",
        help="Name of example folder under outputs/ (default: example1)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    label_to_idx = checkpoint["label_to_idx"]
    sample_rate = checkpoint["sample_rate"]
    duration = checkpoint["duration"]
    n_mels = checkpoint["n_mels"]

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    model = DrumCNNv2(n_mels=n_mels, n_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Build audio transforms (must match training settings)
    n_fft = 1024
    hop_length = 256
    mel_spectrogram, amplitude_to_db = build_transforms(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Build audio transforms
    mel_spectrogram, amplitude_to_db = build_transforms(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Output root: outputs/example1/...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(script_dir, "outputs", args.example_name)
    os.makedirs(output_root, exist_ok=True)

    # Walk input directory and classify
    num_files = 0
    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            src_path = os.path.join(root, fname)
            try:
                pred_label = predict_label(
                    model,
                    src_path,
                    device,
                    sample_rate,
                    duration,
                    n_mels,
                    mel_spectrogram,
                    amplitude_to_db,
                    idx_to_label,
                )
            except Exception as e:
                print(f"Skipping {src_path} due to error: {e}")
                continue

            # Destination folder: outputs/example1/<pred_label>/
            dest_dir = os.path.join(output_root, pred_label)
            os.makedirs(dest_dir, exist_ok=True)

            dest_path = os.path.join(dest_dir, os.path.basename(src_path))

            # Copy file
            shutil.copy2(src_path, dest_path)
            num_files += 1
            print(f"{src_path} -> {dest_dir}")

    print(f"Done. Processed {num_files} .wav files.")
    print(f"Output organized under: {output_root}")


if __name__ == "__main__":
    main()
