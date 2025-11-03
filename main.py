import os
import argparse
import shutil
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import torch
import torchaudio

# ---------------------------------------
# 0. Device
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------
# 1. Audio feature extraction (GPU batch)
# ---------------------------------------

def build_mel_transform(sample_rate=22050, n_mels=64):
    """
    Build a MelSpectrogram transform on the GPU.
    We’ll take mean/std over time of log-mel to get a feature vector.
    """
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        power=2.0,
    ).to(device)
    return transform


def extract_features_torch_batch(
    paths,
    mel_transform,
    target_sr=22050,
    max_duration=2.0,
):
    """
    Extract features for a batch of files:

    - Load audio (CPU, torchaudio)
    - Convert to mono, resample to target_sr
    - Trim to max_duration
    - Pad batch to same length, move to GPU
    - Compute mel spectrogram on GPU
    - Return mean/std over time of log-mel (on CPU as numpy)

    Returns:
      valid_paths: list[str]
      feats: np.ndarray, shape (B, 2 * n_mels)
    """
    wave_list = []
    valid_paths = []

    max_len_samples = int(target_sr * max_duration)

    for p in paths:
        try:
            wav, fs = torchaudio.load(p)  # (C, T), CPU
            # mono
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # resample if needed
            if fs != target_sr:
                wav = torchaudio.functional.resample(wav, fs, target_sr)
            # trim
            if wav.size(-1) > max_len_samples:
                wav = wav[..., :max_len_samples]
            wave_list.append(wav)
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] Could not process {p}: {e}")

    if not wave_list:
        return [], None

    # pad to same length
    lengths = [w.size(-1) for w in wave_list]
    max_len_in_batch = max(lengths)
    batch = torch.zeros(len(wave_list), 1, max_len_in_batch)

    for i, w in enumerate(wave_list):
        batch[i, 0, : w.size(-1)] = w

    # to GPU
    batch = batch.to(device)

    with torch.no_grad():
        mel = mel_transform(batch)  # (B, n_mels, T)
        mel = torch.clamp(mel, min=1e-9)
        log_mel = torch.log(mel)

        # mean / std over time dimension
        mel_mean = log_mel.mean(dim=-1)  # (B, n_mels)
        mel_std = log_mel.std(dim=-1)    # (B, n_mels)
        feats = torch.cat([mel_mean, mel_std], dim=-1)  # (B, 2 * n_mels)

    feats = feats.cpu().numpy()
    return valid_paths, feats


# ---------------------------------------
# 2. File scanning helpers
# ---------------------------------------

def is_audio_file(fname, exts):
    return os.path.splitext(fname)[1].lower() in exts


def collect_files(root, exts):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_audio_file(fname, exts):
                paths.append(os.path.join(dirpath, fname))
    return paths


# ---------------------------------------
# 3. Smart cluster naming
# ---------------------------------------

TOKEN_MAP = {
    "kick": "kick",
    "kik": "kick",
    "bd": "kick",
    "808": "808",
    "snare": "snare",
    "snr": "snare",
    "rim": "rim",
    "clap": "clap",
    "hat": "hat",
    "hihat": "hat",
    "hh": "hat",
    "ohat": "hat",
    "chh": "hat",
    "perc": "perc",
    "shaker": "perc",
    "tom": "tom",
    "ride": "cymbal",
    "crash": "cymbal",
    "cymbal": "cymbal",
    "fx": "fx",
    "impact": "fx",
    "sweep": "fx",
    "vocal": "vox",
    "vox": "vox",
    "chant": "vox",
}

STOPWORDS = {
    "drum", "kit", "one", "shot", "oneshot", "sample", "audio",
    "loop", "wav", "mp3", "sfx", "sound", "snd",
    "trap", "hiphop", "hip", "hop", "vol", "pack",
    "v1", "v2", "v3",
}


def tokenize_filename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    name = name.lower()
    for ch in "-_.()[]{}":
        name = name.replace(ch, " ")
    tokens = [t for t in name.split() if len(t) >= 2 and not t.isdigit()]
    return tokens


def infer_cluster_name(file_paths, max_tokens_for_name=2):
    raw_counter = Counter()
    mapped_counter = Counter()

    for path in file_paths:
        tokens = tokenize_filename(path)
        for t in tokens:
            if t in STOPWORDS:
                continue
            raw_counter[t] += 1
            canonical = TOKEN_MAP.get(t)
            if canonical:
                mapped_counter[canonical] += 1

    if mapped_counter:
        top = [tok for tok, _ in mapped_counter.most_common(max_tokens_for_name)]
        return "-".join(top)

    if raw_counter:
        top = [tok for tok, _ in raw_counter.most_common(max_tokens_for_name)]
        return "-".join(top)

    return "misc"

# ---------------------------------------
# 4. Main clustering + organizing
# ---------------------------------------

def cluster_and_organize(
    source_root,
    dest_root,
    n_clusters=10,
    move=False,
    exts=(".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"),
    batch_size=128,
    target_sr=22050,
    n_mels=64,
    max_duration=2.0,
):

    source_root = os.path.abspath(source_root)
    dest_root = os.path.abspath(dest_root)

    print(f"Using device: {device}")
    print(f"Scanning audio files under: {source_root}")
    files = collect_files(source_root, exts)
    print(f"Found {len(files)} audio files.")

    if not files:
        print("No audio files found. Exiting.")
        return

    mel_transform = build_mel_transform(sample_rate=target_sr, n_mels=n_mels)

    # ---- Batch feature extraction with GPU ----
    print("Extracting features on GPU (batched)...")
    feature_list = []
    valid_files = []

    total = len(files)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = files[start:end]

        v_paths, feats = extract_features_torch_batch(
            batch_paths,
            mel_transform=mel_transform,
            target_sr=target_sr,
            max_duration=max_duration,
        )

        if feats is not None and len(v_paths) > 0:
            # feats is currently (B, 1, 128) → flatten to (B, 128)
            feats = feats.reshape(feats.shape[0], -1)
            valid_files.extend(v_paths)
            feature_list.append(feats)

        print(f"Processed {end}/{total} files...")

    if not feature_list:
        print("No valid features extracted. Exiting.")
        return

    # Filter out any empty batches (safety)
    feature_list = [f for f in feature_list if f is not None and f.size > 0]

    # Stack into (N_total_files, n_features)
    X = np.vstack(feature_list)
    print(f"Feature matrix shape: {X.shape}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    print(f"Clustering into {n_clusters} clusters with MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=256,
        n_init="auto",
        max_iter=200,
    )
    labels = kmeans.fit_predict(X_scaled)

    # Group files by cluster
    cluster_to_files = defaultdict(list)
    for path, label in zip(valid_files, labels):
        cluster_to_files[label].append(path)

    # Infer names for each cluster
    print("Inferring cluster names...")
    cluster_names = {}
    for cluster_id, paths in cluster_to_files.items():
        base_name = infer_cluster_name(paths)
        cluster_names[cluster_id] = base_name

    # Make folders and move/copy files
    os.makedirs(dest_root, exist_ok=True)
    action = "Moving" if move else "Copying"
    print(f"{action} files into: {dest_root}")

    def ensure_unique_path(dest_path: str) -> str:
        if not os.path.exists(dest_path):
            return dest_path
        root, ext = os.path.splitext(dest_path)
        i = 1
        while True:
            candidate = f"{root}_{i}{ext}"
            if not os.path.exists(candidate):
                return candidate
            i += 1

    file_count = 0
    for cluster_id, paths in cluster_to_files.items():
        base_name = cluster_names[cluster_id]
        folder_name = f"{cluster_id:02d}_{base_name}"
        cluster_folder = os.path.join(dest_root, folder_name)
        os.makedirs(cluster_folder, exist_ok=True)

        for src_path in paths:
            dst_path = os.path.join(cluster_folder, os.path.basename(src_path))
            dst_path = ensure_unique_path(dst_path)

            try:
                if move:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"[WARN] Failed to {action.lower()} {src_path}: {e}")
                continue

            file_count += 1
            if file_count % 50 == 0:
                print(f"{action} {file_count} files...")

    print(f"Done. Processed {file_count} files into {len(cluster_to_files)} clusters.")
    print("Cluster folders created:")
    for cid in sorted(cluster_names.keys()):
        print(f"  {cid:02d}: {cluster_names[cid]}")

# ---------------------------------------
# 5. CLI
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cluster drum one-shots with GPU-accelerated log-mel features and auto-name folders."
    )
    parser.add_argument("source", help="Root folder containing all your drum kits")
    parser.add_argument("dest", help="Destination folder for the clustered kit")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters for KMeans (default: 10)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them",
    )
    parser.add_argument(
        "--extensions",
        default=".wav,.mp3,.flac,.aif,.aiff,.ogg",
        help="Comma-separated list of audio extensions",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for GPU feature extraction (default: 128)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=22050,
        help="Target sample rate for resampling (default: 22050)",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=64,
        help="Number of Mel bands (default: 64)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=2.0,
        help="Max audio duration in seconds (default: 2.0)",
    )

    args = parser.parse_args()
    exts = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())

    cluster_and_organize(
        source_root=args.source,
        dest_root=args.dest,
        n_clusters=args.n_clusters,
        move=args.move,
        exts=exts,
        batch_size=args.batch_size,
        target_sr=args.target_sr,
        n_mels=args.n_mels,
        max_duration=args.max_duration,
    )


if __name__ == "__main__":
    main()
