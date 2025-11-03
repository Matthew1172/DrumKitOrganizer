import os
import sys
import argparse
import shutil
from collections import Counter, defaultdict

import numpy as np

# You don't strictly need these now, but leaving them in case you reuse
# the GPU feature pipeline later.
from sklearn.cluster import MiniBatchKMeans  # unused in current flow
from sklearn.preprocessing import StandardScaler  # unused in current flow

import torch
import torchaudio

# ---------------------------------------
# 0. Device
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------
# DrumClassifier imports (your CNN-LSTM)
# ---------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(THIS_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

try:
    from drumclassifier_utils import DrumClassifier
    from drumclassifier_constants import INSTRUMENT_NAMES
except ImportError as e:
    print("ERROR: Could not import DrumClassifier modules.")
    print("Make sure 'code/' with drumclassifier_utils.py and "
          "drumclassifier_constants.py is on PYTHONPATH.")
    print(f"Details: {e}")
    sys.exit(1)


# ---------------------------------------
# 1. (Optional) Audio feature extraction (GPU batch)
#    Not used in the new flow, but kept for future KMeans-within-class.
# ---------------------------------------

def build_mel_transform(sample_rate=22050, n_mels=64):
    """
    Build a MelSpectrogram transform on the GPU.
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
    Extract log-mel mean/std features for a batch of files (GPU).
    Not used in the main flow now.
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


def collect_dirs(root):
    dirs = []
    for dirpath, _, _ in os.walk(root):
        dirs.append(dirpath)
    return dirs


# ---------------------------------------
# 3. Smart cluster naming (still here if you ever
#    want to run KMeans within each instrument class)
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
# 4. Main organizing (DrumClassifier-based)
# ---------------------------------------

def cluster_and_organize(
    source_root,
    dest_root,
    move=False,
    exts=(".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"),
    model_path="C:\\Users\\pecko\\MatthewCode\\DrumClassifer-CNN-LSTM\\models\\mel_cnn_models\\mel_cnn_model_high_v2.model",
    min_conf=0.5,
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

    # Load DrumClassifier
    print("Loading DrumClassifier model...")
    drumcl = DrumClassifier(path_to_model=model_path)
    print("Model loaded.")

    # Classify all files by walking directories and calling
    # predict_proba_directory on each (so nested folders are covered).
    print("Classifying files with DrumClassifier...")
    dirs = collect_dirs(source_root)
    results = {}  # path -> {pred_class, max_prob}

    for idx, d in enumerate(dirs, 1):
        try:
            prob_dict = drumcl.predict_proba_directory(d, format="prob")
        except Exception as e:
            print(f"[WARN] Error classifying directory {d}: {e}")
            continue

        if not prob_dict:
            continue

        for path, probs in prob_dict.items():
            if not is_audio_file(path, exts):
                continue

            probs_arr = np.asarray(probs, dtype=float)
            if probs_arr.size != len(INSTRUMENT_NAMES):
                print(f"[WARN] Unexpected prob length for {path}: {probs_arr.size}")
                continue

            max_idx = int(np.argmax(probs_arr))
            max_prob = float(probs_arr[max_idx])
            label = INSTRUMENT_NAMES[max_idx]

            if max_prob < min_conf:
                pred_class = "unknown"
            else:
                pred_class = label

            abs_path = os.path.abspath(path)
            results[abs_path] = {
                "pred_class": pred_class,
                "max_prob": max_prob,
            }

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(dirs)} directories...")

    if not results:
        print("No classification results. Exiting.")
        return

    # Group by predicted class
    class_to_files = defaultdict(list)
    for path, info in results.items():
        class_to_files[info["pred_class"]].append(path)

    print("Class distribution:")
    for cls in sorted(class_to_files.keys()):
        print(f"  {cls}: {len(class_to_files[cls])} files")

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
    for cls, paths in class_to_files.items():
        folder_name = cls  # e.g. "kick", "snr", "hho", "808", "fx", "unknown"
        class_folder = os.path.join(dest_root, folder_name)
        os.makedirs(class_folder, exist_ok=True)

        for src_path in paths:
            dst_path = os.path.join(class_folder, os.path.basename(src_path))
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

    print(f"Done. Processed {file_count} files into {len(class_to_files)} classes.")
    print("Folders created:")
    for cls in sorted(class_to_files.keys()):
        print(f"  {cls}: {len(class_to_files[cls])} files")


# ---------------------------------------
# 5. CLI
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Organize drum one-shots using DrumClassifier CNN-LSTM classes."
    )
    parser.add_argument("source", help="Root folder containing all your drum kits")
    parser.add_argument("dest", help="Destination folder for the organized kit")
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
        "--model-path",
        default="C:\\Users\\pecko\\MatthewCode\\DrumClassifer-CNN-LSTM\\models\\mel_cnn_models\\mel_cnn_model_high_v2.model",
        help="Path to DrumClassifier .model file",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.5,
        help="Minimum confidence to trust a class; otherwise goes to 'unknown' (default: 0.5)",
    )

    args = parser.parse_args()
    exts = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())

    cluster_and_organize(
        source_root=args.source,
        dest_root=args.dest,
        move=args.move,
        exts=exts,
        model_path=args.model_path,
        min_conf=args.min_conf,
    )


if __name__ == "__main__":
    main()
