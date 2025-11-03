import os
import argparse
import shutil
from collections import Counter, defaultdict

import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Audio feature extraction
# -----------------------------

def extract_features(file_path, sr=22050, n_mfcc=20):
    """
    Load an audio file and extract a feature vector:
    - MFCCs (mean + std)
    - Spectral centroid (mean + std)
    - Spectral rolloff (mean + std)
    - Zero-crossing rate (mean + std)
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        if y.size == 0:
            raise ValueError("Empty audio")

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        # Spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = spec_centroid.mean()
        sc_std = spec_centroid.std()

        # Spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = spec_rolloff.mean()
        sr_std = spec_rolloff.std()

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = zcr.mean()
        zcr_std = zcr.std()

        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [sc_mean, sc_std],
            [sr_mean, sr_std],
            [zcr_mean, zcr_std],
        ])

        return features

    except Exception as e:
        print(f"[WARN] Could not process {file_path}: {e}")
        return None

# -----------------------------
# 2. File scanning helpers
# -----------------------------

def is_audio_file(fname, exts):
    return os.path.splitext(fname)[1].lower() in exts

def collect_files(root, exts):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_audio_file(fname, exts):
                paths.append(os.path.join(dirpath, fname))
    return paths

# -----------------------------
# 3. Smart cluster naming
# -----------------------------

# Optional mapping to canonical drum-ish names
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
    """
    Take all filenames in a cluster, find the most common semantic tokens,
    and map them to human-friendly drum-ish names.
    """
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

    # Prefer mapped drum-style tokens
    if mapped_counter:
        top = [tok for tok, _ in mapped_counter.most_common(max_tokens_for_name)]
        name = "-".join(top)
        return name

    # Fallback: use raw common tokens
    if raw_counter:
        top = [tok for tok, _ in raw_counter.most_common(max_tokens_for_name)]
        name = "-".join(top)
        return name

    return "misc"

# -----------------------------
# 4. Main clustering + organizing
# -----------------------------

def cluster_and_organize(
    source_root,
    dest_root,
    n_clusters=10,
    move=False,
    exts=(".wav", ".mp3"),
):

    source_root = os.path.abspath(source_root)
    dest_root = os.path.abspath(dest_root)

    print(f"Scanning audio files under: {source_root}")
    files = collect_files(source_root, exts)
    print(f"Found {len(files)} audio files.")

    if not files:
        print("No audio files found. Exiting.")
        return

    # Extract features
    print("Extracting features...")
    feature_list = []
    valid_files = []
    for idx, path in enumerate(files, 1):
        feat = extract_features(path)
        if feat is not None:
            feature_list.append(feat)
            valid_files.append(path)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(files)} files for features...")

    if not feature_list:
        print("No valid features extracted. Exiting.")
        return

    X = np.vstack(feature_list)
    print(f"Feature matrix shape: {X.shape}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    print(f"Clustering into {n_clusters} clusters with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
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

    file_count = 0
    for cluster_id, paths in cluster_to_files.items():
        base_name = cluster_names[cluster_id]
        folder_name = f"{cluster_id:02d}_{base_name}"
        cluster_folder = os.path.join(dest_root, folder_name)
        os.makedirs(cluster_folder, exist_ok=True)

        for src_path in paths:
            dst_path = os.path.join(cluster_folder, os.path.basename(src_path))

            # Ensure unique path
            if os.path.exists(dst_path):
                root, ext = os.path.splitext(dst_path)
                i = 1
                while True:
                    candidate = f"{root}_{i}{ext}"
                    if not os.path.exists(candidate):
                        dst_path = candidate
                        break
                    i += 1

            if move:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

            file_count += 1
            if file_count % 50 == 0:
                print(f"{action} {file_count} files...")

    print(f"Done. Processed {file_count} files into {len(cluster_to_files)} clusters.")
    print("Cluster folders created:")
    for cid in sorted(cluster_names.keys()):
        print(f"  {cid:02d}: {cluster_names[cid]}")

# -----------------------------
# 5. CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cluster drum one-shots with KMeans and auto-name folders."
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
        default=".wav,.mp3",
        help="Comma-separated list of audio extensions (default: .wav,.mp3)",
    )

    args = parser.parse_args()
    exts = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())

    cluster_and_organize(
        source_root=args.source,
        dest_root=args.dest,
        n_clusters=args.n_clusters,
        move=args.move,
        exts=exts,
    )

if __name__ == "__main__":
    main()
