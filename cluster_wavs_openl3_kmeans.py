import os
import csv
import argparse
import shutil

import numpy as np
from sklearn.cluster import KMeans

import soundfile as sf
import torchopenl3


def load_audio(path):
    """
    Load audio using soundfile; returns mono float32 array and sr.
    """
    audio, sr = sf.read(path, always_2d=False)

    # If stereo, downmix to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Ensure float32
    audio = audio.astype(np.float32)

    return audio, sr


def get_openl3_embedding_torch(
    audio,
    sr,
    model,
    center=True,
    hop_size=0.1,
):
    """
    Compute TorchOpenL3 embeddings and average over time
    to get one vector per file.

    Returns:
        emb_mean: (D,) numpy array (mean-pooled over time)
    """
    emb, ts = torchopenl3.get_audio_embedding(
        audio,
        sr,
        model=model,
        center=center,
        hop_size=hop_size,
    )
    # emb can be:
    #   (T, D)         - frames, dim
    #   (1, T, D)      - batch, frames, dim

    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()

    emb = np.asarray(emb)

    if emb.ndim == 3:
        # (B, T, D) -> (T, D) assuming B == 1
        if emb.shape[0] != 1:
            raise ValueError(f"Unexpected batch size in embedding: {emb.shape}")
        emb = emb[0]  # (T, D)

    if emb.ndim == 2:
        # (T, D) -> (D,)
        emb_mean = emb.mean(axis=0)
    elif emb.ndim == 1:
        # Already (D,)
        emb_mean = emb
    else:
        raise ValueError(f"Unexpected embedding shape {emb.shape} for sr={sr}")

    # Ensure 1D numpy array
    emb_mean = emb_mean.reshape(-1)

    return emb_mean


def main():
    parser = argparse.ArgumentParser(
        description="Cluster WAV files using k-means on TorchOpenL3 audio embeddings."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to CSV with columns: File Path,Tag")
    parser.add_argument("--n_clusters", type=int, default=8,
                        help="Number of k-means clusters (default: 8)")
    parser.add_argument("--out_dir", default="outputs/kmeans_openl3",
                        help="Output directory for clusters and CSV")
    parser.add_argument("--copy_files", action="store_true",
                        help="If set, copy WAVs into cluster subfolders")
    parser.add_argument("--input_repr", default="mel256",
                        choices=["mel256", "mel128", "linear"],
                        help="OpenL3 input representation")
    parser.add_argument("--content_type", default="music",
                        choices=["music", "env"],
                        help="OpenL3 content type (music or environmental)")
    parser.add_argument("--embedding_size", type=int, default=512,
                        choices=[512, 6144],
                        help="OpenL3 embedding size")
    parser.add_argument("--hop_size", type=float, default=0.1,
                        help="Hop size in seconds between embeddings (default: 0.1)")
    parser.add_argument("--center", action="store_true",
                        help="Center windows (default: False if not set)")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load TorchOpenL3 model once and reuse it
    print(
        f"Loading TorchOpenL3 model: input_repr={args.input_repr}, "
        f"content_type={args.content_type}, embedding_size={args.embedding_size}"
    )
    model = torchopenl3.models.load_audio_embedding_model(
        input_repr=args.input_repr,
        content_type=args.content_type,
        embedding_size=args.embedding_size,
    )

    filepaths = []
    tags = []

    # Read CSV
    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["File Path"]
            tag = row.get("Tag", "")
            if not os.path.isfile(path):
                print(f"[SKIP] Missing file: {path}")
                continue
            filepaths.append(path)
            tags.append(tag)

    if not filepaths:
        raise RuntimeError("No valid files found in CSV.")

    print(f"Found {len(filepaths)} valid WAV files. Extracting TorchOpenL3 embeddings...")

    features = []
    valid_paths = []
    valid_tags = []

    for i, (path, tag) in enumerate(zip(filepaths, tags)):
        try:
            audio, sr = load_audio(path)
            emb = get_openl3_embedding_torch(
                audio,
                sr,
                model=model,
                center=args.center,
                hop_size=args.hop_size,
            )
            # emb should now be (D,) with fixed D (e.g., 512)
            emb = np.asarray(emb).reshape(-1)

            features.append(emb)
            valid_paths.append(path)
            valid_tags.append(tag)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} files...")

        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            continue


    if not features:
        raise RuntimeError("No embeddings extracted successfully.")

    X = np.stack(features, axis=0)  # (N, D)
    print(f"Embedding matrix shape: {X.shape}")

    # K-means
    print(f"Running k-means with k={args.n_clusters}...")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=42,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(X)

    # Save assignments CSV
    assignments_csv = os.path.join(out_dir, "cluster_assignments_openl3.csv")
    with open(assignments_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["File Path", "Tag", "Cluster"])
        for path, tag, cid in zip(valid_paths, valid_tags, cluster_ids):
            writer.writerow([path, tag, int(cid)])

    print(f"Cluster assignments saved to: {assignments_csv}")

    # Optionally copy files into cluster folders
    if args.copy_files:
        print("Copying files into cluster subfolders...")
        for path, cid in zip(valid_paths, cluster_ids):
            cluster_dir = os.path.join(out_dir, f"cluster_{int(cid)}")
            os.makedirs(cluster_dir, exist_ok=True)
            dest_path = os.path.join(cluster_dir, os.path.basename(path))
            shutil.copy2(path, dest_path)
        print(f"Files copied under: {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
