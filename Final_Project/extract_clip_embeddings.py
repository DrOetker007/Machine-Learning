"""Create resumable OpenAI CLIP embeddings for cached listing main images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


class ClipImageDataset(Dataset):
    def __init__(self, paths: list[str], preprocess) -> None:
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, path


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as cached:
        paths = cached["local_paths"].astype(str)
        vectors = cached["embeddings"].astype(np.float32)
    return dict(zip(paths, vectors))


def save_cache(path: Path, values: dict[str, np.ndarray]) -> None:
    paths = np.array(sorted(values), dtype=str)
    vectors = np.vstack([values[item] for item in paths]).astype(np.float16)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, local_paths=paths, embeddings=vectors)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/image_manifest.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/clip_main_image_embeddings.npz")
    )
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    device = torch.device("cuda:0")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    manifest = pd.read_csv(args.manifest, usecols=["local_path"], low_memory=False)
    local_paths = sorted(
        path
        for path in manifest["local_path"].dropna().astype(str).unique()
        if Path(path).exists()
    )
    cache = load_cache(args.output)
    missing = [path for path in local_paths if path not in cache]
    print(f"Available unique images: {len(local_paths):,}")
    print(f"Cached embeddings: {len(local_paths) - len(missing):,}")
    print(f"New embeddings: {len(missing):,}")
    if not missing:
        return

    model, preprocess = clip.load(args.model, device=device, jit=False)
    model.eval()
    loader = DataLoader(
        ClipImageDataset(missing, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    started = time.perf_counter()
    new_paths: list[str] = []
    new_vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_number, (images, paths) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model.encode_image(images)
                features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
            new_vectors.append(features.float().cpu().numpy())
            new_paths.extend(paths)
            if batch_number % 100 == 0 or batch_number == len(loader):
                elapsed = time.perf_counter() - started
                completed = len(new_paths)
                print(
                    f"{completed:,}/{len(missing):,} images "
                    f"({completed / elapsed:.1f}/s)"
                )

    cache.update(dict(zip(new_paths, np.vstack(new_vectors).astype(np.float32))))
    save_cache(args.output, cache)
    elapsed = time.perf_counter() - started
    print(f"Saved {len(cache):,} embeddings to {args.output}")
    print(f"Embedding time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
