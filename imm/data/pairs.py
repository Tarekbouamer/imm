from pathlib import Path
from typing import List, Tuple

import h5py
import torch
from loguru import logger
from torch.utils.data import Dataset

from imm.utils.io import load_image_tensor


def parse_pairs_file(pairs_file: Path) -> List[Tuple[str, str]]:
    """Parse pairs file containing image pairs"""
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs


def read_key_from_h5py(name, _path):
    """Reads a specific key from an HDF5 file."""
    data = {}
    with h5py.File(str(_path), "r", libver="latest") as f:
        if name in f:
            g = f[name]
        else:
            logger.error(f"{name} not found in {_path}")
            return data

        for k, v in g.items():
            data[k] = torch.from_numpy(v.__array__()).float()

    return data


class ImagePairsDataset(Dataset):
    """Dataset for loading pairs of images."""

    def __init__(self, image_dir: Path, pairs: list, resize: int = None):
        # pairs
        self.pairs = pairs

        # image_dir
        self.image_dir = image_dir

        # resize
        self.resize = resize

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        name0, name1 = self.pairs[idx]
        image0_path = self.image_dir / name0
        image1_path = self.image_dir / name1

        # Load images
        image0_tensor = load_image_tensor(image0_path, resize=self.resize)[0]
        image1_tensor = load_image_tensor(image1_path, resize=self.resize)[0]

        return {"image0": image0_tensor, "image1": image1_tensor, "name0": name0, "name1": name1}

    def __repr__(self):
        return f"ImagePairsDataset(image_dir={self.image_dir}, num_pairs={len(self.pairs)})"


class FeaturesPairsDataset(Dataset):
    """Dataset for pairs of images features."""

    def __init__(self, features_path: Path, pairs: list):
        # pairs
        self.pairs = pairs

        # features_path
        self.features_path = features_path

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        name0, name1 = self.pairs[idx]

        # Read features from HDF5 file
        data0 = read_key_from_h5py(name0, self.features_path)
        data1 = read_key_from_h5py(name1, self.features_path)

        # Extend keys with suffixes
        data0 = {f"{k}0": v for k, v in data0.items()}
        data1 = {f"{k}1": v for k, v in data1.items()}

        return {**data0, **data1, "name0": name0, "name1": name1}

    def __repr__(self):
        return f"FeaturesPairsDataset(features_path={self.features_path}, num_pairs={len(self.pairs)})"
