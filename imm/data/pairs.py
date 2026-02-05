from pathlib import Path

import h5py
import torch
from loguru import logger
from torch.utils.data import Dataset

from imm.utils.io import load_image_tensor


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

    def __init__(self, pairs: list, image_dir: Path):
        # pairs
        self.pairs = pairs

        # image_dir
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        name0, name1 = self.pairs[idx]
        image0_path = self.image_dir / name0
        image1_path = self.image_dir / name1

        # Load images
        image0_tensor, _ = load_image_tensor(image0_path)
        image1_tensor, _ = load_image_tensor(image1_path)

        return {"image0": image0_tensor, "image1": image1_tensor, "name0": name0, "name1": name1}

    def __repr__(self):
        return f"ImagePairsDataset(image_dir={self.image_dir}, num_pairs={len(self.pairs)})"


class FeaturesPairsDataset(Dataset):
    """Dataset for pairs of images features."""

    def __init__(self, pairs: list, features_path: Path):
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
