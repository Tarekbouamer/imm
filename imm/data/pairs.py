import threading
from pathlib import Path
from typing import Optional

import h5py
import torch
from torch.utils.data import Dataset

from imm.utils.io import load_image_tensor


class ImagePairsDataset(Dataset):
    """Dataset for loading pairs of images."""

    def __init__(self, image_dir: str | Path, pairs: list, resize: Optional[int] = None):
        # pairs
        self.pairs = pairs

        # image_dir
        self.image_dir = Path(image_dir).expanduser().resolve()

        # resize
        self.resize = resize

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        name0, name1 = self.pairs[idx]
        image0_path = self.image_dir / name0
        image1_path = self.image_dir / name1

        # Load images - unpack explicitly to avoid silent breakage
        image0_tensor, *_ = load_image_tensor(image0_path, resize=self.resize)
        image1_tensor, *_ = load_image_tensor(image1_path, resize=self.resize)

        return {
            "image0": image0_tensor,
            "image1": image1_tensor,
            "name0": name0,
            "name1": name1,
            "path0": str(image0_path),
            "path1": str(image1_path),
            "pair_index": idx,
        }

    def __repr__(self):
        return f"ImagePairsDataset(image_dir={self.image_dir}, num_pairs={len(self.pairs)})"


class FeaturesPairsDataset(Dataset):
    """Dataset for pairs of images features.
    """

    def __init__(self, features_path: str | Path, pairs: list):
        # pairs
        self.pairs = pairs

        # features_path
        self.features_path = Path(features_path).expanduser().resolve()

        # threading
        self._local = threading.local()

    def _get_h5(self) -> h5py.File:
        """Return a per-thread HDF5 file handle."""
        h5 = getattr(self._local, "h5", None)
        if h5 is None:
            h5 = h5py.File(str(self.features_path), "r", libver="latest")
            self._local.h5 = h5
        return h5

    def _read_features(self, name: str) -> dict:
        """Read features for a single image."""
        h5 = self._get_h5()
        if name not in h5:
            raise KeyError(f"'{name}' not found in {self.features_path}")

        g = h5[name]
        out = {}
        for k, v in g.items():
            out[k] = torch.from_numpy(v[()])
        return out

    def close(self):
        """Close the current HDF5 file handle."""
        h5 = getattr(self._local, "h5", None)
        if h5 is not None:
            h5.close()
            self._local.h5 = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        name0, name1 = self.pairs[idx]

        # Read features from HDF5 file
        data0 = self._read_features(name0)
        data1 = self._read_features(name1)

        # Extend keys with suffixes
        data0 = {f"{k}0": v for k, v in data0.items()}
        data1 = {f"{k}1": v for k, v in data1.items()}

        return {**data0, **data1, "name0": name0, "name1": name1, "pair_index": idx}

    def __repr__(self):
        return f"FeaturesPairsDataset(features_path={self.features_path}, num_pairs={len(self.pairs)})"
