from pathlib import Path

import h5py
import torch
from loguru import logger
from torch.utils.data import Dataset

from imm.utils.io import find_images, load_image_tensor


def relative_path(path, root):
    return path.relative_to(root).as_posix()


class ImagesFromList(Dataset):
    """Dataset for loading images from a directory."""

    def __init__(self, root: str, max_img_size: int = -1):
        # root
        self.root = root

        # collect image paths
        self.images_paths = sorted(find_images(root))

        # image names
        self.names = [relative_path(img_path, root) for img_path in self.images_paths]

        #
        self.max_img_size = max_img_size

        logger.info("ImagesFromList:")
        logger.info(f"      Images: {len(self.images_paths)} in {root}")
        logger.info(f"      Max image size: {max_img_size}")

    def __len__(self):
        return len(self.images_paths)

    def get_names(self):
        return self.names

    def __getitem__(self, item):
        out = {}

        img_path = self.images_paths[item]
        img_name = self.names[item]

        # load image
        data = load_image_tensor(img_path, resize=self.max_img_size)
        image = data[0]
        image_cv = data[1]
        scale = data[3]
        original_size = data[4]

        # dict
        out["image"] = image
        out["name"] = img_name
        out["original_size"] = original_size
        out["scale"] = scale

        return out

    def __repr__(self):
        return (
            f"ImagesFromList(root={self.root}, num_images={len(self.images_paths)}, max_img_size={self.max_img_size})"
        )


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
