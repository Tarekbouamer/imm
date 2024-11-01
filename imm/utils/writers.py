from pathlib import Path
from typing import Dict, Union

import h5py
import numpy as np
import torch

from loguru import logger


class H5Writer:
    """H5Writer is a class for writing data to an HDF5 file."""

    def __init__(self, filename: str, mode: str = "w", compression: str = None):
        self.filename = filename
        self.mode = mode
        self.compression = compression
        self.hfile = h5py.File(self.filename, self.mode)

    def close(self):
        """Closes the HDF5 file."""
        self.hfile.close()

    def write(self, data: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """Writes a dictionary of tensors or numpy arrays to the HDF5 file."""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            self.hfile.create_dataset(key, data=value, compression=self.compression)


class FeaturesWriter(H5Writer):
    """FeaturesWriter is a class for writing feature data to an HDF5 file."""

    def __init__(self, save_path: Path) -> None:
        super().__init__(str(save_path))
        logger.info(f"FeaturesWriter initialized at {save_path}")

    def write_features(self, key: str, data: Dict[str, Union[torch.Tensor, np.ndarray]]) -> None:
        """Writes multiple datasets to an HDF5 group."""
        try:
            if key in self.hfile:
                del self.hfile[key]
            grp = self.hfile.create_group(key)
            for k, v in data.items():
                grp.create_dataset(k, data=v)

        except OSError as error:
            logger.error(f"Error writing features for key {key}: {error}")
            raise


class MatchesWriter(H5Writer):
    """MatchesWriter is a class for writing matches to an HDF5 file."""

    def __init__(self, filename: str, mode: str = "w", compression: str = None):
        super().__init__(filename, mode, compression)
        logger.info(f"MatchesWriter initialized at {filename}")

    def write_matches(
        self,
        group_name: str,
        matches: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> None:
        """Writes matches data to an HDF5 group."""
        try:
            if group_name in self.hfile:
                del self.hfile[group_name]
            group = self.hfile.create_group(group_name)
            for key, value in matches.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                group.create_dataset(key, data=value, compression=self.compression)

        except OSError as error:
            logger.error(f"Error writing matches for group {group_name}: {error}")
            raise
