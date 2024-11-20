from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, Union

import h5py
import numpy as np
import torch
from loguru import logger


class H5Writer:
    """H5Writer is a class for writing data to an HDF5 file."""

    def __init__(self, filename: str, mode: str = "a", compression: str = None, chunks: bool = True):
        self.filename = filename
        self.mode = mode
        self.compression = compression
        self.chunks = chunks
        self.hfile = h5py.File(self.filename, self.mode, libver="latest")

    def close(self):
        """Closes the HDF5 file."""
        self.hfile.close()

    def write(self, data: Dict[str, Union[torch.Tensor, np.ndarray]]):
        """Writes a dictionary of tensors or numpy arrays to the HDF5 file."""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            self.hfile.create_dataset(key, data=value, compression=self.compression, chunks=self.chunks)


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
                grp.create_dataset(k, data=v, compression=self.compression, chunks=self.chunks)

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
                group.create_dataset(key, data=value, compression=self.compression, chunks=self.chunks)

        except OSError as error:
            logger.error(f"Error writing matches for group {group_name}: {error}")
            raise


class AsycMatchesWriter:
    """AsycMatchesWriter is a standalone class for writing matches to an HDF5 file using threads."""

    def __init__(self, filename, mode="a", compression=None, chunks=True, num_workers=4):
        self.filename = filename
        self.mode = mode
        self.compression = compression
        self.chunks = chunks
        self.queue = Queue()
        self.threads = [Thread(target=self._process_tasks, daemon=True) for _ in range(num_workers)]
        for thread in self.threads:
            thread.start()

        logger.info(f"AsycMatchesWriter initialized with {num_workers} threads for {filename}")

    def _process_tasks(self):
        """Worker thread to handle tasks from the queue."""
        while True:
            task = self.queue.get()
            if task is None:  # Sentinel value to stop
                break
            try:
                self._process_task(task)
            except Exception as e:
                logger.error(f"Error processing task {task}: {e}")
            finally:
                self.queue.task_done()

    def _process_task(self, task):
        """Process a single task: Write a group to the HDF5 file."""
        group_name, matches = task

        with h5py.File(self.filename, self.mode, libver="latest") as hfile:
            if group_name in hfile:
                del hfile[group_name]  # Remove existing group
            group = hfile.create_group(group_name)
            for key, value in matches.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                group.create_dataset(key, data=value, compression=self.compression, chunks=self.chunks)

    def write_matches(self, group_name, matches):
        """Add a task to the processing queue."""
        self.queue.put((group_name, matches))

    def close(self):
        """Wait for all tasks to complete and stop worker threads."""
        for _ in self.threads:
            self.queue.put(None)  # Sentinel value for each thread
        for thread in self.threads:
            thread.join()  # Wait for all threads to finish
