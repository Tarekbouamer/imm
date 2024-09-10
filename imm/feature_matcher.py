from functools import partial
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from imm.utils.device import to_cuda, to_numpy

# from imm.utils.io import H5Writer
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

from imm.extractors._helper import create_extractor
from imm.matchers._helper import create_matcher
from queue import Queue
from threading import Thread


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
            data = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
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


class ExtractorManager:
    """
    Manages the extraction of features from images using a specified model.
    """

    def __init__(
        self,
        name: str,
        cfg: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = "cuda",
        batch_size: int = 1,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ExtractorManager with the specified model and settings.
        """

        self.name = name

        self.extractor = create_extractor(name, cfg=cfg, **kwargs)
        self.extractor.eval()

        self.device = torch.device(device)
        self.extractor.to(self.device)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.writer = None

        logger.info(f"ExtractorManager model: {name}, batch_size={batch_size}, num_workers={num_workers}, device={self.device}")

    @torch.inference_mode()
    def extract_image(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract features from a single image, scaling keypoints if size is provided.
        """
        data = to_cuda(data)
        preds = self.extractor.extract(data)

        preds = {k: v[0] if isinstance(v, (List, Tuple)) else v for k, v in preds.items()}

        if "size" in data:
            size = data["size"][0]
            current_size = data["image"].shape[-2:][::-1]
            scales = torch.Tensor((size[0] / current_size[0], size[1] / current_size[1])).to(size).cuda()
            preds["kpts"] = (preds["kpts"] + 0.5) * scales[None] - 0.5
            preds["uncertainty"] = preds.pop("uncertainty", 1.0) * scales.mean()
            preds["size"] = size

        return to_numpy(preds)

    def extract_dataset(self, dataset: torch.utils.data.Dataset, save_path: Optional[str] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Extract features from a dataset and optionally save the results.
        """
        logger.info(f"Starting feature extraction for dataset with {len(dataset)} items")

        if save_path:
            self.writer = FeaturesWriter(save_path)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        logger.info(f"DataLoader created with {len(dataloader)} batches")

        results = []
        start_time = time.time()

        for _, batch in enumerate(tqdm(dataloader, colour="green", desc="extract locals".rjust(15))):
            if "name" not in batch:
                logger.error("Batch dictionary is missing 'name' key")
                raise ValueError("Batch dictionary must contain 'name' key")

            batch_preds = self.extract_image(batch)

            for name, pred in zip(batch["name"], batch_preds):
                results.append(pred)
                if self.writer:
                    self.writer.write_features(key=name, data=batch_preds)

        total_time = time.time() - start_time
        logger.info(f"Dataset extraction completed in {total_time:.2f} seconds")

        if self.writer:
            self.writer.close()

        return results

    def __repr__(self) -> str:
        """
        Return a string representation of the ExtractorManager.
        """
        return f"ExtractorManager(model={self.extractor.__class__.__name__}, device={self.device}, batch_size={self.batch_size}, num_workers={self.num_workers})"


def path2key(name: str) -> str:
    """key id for item from its path

    Args:
        name str: path to item

    Returns:
        str: key
    """
    return name.replace("/", "-")


def pairs2key(
    name0: str,
    name1: str,
) -> str:
    """key id of pairs items

    Args:
        name0 str: path to item0
        name1 str: path to item1

    Returns:
        str: key
    """
    separator = "/"
    return separator.join((path2key(name0), path2key(name1)))


class WorkQueue:
    def __init__(self, work_fn, num_workers=1):
        self.queue = Queue(num_workers)
        self.threads = [Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_workers)]

        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)

        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        while True:
            item = self.queue.get()
            if item is None:
                break
            work_fn(item)

    def put(self, data):
        self.queue.put(data)


class MatcherManager:
    """Matcher Manager class that integrates functionalities for matching descriptors, sequences, and query-database pairs."""

    def __init__(self, name: str, cfg: Dict, device: Optional[str] = None, **kwargs) -> None:
        """
        Initializes the MatcherManager with configuration parameters.

        Args:
            name (str): Name of the matcher model.
            cfg (Dict): Configuration parameters.
            device (Optional[str]): Device to run the model on ('cuda', 'cpu', etc.). Defaults to 'cuda' if available, otherwise 'cpu'.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Create the matcher model
        self.matcher = create_matcher(name=name, cfg=cfg, **kwargs)
        self.matcher.to(self.device)
        self.matcher.eval()

        logger.info(f"Initialized {self.name} matcher on {self.device}")

    @torch.inference_mode()
    def match_pair(self, data: Dict[str, Union[torch.Tensor, List, Tuple]]) -> Dict:
        """
        Matches a pair of image descriptors.

        Args:
            data (Dict[str, Union[torch.Tensor, List, Tuple]]): Input data containing descriptors for a pair of images.

        Returns:
            Dict: Predictions for the pair, including matches and scores.
        """

        data = to_cuda(data)
        preds = self.matcher(data)
        return preds

    @torch.inference_mode()
    def match_sequence(
        self,
        sequence: Any,
        save_path: Optional[Path] = None,
        num_workers: int = 4,
    ) -> Optional[Path]:
        """
        Matches a sequence of image descriptors and optionally saves the results.

        Args:
            sequence (Any): The sequence data.
            save_path (Optional[Path]): The path to save the matches. If None, results are not saved.
            num_workers (int): Number of worker threads to use. Defaults to 4.

        Returns:
            Optional[Path]: Path where matches are saved, or None if not saved.
        """
        seq_dl = (
            sequence
            if isinstance(sequence, DataLoader)
            else DataLoader(
                sequence,
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
            )
        )

        writer_queue = None
        if save_path is not None:
            writer = MatchesWriter(save_path)
            writer_queue = WorkQueue(partial(writer.write_matches), num_workers)

        for it, (src_name, dst_name, data) in enumerate(tqdm(seq_dl, total=len(seq_dl))):
            preds = self.match_pair(data)
            pair_key = pairs2key(src_name[0], dst_name[0])

            if writer_queue:
                writer_queue.put((pair_key, preds))

        if writer_queue:
            writer_queue.join()
            writer.close()
            logger.info(f"Matches saved to {save_path}")
            return save_path

        return None

    # @torch.inference_mode()
    # def match_query_database(
    #     self, q_preds: Dict, pairs: List[Tuple[str, str]]
    # ) -> Dict:
    #     """
    #     Matches query descriptors with database descriptors.

    #     Args:
    #         q_preds (Dict): Query descriptors.
    #         pairs (List[Tuple[str, str]]): List of (query, database) pairs.

    #     Returns:
    #         Dict: Dictionary of matched pairs and their predictions.
    #     """
    #     local_features_path = (
    #         Path(self.cfg["visloc_path"]) / "db_local_features.h5"
    #     )
    #     local_features_loader = LocalFeaturesReader(
    #         save_path=local_features_path
    #     )
    #     q_preds = wrap_keys_with_extenstion(q_preds, ext="0")

    #     pairs_matches = {}

    #     for src_name, dst_name in pairs:
    #         db_preds = local_features_loader.load(dst_name)
    #         db_preds = wrap_keys_with_extenstion(db_preds, ext="1")
    #         preds = self.match_pair({**q_preds, **db_preds})
    #         pair_key = pairs2key(src_name, dst_name)
    #         pairs_matches[pair_key] = preds

    #     return pairs_matches

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (" f"name: {self.name} " f"model_name: {self.model_name} " f"device: {self.device})"
