import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from imm.extractors._helper import create_extractor
from imm.matchers._helper import create_matcher
from imm.misc import extend_keys_with_suffix
from imm.settings import img0_path as default_img0_path
from imm.settings import img1_path as default_img1_path
from imm.utils.dataset import FeaturesPairsDataset, ImagePairsDataset
from imm.utils.device import detect_device, to_cpu, to_cuda, to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.viz2d import MatchVisualizer
from imm.utils.warnings import suppress_warnings
from imm.utils.writers import AsycMatchesWriter, MatchesWriter


def path2key(name: str) -> str:
    """Converts a file path to a key."""
    return name.replace("/", "-")


def pairs2key(name0: str, name1: str) -> str:
    """Creates a key for a pair of items."""
    separator = "/"
    return separator.join((path2key(name0), path2key(name1)))


def load_and_process_image(
    image_path: str, max_img_size: Optional[int], device: torch.device
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and process an image."""
    logger.info(f"Loading image: {image_path}")
    data = load_image_tensor(image_path, resize=max_img_size)
    return data[0].to(device), data[1]


class Matching:
    def __init__(
        self,
        matcher_name: str = "superglue_outdoor",
        extractor_name: str = "superpoint",
        max_keypoints: int = -1,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initializes the Matching class and the matcher model. Sets up the extractor if needed.
        """
        self.device = device if device else detect_device()
        self.matcher = create_matcher(name=matcher_name, cfg={}, **kwargs)
        self.matcher.to(self.device)
        self.matcher.eval()
        logger.info(f"Initialized {matcher_name} matcher on {self.device}")

        self.extractor = None
        if "image0" not in self.matcher.required_inputs:
            self.set_extractor(extractor_name, max_keypoints)

    def set_extractor(self, extractor_name: str, max_keypoints: int) -> None:
        self.extractor = create_extractor(extractor_name, cfg={"max_keypoints": max_keypoints})
        self.extractor.eval().to(self.device)
        logger.info(f"Initialized {extractor_name} extractor on {self.device}")

    def extract_features(self, image: torch.Tensor, suffix: str) -> Dict[str, Any]:
        logger.info(f"Extracting features for image{suffix}")
        preds = self.extractor.extract({"image": image})
        preds = extend_keys_with_suffix(preds, suffix)
        h, w = image.shape[-2:]
        preds[f"size{suffix}"] = torch.tensor([w, h])
        return preds

    @torch.inference_mode()
    def match_features(self, data: Dict[str, Union[torch.Tensor, List, Tuple]]) -> Dict:
        """Matches a pair of descriptors or raw images."""
        data = to_cuda(data) if self.device == "cuda" else to_cpu(data)
        preds = self.matcher.match(data)
        return to_numpy(preds)

    def filter_matches(self, preds: Dict[str, np.ndarray], match_thd: float) -> Dict[str, np.ndarray]:
        if match_thd <= 0:
            return preds
        else:
            raise NotImplementedError("Filtering matches based on threshold is not implemented yet")
            # TODO: Implement filtering based on threshold

    @torch.inference_mode()
    def match_images(self, image0: torch.Tensor, image1: torch.Tensor, match_thd: float = 0.0) -> Dict[str, np.ndarray]:
        if "image0" in self.matcher.required_inputs:
            data = {"image0": image0, "image1": image1}
            preds = self.match_features(data)
            return self.filter_matches(preds, match_thd)
        else:
            features0 = self.extract_features(image0, "0")
            features1 = self.extract_features(image1, "1")
            logger.info(f"Matching img0: {len(features0['kpts0'][0])}, img1: {len(features1['kpts1'][0])}")
            data = {**features0, **features1}
            preds = self.match_features(data)
            return self.filter_matches(preds, match_thd)

    def compute_match_statistics(
        self, idx: int, matches: Dict[str, np.ndarray], name0: str = "", name1: str = ""
    ) -> Dict[str, Any]:
        """Computes statistics about the matches."""

        num_matches = len(matches.get("mkpts0", []))
        kpts0 = len(matches.get("kpts0", []))
        kpts1 = len(matches.get("kpts1", []))

        print(f"Iteration {idx + 1}:")
        print(f"    Image pair: {name0} - {name1}")
        print(f"    Number of matches: {num_matches}")
        print(f"    keypoints: {kpts0} - {kpts1}")

        return {
            "num_matches": num_matches,
            "num_kpts0": kpts0,
            "num_kpts1": kpts1,
        }

    @torch.inference_mode()
    def match_sequence_images(
        self,
        dataset: ImagePairsDataset,
        save_path: Path,
        batch_size: int = 1,
        num_workers: int = 4,
        print_freq: int = 100,
        match_thd: float = 0.0,
    ) -> None:
        """Processes a dataset of image pairs and saves matching results."""
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        writer = MatchesWriter(save_path)

        start_time = time.time()

        for idx, data in enumerate(tqdm(dataloader, desc="Matching images".rjust(15), colour="green")):
            name0, name1 = data["name0"][0], data["name1"][0]
            image0, image1 = data["image0"][0], data["image1"][0]

            # Match images directly
            preds = self.match_images(image0, image1, match_thd=match_thd)
            pair_key = pairs2key(name0, name1)

            # Write matches and stats to file
            writer.write_matches(pair_key, preds)

            # Compute statistics
            if (idx + 1) % print_freq == 0:
                stats = self.compute_match_statistics(idx, preds, name0, name1)

        writer.close()
        total_time = time.time() - start_time
        logger.info(f"Matches saved to {save_path}")
        logger.info(f"Total processing time for image pairs: {total_time:.2f} seconds")

    @torch.inference_mode()
    def match_sequence_features(
        self,
        dataset: FeaturesPairsDataset,
        save_path: Path,
        batch_size: int = 1,
        num_workers: int = 16,
        print_freq: int = 100,
        keys: Optional[List[str]] = None,
        match_thd: float = 0.0,
    ) -> None:
        """Processes a dataset of pre-extracted feature pairs and saves matching results."""
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        writer = AsycMatchesWriter(save_path, num_workers=num_workers)

        if match_thd > 0:
            raise NotImplementedError("Filtering matches based on threshold is not implemented yet")

        start_time = time.time()
        for idx, data in enumerate(tqdm(dataloader, desc="Matching features".rjust(15), colour="green")):
            name0, name1 = data["name0"][0], data["name1"][0]

            # Match pre-extracted features
            preds = self.match_features(data)
            pair_key = pairs2key(name0, name1)

            # Filter keys
            wpreds = {k: preds[k] for k in keys} if keys else preds

            # Write matches and stats to file
            writer.write_matches(pair_key, wpreds)

            # Compute statistics
            if (idx + 1) % print_freq == 0:
                stats = self.compute_match_statistics(idx, preds, name0, name1)

        writer.close()
        total_time = time.time() - start_time
        logger.info(f"Matches saved to {save_path}")
        logger.info(f"Total processing time for feature pairs: {total_time:.2f} seconds")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(matcher={self.matcher}, extractor={self.extractor}, device={self.device})"


@click.command()
@click.argument("img0_path", type=click.Path(exists=True), default=default_img0_path)
@click.argument("img1_path", type=click.Path(exists=True), default=default_img1_path)
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--max_keypoints", default=-1, help="Maximum number of keypoints", type=int)
@click.option("--max_img_size", default=None, type=int, help="Max image size")
@click.option("--match_thd", default=0.0, help="Matching score threshold")
@click.option("--visualize", is_flag=True, help="Enable visualization")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.option("--save_path", default=None, help="Output directory ")
@suppress_warnings()
def match_images(
    img0_path: str,
    img1_path: str,
    matcher: str,
    extractor: str,
    max_keypoints: int,
    max_img_size: int,
    match_thd: float,
    visualize: bool,
    force_cpu: bool,
    save_path: str,
) -> None:
    """Match a pair of images using a given matcher and extractor.

    Args:
        img0_path (str): Path to the first image.
        img1_path (str): Path to the second image.
        matcher (str): Name of the matcher.
        extractor (str): Name of the extractor.
        max_keypoints (int): Maximum number of keypoints.
        max_img_size (int): Maximum image size.
        match_thd (float): Matching score threshold.
        visualize (bool): Enable or disable visualization.
        force_cpu (bool): Force the use of CPU instead of GPU.
        save_path (str): Output directory.
    """

    logger.info("Starting image matching process")
    logger.info(
        f"Matcher: {matcher}, Extractor: {extractor}, Max keypoints: {max_keypoints}",
        f"Threshold: {match_thd}, Max size: {max_img_size}",
    )

    # Determine device
    device = detect_device(force_cpu)

    # Paths
    img0_path = Path(img0_path)
    img1_path = Path(img1_path)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, max_img_size, device)
    image1, image1_cv = load_and_process_image(img1_path, max_img_size, device)

    # Match images
    matcher = Matching(matcher_name=matcher, extractor_name=extractor, max_keypoints=max_keypoints, device=device)

    matches = matcher.match_images(image0, image1)

    # Visualize matches
    visualizer = MatchVisualizer()
    visualizer.draw_matches(image0_cv, image1_cv, **matches, title="Matches", show_image=visualize)

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.is_dir():
            output_file = save_path / f"matches_{img0_path.stem}_{img1_path.stem}.png"
        else:
            output_file = save_path

        visualizer.save(str(output_file))
        logger.info(f"Visualization saved to {output_file}")

    logger.success("Image matching done")


if __name__ == "__main__":
    match_images()
