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
from imm.utils.logger import setup_logger
from imm.utils.viz2d import MatchVisualizer
from imm.utils.warnings import suppress_warnings
from imm.utils.writers import MatchesWriter


def path2key(name: str) -> str:
    """Converts a file path to a key."""
    return name.replace("/", "-")


def pairs2key(name0: str, name1: str) -> str:
    """Creates a key for a pair of items."""
    separator = "/"
    return separator.join((path2key(name0), path2key(name1)))


def load_and_process_image(image_path: str, max_size: Optional[int], device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and process an image."""
    logger.info(f"Loading image: {image_path}")
    data = load_image_tensor(image_path, resize=max_size)
    return data[0].to(device), data[1]


# TODO: add match pair for matching two images
# TODO: add match sequence for matching a sequence of pairs directly on image level
# TODO: add match features for matching a sequence of pairs on feature level


class Matching:
    def __init__(
        self,
        matcher_name: str,
        cfg: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        extractor_name: str = "superpoint",
        **kwargs: Any,
    ):
        """
        Initializes the Matching class and the matcher model. Sets up the extractor if needed.
        """
        self.device = device if device else detect_device()
        self.matcher = create_matcher(name=matcher_name, cfg=cfg, **kwargs)
        self.matcher.to(self.device)
        self.matcher.eval()
        logger.info(f"Initialized {matcher_name} matcher on {self.device}")

        self.extractor = None
        if "image0" not in self.matcher.required_inputs:
            self.set_extractor(extractor_name)

    def set_extractor(self, extractor_name: str):
        self.extractor = create_extractor(extractor_name)
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

    def match_images(self, image0: torch.Tensor, image1: torch.Tensor) -> Dict[str, Any]:
        if "image0" in self.matcher.required_inputs:
            match_data = {"image0": image0, "image1": image1}
            return self.match_features(match_data)
        else:
            features0 = self.extract_features(image0, "0")
            features1 = self.extract_features(image1, "1")
            logger.info(f"Matching img0: {len(features0['kpts0'][0])}, img1: {len(features1['kpts1'][0])}")
            match_data = {**features0, **features1}
            return self.match_features(match_data)

    def compute_match_statistics(self, matches: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Computes statistics about the matches."""
        num_matches = len(matches.get("mkpts0", []))
        if num_matches == 0:
            logger.warning("No matches found")
            return {
                "num_matches": num_matches,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
            }

        avg_score = np.mean(matches["mscores"])
        max_score = np.max(matches["mscores"])
        min_score = np.min(matches["mscores"])

        print(f"Number of matches: {num_matches}")
        print(f"Average score: {avg_score:.3f}")
        print(f"Max score: {max_score:.3f}")
        print(f"Min score: {min_score:.3f}")

        return {
            "num_matches": num_matches,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
        }

    @torch.inference_mode()
    def match_sequence_images(
        self, dataset: ImagePairsDataset, save_path: Path, batch_size: int = 1, num_workers: int = 4, print_freq: int = 10
    ) -> None:
        """Processes a dataset of image pairs and saves matching results."""
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        writer = MatchesWriter(save_path)

        start_time = time.time()

        for idx, data in enumerate(tqdm(dataloader, desc="Matching images".rjust(15), colour="green")):
            name0, name1 = data["name0"][0], data["name1"][0]
            image0, image1 = data["image0"][0], data["image1"][0]

            # Match images directly
            preds = self.match_images(image0, image1)
            pair_key = pairs2key(name0, name1)

            # Write matches and stats to file
            writer.write_matches(pair_key, preds)

            # Compute statistics
            if (idx + 1) % print_freq == 0:
                stats = self.compute_match_statistics(preds)

        writer.close()
        total_time = time.time() - start_time
        logger.info(f"Matches saved to {save_path}")
        logger.info(f"Total processing time for image pairs: {total_time:.2f} seconds")

    @torch.inference_mode()
    def match_sequence_features(
        self, dataset: FeaturesPairsDataset, save_path: Path, batch_size: int = 1, num_workers: int = 4, print_freq: int = 10
    ) -> None:
        """Processes a dataset of pre-extracted feature pairs and saves matching results."""
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        writer = MatchesWriter(save_path)

        start_time = time.time()

        for idx, data in enumerate(tqdm(dataloader, desc="Matching features".rjust(15), colour="green")):
            name0, name1 = data["name0"][0], data["name1"][0]

            # Match pre-extracted features
            preds = self.match_features(data)
            pair_key = pairs2key(name0, name1)

            # Write matches and stats to file
            writer.write_matches(pair_key, preds)

            # Compute statistics
            if (idx + 1) % print_freq == 0:
                stats = self.compute_match_statistics(preds)

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
@click.option("--max_size", default=None, type=int, help="Max image size")
@click.option("--output_dir", default="output", help="Output directory for logs and visualization")
@click.option("--threshold", default=0.0, help="Matching score threshold")
@click.option("--visualize/--no-visualize", default=True, help="Enable or disable visualization")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.help_option("--help", "-h")
@suppress_warnings()
def match_images(
    img0_path: str,
    img1_path: str,
    matcher: str,
    extractor: str,
    max_size: Optional[int],
    output_dir: str,
    threshold: float,
    visualize: bool,
    force_cpu: bool,
) -> None:
    """Match features between two images."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logger(app_name="imm")

    logger.info("Starting image matching process")
    logger.info(f"Matcher: {matcher}, Extractor: {extractor}, Max size: {max_size}, Threshold: {threshold}")

    # Determine device
    device = detect_device(force_cpu)

    # Paths
    img0_path = Path(img0_path)
    img1_path = Path(img1_path)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, max_size, device)
    image1, image1_cv = load_and_process_image(img1_path, max_size, device)

    # Match images
    matcher_model = Matching(matcher_name=matcher, device=device, extractor_name=extractor)

    matches = matcher_model.match_images(image0, image1)

    # Statistics
    stats = matcher_model.compute_match_statistics(matches)
    logger.info(f"Match statistics: {stats}")

    # Visualize matches
    if visualize:
        logger.info("Visualizing matches")
        visualizer = MatchVisualizer()
        visualizer.visualize_matches(
            image0_cv,
            image1_cv,
            kpts0=matches["kpts0"],
            kpts1=matches["kpts1"],
            mkpts0=matches["mkpts0"],
            mkpts1=matches["mkpts1"],
            scores=matches["mscores"],
        )
        output_file = output_path / f"matches_{img0_path.stem}_{img1_path.stem}.png"
        visualizer.save(str(output_file))
        logger.info(f"Visualization saved to {output_file}")

    logger.success("Image matching done")


if __name__ == "__main__":
    match_images()
