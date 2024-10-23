from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import numpy as np
import torch
from loguru import logger

from imm.extractors._helper import create_extractor
from imm.matchers._helper import create_matcher
from imm.misc import extend_keys_with_suffix
from imm.settings import img0_path as default_img0_path
from imm.settings import img1_path as default_img1_path
from imm.utils.device import to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.logger import setup_logger
from imm.utils.viz2d import MatchVisualizer
from imm.utils.warnings import suppress_warnings


def load_and_process_image(image_path: str, max_size: Optional[int], device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and process an image."""
    logger.info(f"Loading image: {image_path}")
    data = load_image_tensor(image_path, resize=max_size)
    return data[0].to(device), data[1]


def extract_features(extractor: Any, image: torch.Tensor, suffix: str) -> Dict[str, Any]:
    """Extract features from an image and extend keys with a suffix."""
    logger.info(f"Extracting features for image{suffix}")
    preds = extractor.extract({"image": image})
    preds = extend_keys_with_suffix(preds, suffix)
    h, w = image.shape[-2:]
    preds[f"size{suffix}"] = torch.tensor([w, h])
    return preds


def compute_match_statistics(matches: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute statistics about the matches."""
    num_matches = len(matches["mkpts0"])
    if num_matches == 0:
        return {"num_matches": 0, "avg_score": 0, "max_score": 0, "min_score": 0}

    return {
        "num_matches": num_matches,
        "avg_score": float(np.mean(matches["mscores"])),
        "max_score": float(np.max(matches["mscores"])),
        "min_score": float(np.min(matches["mscores"])),
    }


@click.command()
@click.argument("img0_path", type=click.Path(exists=True), default=default_img0_path)
@click.argument("img1_path", type=click.Path(exists=True), default=default_img1_path)
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--max_size", default=None, type=int, help="Max image size")
@click.option("--output_dir", default="output", help="Output directory for logs and visualization")
@click.option("--threshold", default=0.0, help="Matching score threshold")
@click.option("--visualize/--no-visualize", default=True, help="Enable or disable visualization")
@click.option("--use_gpu/--no-gpu", default=True, help="Use GPU if available")
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
    use_gpu: bool,
) -> None:
    """Match features between two images."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logger(app_name="imm")

    logger.info("Starting image matching process")
    logger.info(f"Matcher: {matcher}, Extractor: {extractor}, Max size: {max_size}, Threshold: {threshold}")

    # Determine device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Convert paths
    img0_path = Path(img0_path)
    img1_path = Path(img1_path)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, max_size, device)
    image1, image1_cv = load_and_process_image(img1_path, max_size, device)

    # to cuda
    if device.type == "cuda":
        image0 = image0.cuda()
        image1 = image1.cuda()

    # Create matcher
    matcher = create_matcher(matcher)
    matcher.eval().to(device)

    #
    if "image0" in matcher.required_inputs:
        m_input = {"image0": image0, "image1": image1}
        matches = matcher.match(m_input)
        matches = to_numpy(matches)
    else:
        # Create extractor
        extractor = create_extractor(extractor)
        extractor.eval().to(device)

        # Extract features
        features0 = extract_features(extractor, image0, "0")
        features1 = extract_features(extractor, image1, "1")

        #
        logger.info(f"Matching img0: {len(features0['kpts0'][0])}, img1: {len(features1['kpts1'][0])}")
        matches = matcher.match({**features0, **features1})
        matches = to_numpy(matches)

    # Compute and log statistics
    stats = compute_match_statistics(matches)
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
        output_file = output_path / f"matches_{Path(img0_path).stem}_{Path(img1_path).stem}.png"
        visualizer.save(str(output_file))
        logger.info(f"Visualization saved to {output_file}")

    logger.success("Image matching Done")


if __name__ == "__main__":
    match_images()
