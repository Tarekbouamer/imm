import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import h5py
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from imm.data import FeaturesPairsDataset, ImagePairsDataset
from imm.extractors._helper import create_extractor
from imm.matchers._helper import create_matcher
from imm.utils.data import extend_keys_with_suffix
from imm.utils.device import detect_device, to_cpu, to_cuda, to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.logger import set_log_dir
from imm.utils.manifest import create_matching_manifest
from imm.utils.warnings import suppress_warnings
from imm.viz.viz2d import MatchVisualizer
from imm.writers import AsyncMatchesWriter, MatchesWriter


def filter_existing_matches(dataset, matches_file: Path, manifest_path: Path) -> tuple:
    """Read existing matches and filter dataset to skip already matched pairs.
    """
    # Read existing matches from HDF5 file
    existing_keys = set()
    if matches_file.exists():
        try:
            with h5py.File(matches_file, "r") as h5_file:
                existing_keys = set(h5_file.keys())
        except Exception as e:
            logger.warning(f"Could not read HDF5 file: {e}")

    # Filter dataset based on existing matches
    skipped_count = 0
    if existing_keys and hasattr(dataset, 'pairs'):
        indices = [
            i for i, (name0, name1) in enumerate(dataset.pairs)
            if pairs2key(name0, name1) not in existing_keys
        ]
        skipped_count = len(dataset.pairs) - len(indices)
        if skipped_count > 0:
            logger.warning(f"Skipping {skipped_count} already matched pairs")

        if not indices:
            logger.warning("All pairs already matched")
            early_result = MatchingResult(
                output_file=matches_file,
                manifest_path=manifest_path,
                processed_pairs=[],
                skipped_pairs=skipped_count,
                failed_pairs=0,
                total_time=0.0,
            )
            return dataset, skipped_count, early_result

        dataset = Subset(dataset, indices)

    return dataset, skipped_count, None


@dataclass
class MatchingResult:
    """Result from matching operations."""
    output_file: Path
    manifest_path: Path
    processed_pairs: list[list[str]]
    skipped_pairs: int
    failed_pairs: int
    total_time: float
    match_counts: Optional[list[int]] = None
    processing_times_ms: Optional[list[float]] = None


def path2key(name: str) -> str:
    """Converts a file path to a key."""
    return name.replace("/", "-")


def pairs2key(name0: str, name1: str) -> str:
    """Creates a key for a pair of items."""
    separator = "-"
    return separator.join((path2key(name0), path2key(name1)))


def parse_pairs_file(pairs_file: Path) -> List[Tuple[str, str]]:
    """Parse pairs file containing image pairs."""
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs


def load_and_process_image(
    image_path: str | Path, resize: Optional[int], device: str
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and process an image."""
    logger.debug(f"Loading image: {image_path}")
    data = load_image_tensor(image_path, resize=resize)
    return data[0].to(device), data[1]


class Matching:
    def __init__(
        self,
        matcher_name: str = "superglue_outdoor",
        extractor_name: str | None = "superpoint",
        max_keypoints: int = -1,
        device: Optional[str] = None,
        match_thd: float = 0.0,
        **kwargs: Any,
    ):
        """
        Initializes the Matching class and the matcher model. Sets up the extractor if needed.
        """
        self.device = str(device) if device else detect_device()
        self.matcher = create_matcher(name=matcher_name, cfg={
                                      "match_threshold": match_thd}, **kwargs)
        self.matcher.to(self.device)
        self.matcher.eval()
        logger.info(f"Initialized {matcher_name} matcher on {self.device}")

        self.extractor = None
        # Only initialize extractor for sparse matchers when extractor name is provided
        if extractor_name and "image0" not in self.matcher.required_inputs:
            self.set_extractor(extractor_name, max_keypoints)

    def set_extractor(self, extractor_name: str, max_keypoints: int) -> None:
        self.extractor = create_extractor(
            extractor_name, cfg={"max_keypoints": max_keypoints})
        self.extractor.eval().to(self.device)
        logger.info(f"Initialized {extractor_name} extractor on {self.device}")

    def extract_features(self, image: torch.Tensor, suffix: str) -> Dict[str, Any]:

        if self.extractor is None:
            raise ValueError("Extractor is required but was not initialized")

        logger.info(f"Extracting features for image{suffix}")
        preds = self.extractor.extract({"image": image})
        preds = extend_keys_with_suffix(preds, suffix)
        h, w = image.shape[-2:]
        preds[f"size{suffix}"] = torch.tensor([w, h])
        return preds

    @torch.inference_mode()
    def match_features(self, data: Dict[str, Any], match_thd: float = 0.0) -> Dict[str, Any]:
        """Matches a pair of descriptors or raw images."""
        data = to_cuda(data) if self.device == "cuda" else to_cpu(
            data)  # type: ignore
        preds = self.matcher.match(data)
        preds = to_numpy(preds)
        return self.filter_matches(preds, match_thd)

    def filter_matches(self, preds: Dict[str, np.ndarray], match_thd: float) -> Dict:
        if match_thd <= 0:
            return preds
        else:
            # Indices
            matches = preds["matches"]
            mscores = preds["mscores"]

            # Filter matches based on -1 and matching threshold
            valid0 = np.where(matches != -1)
            valid1 = np.where(mscores > match_thd)
            valid = np.intersect1d(valid0, valid1)

            non_valid = np.setdiff1d(np.arange(len(matches)), valid)

            # Update non-valid matches with -1
            preds["matches"][non_valid] = -1

            preds["mkpts0"] = preds["kpts0"][valid]
            preds["mkpts1"] = preds["kpts1"][matches[valid]]

            return preds

    @torch.inference_mode()
    def match_images(self, image0: torch.Tensor, image1: torch.Tensor, match_thd: float = 0.0) -> Dict[str, np.ndarray]:
        if "image0" in self.matcher.required_inputs:
            data = {"image0": image0, "image1": image1}
            preds = self.match_features(data, match_thd=match_thd)
            return preds
        else:
            if self.extractor is None:
                raise ValueError(
                    "Extractor is required for sparse matchers but was not initialized")
            features0 = self.extract_features(image0, "0")
            features1 = self.extract_features(image1, "1")
            logger.info(
                f"Matching img0: {len(features0['kpts0'][0])}, img1: {len(features1['kpts1'][0])}")
            data = {**features0, **features1}
            preds = self.match_features(data, match_thd=match_thd)
            return preds

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
        override: bool = False,
    ) -> MatchingResult:
        """Processes a dataset of image pairs and saves matching results."""
        if batch_size != 1:
            raise ValueError("Only batch_size=1 is supported")

        # Treat save_path as directory and create matches.h5 inside
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        matches_file = save_dir / "matches.h5"
        manifest_path = save_dir / "matching_manifest.json"

        # Filter dataset to skip already matched pairs
        if not override:
            dataset, skipped_count, early_result = filter_existing_matches(
                dataset, matches_file, manifest_path
            )
            if early_result:
                return early_result
        else:
            skipped_count = 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        start_time = time.time()
        with MatchesWriter(matches_file) as writer:
            processed_pairs = []
            match_counts = []
            processing_times = []
            failed_count = 0

            for idx, data in enumerate(tqdm(dataloader, desc="Matching images".rjust(15), colour="green")):
                name0, name1 = data["name0"][0], data["name1"][0]
                image0 = data["image0"][0].to(self.device)
                image1 = data["image1"][0].to(self.device)

                pair_start = time.time()
                try:
                    # Match images directly
                    preds = self.match_images(
                        image0, image1, match_thd=match_thd)
                    pair_key = pairs2key(name0, name1)

                    # Write matches and stats to file
                    writer.write_matches(pair_key, preds)
                    processed_pairs.append([name0, name1])

                    # Track statistics
                    if "mkpts0" in preds:
                        match_counts.append(len(preds["mkpts0"]))
                    processing_times.append((time.time() - pair_start) * 1000)

                    # Compute statistics
                    if (idx + 1) % print_freq == 0:
                        stats = self.compute_match_statistics(
                            idx, preds, name0, name1)
                except Exception as e:
                    logger.exception(f"Failed to match pair {name0} - {name1}")
                    failed_count += 1
                    continue

        total_time = time.time() - start_time

        # Get matcher config
        config = {
            "match_threshold": getattr(self.matcher, "match_threshold", 0.0),
        }

        # Save manifest
        create_matching_manifest(
            matcher_name=self.matcher.__class__.__name__,
            config=config,
            device=self.device,
            total_time=total_time,
            processed_pairs=processed_pairs,
            manifest_path=manifest_path,
            extractor_name=self.extractor.__class__.__name__ if self.extractor else None,
            skipped_pairs=skipped_count,
            failed_pairs=failed_count,
            match_counts=match_counts if match_counts else None,
            processing_times_ms=processing_times if processing_times else None,
            resume_mode=not override and skipped_count > 0,
        )

        logger.info(f"Matches saved to {matches_file}")
        logger.info(f"Manifest saved to {manifest_path}")
        logger.info(
            f"Total processing time for image pairs: {total_time:.2f} seconds")

        return MatchingResult(
            output_file=matches_file,
            manifest_path=manifest_path,
            processed_pairs=processed_pairs,
            skipped_pairs=skipped_count,
            failed_pairs=failed_count,
            total_time=total_time,
            match_counts=match_counts if match_counts else None,
            processing_times_ms=processing_times if processing_times else None,
        )

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
        override: bool = False,
    ) -> MatchingResult:
        """Processes a dataset of pre-extracted feature pairs and saves matching results."""
        if batch_size != 1:
            raise ValueError("Only batch_size=1 is supported")

        # Treat save_path as directory and create matches.h5 inside
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        matches_file = save_dir / "matches.h5"
        manifest_path = save_dir / "matching_manifest.json"

        # Filter dataset to skip already matched pairs
        if not override:
            dataset, skipped_count, early_result = filter_existing_matches(
                dataset, matches_file, manifest_path
            )
            if early_result:
                return early_result
        else:
            skipped_count = 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        start_time = time.time()
        with AsyncMatchesWriter(matches_file, num_workers=num_workers) as writer:
            processed_pairs = []
            match_counts = []
            processing_times = []
            failed_count = 0

            for idx, data in enumerate(tqdm(dataloader, desc="Matching features".rjust(15), colour="green")):
                name0, name1 = data["name0"][0], data["name1"][0]

                pair_start = time.time()
                try:
                    # Match pre-extracted features
                    preds = self.match_features(data, match_thd=match_thd)
                    pair_key = pairs2key(name0, name1)

                    # Filter keys
                    wpreds = {k: preds[k] for k in keys} if keys else preds

                    # Write matches and stats to file
                    writer.write_matches(pair_key, wpreds)
                    processed_pairs.append([name0, name1])

                    # Track statistics
                    if "mkpts0" in preds:
                        match_counts.append(len(preds["mkpts0"]))
                    processing_times.append((time.time() - pair_start) * 1000)

                    # Compute statistics
                    if (idx + 1) % print_freq == 0:
                        stats = self.compute_match_statistics(
                            idx, preds, name0, name1)
                except Exception as e:
                    logger.exception(f"Failed to match pair {name0} - {name1}")
                    failed_count += 1
                    continue

        total_time = time.time() - start_time

        # Get matcher config
        config = {
            "match_threshold": getattr(self.matcher, "match_threshold", 0.0),
        }

        # Save manifest
        create_matching_manifest(
            matcher_name=self.matcher.__class__.__name__,
            config=config,
            device=self.device,
            total_time=total_time,
            processed_pairs=processed_pairs,
            manifest_path=manifest_path,
            features_file=str(dataset.features_path) if hasattr(
                dataset, 'features_path') else None,
            skipped_pairs=skipped_count,
            failed_pairs=failed_count,
            match_counts=match_counts if match_counts else None,
            processing_times_ms=processing_times if processing_times else None,
            resume_mode=not override and skipped_count > 0,
        )

        logger.info(f"Matches saved to {matches_file}")
        logger.info(f"Manifest saved to {manifest_path}")
        logger.info(
            f"Total processing time for feature pairs: {total_time:.2f} seconds")

        return MatchingResult(
            output_file=matches_file,
            manifest_path=manifest_path,
            processed_pairs=processed_pairs,
            skipped_pairs=skipped_count,
            failed_pairs=failed_count,
            total_time=total_time,
            match_counts=match_counts if match_counts else None,
            processing_times_ms=processing_times if processing_times else None,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(matcher={self.matcher}, extractor={self.extractor}, device={self.device})"


@click.group()
@click.help_option("--help", "-h")
def cli():
    """Match images supports {pair, sequence of images, sequence of features}."""
    pass


@cli.command('pair')
@click.argument("img0_path", type=click.Path(exists=True))
@click.argument("img1_path", type=click.Path(exists=True))
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--max_keypoints", default=-1, type=int, help="Maximum number of keypoints")
@click.option("--match_thd", default=0.0, help="Matching score threshold")
@click.option("--resize", default=640, type=int, help="Resize to max dimension")
@click.option("--output", default=None, help="Directory to save output")
@click.option("--show", is_flag=True, help="Display visualization")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.help_option("--help", "-h")
@suppress_warnings()
def match_pair(
    img0_path: str,  # type: ignore
    img1_path: str,  # type: ignore
    matcher: str,
    extractor: str,
    max_keypoints: int,
    match_thd: float,
    resize: int,
    output: Optional[str],
    show: bool,
    force_cpu: bool,
) -> None:
    """Match a pair of images."""
    device = detect_device(force_cpu)

    # Configure logging to output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        set_log_dir(log_dir=output_dir, app_name="match")

    img0_path: Path = Path(img0_path)
    img1_path: Path = Path(img1_path)

    image0, image0_cv = load_and_process_image(img0_path, resize, device)
    image1, image1_cv = load_and_process_image(img1_path, resize, device)

    matching: Matching = Matching(matcher_name=matcher, extractor_name=extractor,
                                  max_keypoints=max_keypoints, device=device)

    m_preds = matching.match_images(image0, image1, match_thd=match_thd)

    visualizer = MatchVisualizer()
    visualizer.draw_matches(image0_cv,
                            image1_cv,
                            image0_cv,
                            image1_cv,
                            m_preds["kpts0"],
                            m_preds["kpts1"],
                            m_preds["mkpts0"],
                            m_preds["mkpts1"],
                            title="Matches", show_image=show)

    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / \
            f"matches_{img0_path.stem}_{img1_path.stem}.png"
        visualizer.save(str(output_file))
        logger.info(f"Visualization saved to {output_file}")

    logger.success("Pair matching completed")


@cli.command('images')
@click.argument("images_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--pairs", required=True, type=click.Path(exists=True), help="Path to pairs file")
@click.option("--output", required=True, type=click.Path(), help="Output directory for matches")
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--max_keypoints", default=-1, type=int, help="Maximum number of keypoints")
@click.option("--match_thd", default=0.0, help="Matching score threshold")
@click.option("--resize", default=640, type=int, help="Resize to max dimension")
@click.option("--batch_size", default=1, type=int, help="Batch size for sequence matching")
@click.option("--num_workers", default=4, type=int, help="Number of workers for sequence matching")
@click.option("--print_freq", default=100, type=int, help="Print frequency for sequence matching")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.help_option("--help", "-h")
@suppress_warnings()
def match_images(
    images_dir: str,
    pairs: str,
    output: str,
    matcher: str,
    extractor: str,
    max_keypoints: int,
    match_thd: float,
    resize: int,
    batch_size: int,
    num_workers: int,
    print_freq: int,
    force_cpu: bool,
) -> None:
    """Match multiple image pairs from directory."""
    device = detect_device(force_cpu)

    # Configure logging to output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(log_dir=output_dir, app_name="match")

    images_dir_path = Path(images_dir)
    pairs_path = Path(pairs)

    pairs_data = parse_pairs_file(pairs_path)
    dataset = ImagePairsDataset(images_dir_path, pairs_data, resize=resize)

    matching: Matching = Matching(
        matcher_name=matcher,
        extractor_name=extractor,
        max_keypoints=max_keypoints,
        device=device,
        match_thd=match_thd
    )

    matching.match_sequence_images(
        dataset=dataset,
        save_path=Path(output),
        batch_size=batch_size,
        num_workers=num_workers,
        print_freq=print_freq,
        match_thd=match_thd
    )

    logger.success(
        f"Image sequence matching completed. Results saved to {output}")


@cli.command('features')
@click.argument("features_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--pairs", required=True, type=click.Path(exists=True), help="Path to pairs file")
@click.option("--output", required=True, type=click.Path(), help="Output directory for matches")
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--batch_size", default=1, type=int, help="Batch size for sequence matching")
@click.option("--num_workers", default=16, type=int, help="Number of workers for sequence matching")
@click.option("--print_freq", default=100, type=int, help="Print frequency for sequence matching")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.help_option("--help", "-h")
@suppress_warnings()
def match_features(
    features_path: str,
    pairs: str,
    output: str,
    matcher: str,
    batch_size: int,
    num_workers: int,
    print_freq: int,
    force_cpu: bool,
) -> None:
    """Match from pre-extracted features file."""
    device = detect_device(force_cpu)

    # Configure logging to output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(log_dir=output_dir, app_name="match")

    features_path_obj = Path(features_path)
    pairs_path = Path(pairs)

    if not features_path_obj.suffix == '.h5':
        raise ValueError(
            f"Features path must end in .h5, got: {features_path}")

    pairs_data = parse_pairs_file(pairs_path)
    dataset = FeaturesPairsDataset(features_path_obj, pairs_data)

    matching: Matching = Matching(
        matcher_name=matcher,
        extractor_name=None,
        max_keypoints=-1,
        device=device
    )

    matching.match_sequence_features(
        dataset=dataset,
        save_path=Path(output),
        batch_size=batch_size,
        num_workers=num_workers,
        print_freq=print_freq,
        match_thd=0.0
    )

    logger.success(
        f"Features matching completed. Results saved to {output}")


if __name__ == "__main__":
    cli()
