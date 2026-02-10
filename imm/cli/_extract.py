import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import click
import h5py
import matplotlib
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Set matplotlib backend before any pyplot imports
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

from imm.data import ImagesFromList
from imm.extractors._helper import create_extractor
from imm.utils.device import detect_device, to_cpu, to_cuda, to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.logger import set_log_dir
from imm.utils.manifest import save_extraction_manifest
from imm.viz import KeypointVisualizer
from imm.writers import FeaturesWriter


def filter_existing_extractions(dataset, save_path: Path, manifest_path: Path):
    """Read existing extractions and filter dataset to skip already extracted images.
    """
    # Read existing extractions from HDF5 file
    existing_keys = set()
    if save_path.exists():
        try:
            with h5py.File(save_path, "r") as h5_file:
                existing_keys = set(h5_file.keys())
        except Exception as e:
            logger.warning(f"Could not read HDF5 file: {e}")
    # Filter dataset based on existing extractions
    skipped_count = 0
    if existing_keys and hasattr(dataset, 'images_paths'):
        indices = [
            i for i, p in enumerate(dataset.images_paths)
            if p.name not in existing_keys
        ]
        skipped_count = len(dataset.images_paths) - len(indices)
        if skipped_count > 0:
            logger.warning(
                f"Skipping {skipped_count} already extracted images")

        if not indices:
            logger.warning("All images already extracted")
            early_result = ExtractionResult(
                output_file=save_path,
                manifest_path=manifest_path,
                processed_images=[],
                skipped_images=skipped_count,
                failed_images=0,
                total_time=0.0,
            )
            return dataset, skipped_count, early_result

        dataset = Subset(dataset, indices)

    return dataset, skipped_count, None


@dataclass
class ExtractionResult:
    """Result from dataset extraction."""
    output_file: Path
    manifest_path: Path
    processed_images: list[str]
    skipped_images: int
    failed_images: int
    total_time: float
    keypoint_counts: Optional[list[int]] = None
    processing_times_ms: Optional[list[float]] = None


class Extraction:
    def __init__(self, extractor: str, extractor_cfg: Optional[Dict[str, Any]] = None, device: str = "cpu", **kwargs: Any):
        """
        Initializes the feature extractor .
        """
        self.device = device
        self.extractor = create_extractor(
            name=extractor, cfg=extractor_cfg, **kwargs)
        self.extractor.to(self.device)
        self.extractor.eval()
        logger.info(f"Initialized {extractor} extractor on {device}")

    @torch.inference_mode()
    def extract_image(self, data: Mapping[str, Any]) -> Any:
        """
        Extracts features from a single input.

        Args:
            data: Batch data containing tensors and metadata (image, name, original_size, etc.)

        Returns:
            Dictionary of extracted features as numpy arrays
        """
        data = to_cuda(data) if self.device == "cuda" else to_cpu(
            data)  # type: ignore
        preds = self.extractor.extract(data)
        return to_numpy(preds)

    @staticmethod
    def print_extraction_details(iteration: int, name: str, preds: Dict[str, Any]) -> None:
        num_kpts = preds.get("kpts", []).shape[0] if "kpts" in preds else 0
        descriptor_size = preds.get(
            "desc", []).shape if "desc" in preds else (0,)
        size = preds.get("size", (0, 0))
        original_size = preds.get("original_size", (0, 0))

        print(f"Iteration {iteration + 1}:")
        print(f"    Image name: {name}")
        print(f"    Number of kpts: {num_kpts}")
        print(f"    Descriptor size: {descriptor_size}")
        print(f"    Image size: {size}")
        print(f"    Original size: {original_size}")

    @torch.inference_mode()
    def extract_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        save_path: Path,
        batch_size: int = 1,
        num_workers: int = 4,
        print_freq: int = 40,
        override: bool = False,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """
        Extracts features from a dataset and saves them to an HDF5 file.
        """
        manifest_path = save_path.parent / "extraction_manifest.json"

        # Filter dataset to skip already extracted images
        if not override:
            dataset, skipped_count, early_result = filter_existing_extractions(
                dataset, save_path, manifest_path
            )
            if early_result:
                return early_result
        else:
            skipped_count = 0

        # Create or use provided DataLoader
        if dataloader is None:
            dl_kwargs = dataloader_kwargs or {}
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True if self.device == "cuda" else False,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
                **dl_kwargs,
            )

        # Set up HDF5 writer
        with FeaturesWriter(save_path) as writer:
            start_time = time.time()
            processed_images = []
            keypoint_counts = []
            processing_times = []
            failed_count = 0

            for idx, data in enumerate(tqdm(dataloader, desc="Extracting".rjust(15), colour="blue")):
                name = data["name"][0]
                original_size = data["original_size"][0]

                img_start = time.time()
                try:
                    # Extract features
                    preds = self.extract_image(data)
                    preds = {k: v[0] if isinstance(
                        v, (list, tuple)) else v for k, v in preds.items()}

                    # Save features to HDF5 file
                    preds["original_size"] = original_size
                    writer.write_features(name, preds)
                    processed_images.append(name)

                    # Track statistics
                    if "kpts" in preds:
                        keypoint_counts.append(len(preds["kpts"]))
                    processing_times.append((time.time() - img_start) * 1000)

                    # Print extraction details
                    if (idx + 1) % print_freq == 0:
                        self.print_extraction_details(idx, name, preds)

                except Exception as e:
                    logger.exception(f"Failed to extract {name}")
                    failed_count += 1
                    continue

        total_time = time.time() - start_time

        # Get extractor config
        config = {
            "max_keypoints": getattr(self.extractor, "max_keypoints", -1),
            "det_threshold": getattr(self.extractor, "det_threshold", 0.0),
        }

        # Save manifest with statistics
        manifest_path = save_path.parent / "extraction_manifest.json"
        save_extraction_manifest(
            extractor_name=self.extractor.__class__.__name__,
            config=config,
            device=self.device,
            total_time=total_time,
            processed_images=processed_images,
            manifest_path=manifest_path,
            skipped_images=skipped_count,
            failed_images=failed_count,
            keypoint_counts=keypoint_counts if keypoint_counts else None,
            processing_times_ms=processing_times if processing_times else None,
            resume_mode=not override and skipped_count > 0,
        )

        logger.info(f"Features saved to {save_path}")
        logger.info(f"Total extraction time: {total_time:.2f} seconds")

        return ExtractionResult(
            output_file=save_path,
            manifest_path=manifest_path,
            processed_images=processed_images,
            skipped_images=skipped_count,
            failed_images=failed_count,
            total_time=total_time,
            keypoint_counts=keypoint_counts if keypoint_counts else None,
            processing_times_ms=processing_times if processing_times else None,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(extractor={self.extractor}, device={self.device})"


def extract_image(
    img_path: str | Path,
    extractor: str,
    max_keypoints: int,
    resize: int,
    output: Optional[str],
    show: bool,
    force_cpu: bool,
) -> None:
    """Extract features from a single image.

    Args:
        img_path: Path to the image file
        extractor: Name of the extractor model
        max_keypoints: Maximum number of keypoints to extract
        resize: Resize image to max dimension
        output: Directory to save visualization
        show: Display visualization
        force_cpu: Force CPU usage
    """
    device = detect_device(force_cpu)

    # Configure logging to output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        set_log_dir(log_dir=output_dir, app_name="extract")

    extraction = Extraction(extractor=extractor, extractor_cfg={
        "max_keypoints": max_keypoints}, device=device)

    img_path = Path(img_path)
    logger.info(f"Extracting features from {img_path}")

    # Load image
    data = load_image_tensor(str(img_path), resize)
    image, image_cv = data[0], data[1]

    # Extract features from a single image
    preds = extraction.extract_image({"image": image})

    # Flatten the output
    preds = {k: v[0] if isinstance(
        v, (list, tuple)) else v for k, v in preds.items()}

    # Extract keypoints, scores, and descriptors
    kpts = preds.get("kpts", None)
    scores = preds.get("scores", None)
    descs = preds.get("desc", None)

    logger.info(f"Keypoints: {kpts.shape if kpts is not None else 0}")
    logger.info(f"Descriptors: {descs.shape if descs is not None else 0}")

    # Visualize keypoints
    visualizer = KeypointVisualizer()
    if kpts is not None:
        visualizer.draw_keypoints(image_cv, kpts, scores, show_image=show)
    else:
        logger.warning("No keypoints detected in image")

    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"keypoints_{img_path.stem}.png"
        visualizer.save(str(output_file))
        logger.info(f"Visualization saved to {output_file}")

    logger.success("Image extraction completed")


def extract_dataset(
    dataset_dir: str,
    output: str,
    extractor: str,
    max_keypoints: int,
    resize: int,
    batch_size: int,
    num_workers: int,
    override: bool,
    print_freq: int,
    force_cpu: bool,
) -> None:
    """Extract features from a dataset of images.

    Args:
        dataset_dir: Path to the dataset directory
        output: Output directory for extracted features
        extractor: Name of the extractor model
        max_keypoints: Maximum number of keypoints to extract
        resize: Resize images to max dimension
        batch_size: Batch size for extraction
        num_workers: Number of DataLoader workers
        override: Override existing extracted features
        print_freq: Print frequency for extraction details
        force_cpu: Force CPU usage
    """
    device = detect_device(force_cpu)

    # Configure logging to output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(log_dir=output_dir, app_name="extract")

    extraction = Extraction(extractor=extractor, extractor_cfg={
        "max_keypoints": max_keypoints}, device=device)

    # Create a dataset from the directory
    dataset = ImagesFromList(dataset_dir, resize=resize)

    # Set up save path for dataset features
    save_path = output_dir / "features.h5"

    # Extract features from the dataset
    result = extraction.extract_dataset(
        dataset=dataset,
        save_path=save_path,
        batch_size=batch_size,
        num_workers=num_workers,
        print_freq=print_freq,
        override=override,
    )

    logger.success(f"Dataset extraction completed. Results saved to {output}")


@click.group()
@click.help_option("--help", "-h")
def cli():
    """Extract features from {image, dataset}."""
    pass


@cli.command('image')
@click.argument("img_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--max_keypoints", default=-1, type=int, help="Maximum number of keypoints")
@click.option("--resize", default=640, type=int, help="Resize to max dimension")
@click.option("--output", default=None, help="Directory to save extracted features")
@click.option("--show", is_flag=True, help="Display visualization")
@click.option("--force_cpu", is_flag=True, help="Force using CPU")
@click.help_option("--help", "-h")
def cmd_extract_image(
    img_path: str,
    extractor: str,
    max_keypoints: int,
    resize: int,
    output: Optional[str],
    show: bool,
    force_cpu: bool,
) -> None:
    """Extract features from a single image."""
    extract_image(
        img_path=img_path,
        extractor=extractor,
        max_keypoints=max_keypoints,
        resize=resize,
        output=output,
        show=show,
        force_cpu=force_cpu,
    )


@cli.command('dataset')
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", required=True, type=click.Path(), help="Output directory for extracted features")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--max_keypoints", default=-1, type=int, help="Maximum number of keypoints")
@click.option("--resize", default=640, type=int, help="Resize to max dimension")
@click.option("--batch_size", default=1, type=int, help="Batch size for dataset extraction")
@click.option("--num_workers", default=4, type=int, help="Number of workers for DataLoader")
@click.option("--override", is_flag=True, help="Override existing features (default: resume mode)")
@click.option("--print_freq", default=100, type=int, help="Frequency to print extraction details")
@click.option("--force_cpu", is_flag=True, help="Force using CPU")
@click.help_option("--help", "-h")
def cmd_extract_dataset(
    dataset_dir: str,
    output: str,
    extractor: str,
    max_keypoints: int,
    resize: int,
    batch_size: int,
    num_workers: int,
    override: bool,
    print_freq: int,
    force_cpu: bool,
) -> None:
    """Extract features from a dataset of images."""
    extract_dataset(
        dataset_dir=dataset_dir,
        output=output,
        extractor=extractor,
        max_keypoints=max_keypoints,
        resize=resize,
        batch_size=batch_size,
        num_workers=num_workers,
        override=override,
        print_freq=print_freq,
        force_cpu=force_cpu,
    )


if __name__ == "__main__":
    cli()
