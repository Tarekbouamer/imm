import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import h5py
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from imm.extractors._helper import create_extractor
from imm.settings import img0_path
from imm.data import ImagesFromList
from imm.utils.device import detect_device, to_cpu, to_cuda, to_numpy
from imm.utils.io import load_image_tensor
from imm.writers import FeaturesWriter
from imm.utils import create_extraction_manifest, save_manifest
from imm.viz import KeypointVisualizer


class Extraction:
    def __init__(self, extractor: str, cfg: Optional[Dict[str, Any]] = None, device: str = "cpu", **kwargs: Any):
        """
        Initializes the feature extractor .
        """
        self.device = device
        self.extractor = create_extractor(name=extractor, cfg=cfg, **kwargs)
        self.extractor.to(self.device)
        self.extractor.eval()
        logger.info(f"Initialized {extractor} extractor on {device}")

    @torch.inference_mode()
    def extract_single_image(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Extracts features from a single input.
        """
        data = to_cuda(data) if self.device == "cuda" else to_cpu(data)
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
    ) -> None:
        """
        Extracts features from a dataset and saves them to an HDF5 file.
        """
        # Get existing images from HDF5 if not override
        existing_images = set()
        skipped_count = 0
        if not override and save_path.exists():
            try:
                with h5py.File(save_path, "r") as h5_file:
                    existing_images = set(h5_file.keys())
                logger.info(
                    f"Found {len(existing_images)} already extracted images")
            except Exception as e:
                logger.warning(f"Could not read HDF5 file: {e}")

        # Filter dataset to skip already extracted images
        if existing_images:
            original_count = len(dataset.images_paths)
            dataset.images_paths = [
                p for p in dataset.images_paths if p.stem not in existing_images]
            skipped_count = original_count - len(dataset.images_paths)
            logger.info(f"Skipping {skipped_count} already extracted images")

        if len(dataset.images_paths) == 0:
            logger.info("All images already extracted")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )
        writer = FeaturesWriter(save_path)

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
                preds = self.extract_single_image(data)
                preds = {k: v[0] if isinstance(
                    v, (List, Tuple)) else v for k, v in preds.items()}

                # Save features to HDF5 file
                preds["original_size"] = original_size
                writer.write_features(name, preds)
                processed_images.append(name)

                # Track statistics
                if "keypoints" in preds:
                    keypoint_counts.append(len(preds["keypoints"]))
                processing_times.append((time.time() - img_start) * 1000)

                # Print extraction details
                if (idx + 1) % print_freq == 0:
                    self.print_extraction_details(idx, name, preds)

            except Exception as e:
                logger.error(f"Failed to extract {name}: {e}")
                manifest_writer.add_error(name, str(e))
                failed_count += 1
                continue

        writer.close()
        total_time = time.time() - start_time

        # Get extractor config
        config = {
            "max_keypoints": getattr(self.extractor, "max_keypoints", -1),
            "det_threshold": getattr(self.extractor, "det_threshold", 0.0),
        }

        # Save manifest with statistics
        manifest_path = save_path.parent / f"{save_path.stem}_manifest.json"
        manifest = create_extraction_manifest(
            extractor_name=self.extractor.__class__.__name__,
            config=config,
            device=self.device,
            total_time=total_time,
            processed_images=processed_images,
            output_file=str(save_path),
            skipped_images=skipped_count,
            failed_images=failed_count,
            keypoint_counts=keypoint_counts if keypoint_counts else None,
            processing_times_ms=processing_times if processing_times else None,
            resume_mode=not override and len(existing_images) > 0,
        )
        save_manifest(manifest, manifest_path)

        logger.info(f"Features saved to {save_path}")
        logger.info(f"Total extraction time: {total_time:.2f} seconds")

    def __repr__(self):
        return f"{self.__class__.__name__}(extractor={self.extractor}, device={self.device})"


@click.command()
@click.option("--img_path", default=img0_path, help="Path to the image or dataset")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--max_keypoints", default=-1, help="Maximum number of keypoints")
@click.option("--det_thd", default=0.0, help="Detector threshold")
@click.option("--resize", default=640, help="Resize to max dimension")
@click.option("--output", default=None, help="Directory to save extracted features")
@click.option("--show", is_flag=True, help="Display visualization")
@click.option("--batch_size", default=1, help="Batch size for dataset extraction")
@click.option("--num_workers", default=4, help="Number of workers for DataLoader")
@click.option("--override", is_flag=True, help="Override existing extracted features")
@click.option("--print_freq", default=100, help="Frequency to print extraction details")
@click.option("--force_cpu", is_flag=True, help="Force using CPU")
@click.help_option("--help", "-h")
def extract(
    img_path: str,
    extractor: str,
    max_keypoints: int,
    det_thd: float,
    resize: int,
    output: Optional[str],
    show: bool,
    batch_size: int,
    num_workers: int,
    override: bool,
    print_freq: int,
    force_cpu: bool,
):
    """Extracts features from an image or a dataset."""
    #  device
    device = detect_device(force_cpu)

    if det_thd > 0:
        raise NotImplementedError("Detector threshold is not implemented yet.")

    # Feature extractor
    extractor = Extraction(extractor=extractor, cfg={
                           "max_keypoints": max_keypoints}, device=device)

    img_path = Path(img_path)
    if img_path.is_file():
        logger.info(f"Extracting features from {img_path}")

        # Load image
        data = load_image_tensor(str(img_path), resize)
        image, image_cv = data[0], data[1]

        # Extract features from a single image
        preds = extractor.extract_single_image({"image": image})

        # Flatten the output
        preds = {k: v[0] if isinstance(
            v, list) else v for k, v in preds.items()}

        # Extract keypoints, scores, and descriptors
        kpts = preds.get("kpts", None)
        scores = preds.get("scores", None)
        descs = preds.get("desc", None)

        logger.info(f"Keypoints: {kpts.shape if kpts is not None else 0}")
        logger.info(f"Descriptors: {descs.shape if descs is not None else 0}")

        # Visualize keypoints
        visualizer = KeypointVisualizer()
        visualizer.draw_keypoints(image_cv, kpts, scores, show_image=show)

        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"keypoints_{img_path.stem}.png"
            visualizer.save(str(output_file))
            logger.info(f"Visualization saved to {output_file}")

        return preds

    elif img_path.is_dir():
        # Create a dataset from the directory
        dataset = ImagesFromList(img_path, max_img_size=resize)

        # Set up save path for dataset features
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "features.h5"

        # Extract features from the dataset
        extractor.extract_dataset(
            dataset=dataset,
            save_path=save_path,
            batch_size=batch_size,
            num_workers=num_workers,
            print_freq=print_freq,
            override=override,
        )

        return save_path
    else:
        logger.error(
            f"Invalid path: {img_path}. Please provide a valid image or dataset path.")
        return


if __name__ == "__main__":
    extract()
