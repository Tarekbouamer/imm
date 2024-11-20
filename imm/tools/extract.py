import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from imm.extractors._helper import create_extractor
from imm.settings import img0_path
from imm.utils.dataset import ImagesFromList
from imm.utils.device import detect_device, to_cpu, to_cuda, to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.viz2d import KeypointVisualizer
from imm.utils.writers import FeaturesWriter


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
        descriptor_size = preds.get("desc", []).shape if "desc" in preds else (0,)
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
    ) -> None:
        """
        Extracts features from a dataset and saves them to an HDF5 file.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        writer = FeaturesWriter(save_path)

        start_time = time.time()

        for idx, data in enumerate(tqdm(dataloader, desc="Extracting".rjust(15), colour="blue")):
            name = data["name"][0]
            original_size = data["original_size"][0]

            # Extract features
            preds = self.extract_single_image(data)
            preds = {k: v[0] if isinstance(v, (List, Tuple)) else v for k, v in preds.items()}

            # Save features to HDF5 file
            preds["original_size"] = original_size
            writer.write_features(name, preds)

            # Print extraction details
            if (idx + 1) % print_freq == 0:
                self.print_extraction_details(idx, name, preds)

        writer.close()
        total_time = time.time() - start_time
        logger.info(f"Features saved to {save_path}")
        logger.info(f"Total extraction time: {total_time:.2f} seconds")

    def __repr__(self):
        return f"{self.__class__.__name__}(extractor={self.extractor}, device={self.device})"


@click.command()
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--img_path", default=img0_path, help="Path to the image or dataset")
@click.option("--max_keypoints", default=-1, help="Maximum number of keypoints")
@click.option("--det_thd", default=0.0, help="Detector threshold")
@click.option("--max_img_size", default=640, help="Maximum image size for the extractor")
@click.option("--batch_size", default=1, help="Batch size for dataset extraction")
@click.option("--num_workers", default=4, help="Number of workers for DataLoader")
@click.option("--force_cpu", is_flag=True, help="Force using CPU")
@click.option("--print_freq", default=100, help="Frequency to print extraction details")
@click.option("--output_dir", default="output", help="Path to save extracted features")
@click.help_option("--help", "-h")
def extract(
    extractor: str,
    img_path: str,
    max_keypoints: int,
    det_thd: float,
    max_img_size: int,
    batch_size: int,
    num_workers: int,
    force_cpu: bool,
    print_freq: int,
    output_dir: str,
):
    """Extracts features from an image or a dataset."""
    #  device
    device = detect_device(force_cpu)

    if det_thd > 0:
        raise NotImplementedError("Detector threshold is not implemented yet.")

    # Feature extractor
    extractor = Extraction(extractor=extractor, cfg={"max_keypoints": max_keypoints}, device=device)

    img_path = Path(img_path)
    if img_path.is_file():
        logger.info(f"Extracting features from {img_path}")

        # Load image
        data = load_image_tensor(str(img_path), max_img_size)
        image, image_cv = data[0], data[1]

        # Extract features from a single image
        preds = extractor.extract_single_image({"image": image})

        # Flatten the output
        preds = {k: v[0] if isinstance(v, list) else v for k, v in preds.items()}

        # Extract keypoints, scores, and descriptors
        kpts = preds.get("kpts", None)
        scores = preds.get("scores", None)
        descs = preds.get("desc", None)

        logger.info(f"Keypoints: {kpts.shape if kpts is not None else 0}")
        logger.info(f"Descriptors: {descs.shape if descs is not None else 0}")

        # Visualize keypoints
        visualizer = KeypointVisualizer()
        visualizer.draw_keypoints(image_cv, kpts, scores)

        return preds

    elif img_path.is_dir():
        # Create a dataset from the directory
        dataset = ImagesFromList(img_path, max_img_size=max_img_size)

        # Set up save path for dataset features
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "features.h5"

        # Extract features from the dataset
        extractor.extract_dataset(
            dataset=dataset, save_path=save_path, batch_size=batch_size, num_workers=num_workers, print_freq=print_freq
        )

        return save_path
    else:
        logger.error(f"Invalid path: {img_path}. Please provide a valid image or dataset path.")
        return


if __name__ == "__main__":
    extract()
