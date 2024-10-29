import time
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from imm.extractors._helper import create_extractor
from imm.utils.device import to_cpu, to_cuda, to_numpy
from imm.writers import FeaturesWriter


def initialize_extractor(
    name: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    **kwargs: Any
) -> torch.nn.Module:
    """
    Initializes the feature extractor model.
    """

    extractor = create_extractor(name=name, cfg=cfg, **kwargs)
    extractor.to(device)
    extractor.eval()
    logger.info(f"Initialized {name} extractor on {device}")
    return extractor


@torch.inference_mode()
def extract_single_image(
    extractor: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Extracts features from a single input.
    """
    data = to_cuda(data) if device == "cuda" else to_cpu(data)
    preds = extractor.extract(data)
    return to_numpy(preds)


def print_extraction_details(iteration: int, preds: Dict[str, Any]) -> None:
    num_kpts = preds.get("kpts", []).shape[0] if "kpts" in preds else 0
    descriptor_size = preds.get("desc", []).shape if "desc" in preds else (0,)
    size = preds.get("size", (0, 0))
    original_size = preds.get("original_size", (0, 0))

    print(f"Iteration {iteration + 1}:")
    print(f"    Number of kpts: {num_kpts}")
    print(f"    Descriptor size: {descriptor_size}")
    print(f"    Image size: {size}")
    print(f"    Original size: {original_size}")


@torch.inference_mode()
def extract_dataset(
    extractor: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    save_path: Path,
    batch_size: int = 1,
    num_workers: int = 4,
    device: str = "cpu",
    print_freq: int = 10
) -> None:
    """
    Extracts features from a dataset and saves them to an HDF5 file.
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)
    writer = FeaturesWriter(save_path)

    start_time = time.time()

    for idx, data in enumerate(tqdm(dataloader, desc="Extracting".rjust(15), colour="blue")):
        name = data["name"][0]
        original_size = data["original_size"][0]

        # extract features
        preds = extract_single_image(extractor, data, device)
        preds = {k: v[0] if isinstance(
            v, (List, Tuple)) else v for k, v in preds.items()}

        # save features to HDF5 file
        preds["original_size"] = original_size
        writer.write_features(name, preds)

        # print extraction details
        if (idx + 1) % print_freq == 0:
            print_extraction_details(idx, preds)

    writer.close()
    total_time = time.time() - start_time
    logger.info(f"Features saved to {save_path}")
    logger.info(f"Total extraction time: {total_time:.2f} seconds")
