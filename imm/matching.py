import time
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from imm.matchers._helper import create_matcher
from imm.utils.device import to_cpu, to_cuda, to_numpy
from imm.writers import MatchesWriter


def path2key(name: str) -> str:
    """
    Converts a file path to a key.
    """
    return name.replace("/", "-")


def pairs2key(name0: str, name1: str) -> str:
    """
    Creates a key for a pair of items.
    """
    separator = "/"
    return separator.join((path2key(name0), path2key(name1)))


def initialize_matcher(
    name: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    **kwargs: Any
) -> torch.nn.Module:
    """
    Initializes the matcher model.
    """
    matcher = create_matcher(name=name, cfg=cfg, **kwargs)
    matcher.to(device)
    matcher.eval()
    logger.info(f"Initialized {name} matcher on {device}")
    return matcher


@torch.inference_mode()
def match_pair(
    matcher: torch.nn.Module,
    data: Dict[str, Union[torch.Tensor, List, Tuple]],
    device: str = "cpu"
) -> Dict:
    """
    Matches a pair of descriptors.
    """
    data = to_cuda(data) if device == "cuda" else to_cpu(data)
    preds = matcher.match(data)
    return to_numpy(preds)


def print_matching_details(iteration: int, preds: Dict[str, Any]) -> None:
    num_matches = preds.get(
        "matches", []).shape[0] if "matches" in preds else 0
    mkpt0 = preds.get(
        "mkpts0", []).shape[0] if "mkpts0" in preds else (0,)

    print(f"Iteration {iteration + 1}:")
    print(f"    Number of matches: {num_matches}")
    print(f"    Matched keypoints: {mkpt0}")


@torch.inference_mode()
def match_sequence(
    matcher: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    save_path: Path,
    batch_size: int = 1,
    num_workers: int = 4,
    device: str = "cpu",
    print_freq: int = 10
) -> None:
    """
    Matches a sequence of descriptor pairs and saves them to an HDF5 file.
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)
    writer = MatchesWriter(save_path)

    start_time = time.time()

    for idx, data in enumerate(tqdm(dataloader, desc="Matching".rjust(15), colour="green")):
        name0, name1 = data["name0"][0], data["name1"][0]

        # match the pair
        preds = match_pair(matcher, data, device)
        pair_key = pairs2key(name0, name1)

        # write matches
        writer.write_matches(pair_key, preds)

        # print matching details
        if (idx + 1) % print_freq == 0:
            print_matching_details(idx, preds)

    writer.close()
    total_time = time.time() - start_time
    logger.info(f"Matches saved to {save_path}")
    logger.info(f"Total matching time: {total_time:.2f} seconds")
