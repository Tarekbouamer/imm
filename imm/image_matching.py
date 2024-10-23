from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import click
import numpy as np
import torch
from imm.utils.device import to_numpy
from imm.utils.io import load_image_tensor
from loguru import logger

from imm.base import FeatureModel, MatcherModel
from imm.extractors._helper import create_extractor
from imm.matchers._helper import create_matcher
from imm.misc import extend_keys_with_suffix
from imm.settings import img0_path as default_img0_path
from imm.settings import img1_path as default_img1_path
from imm.utils.logger import setup_logger
from imm.utils.viz2d import MatchVisualizer


class ImageMatcher:
    def __init__(
        self,
        extractor: Union[str, FeatureModel],
        matcher: Union[str, MatcherModel],
        use_gpu: bool = True,
    ):
        # Set device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Create extractor and matcher instances
        self.extractor = create_extractor(extractor) if isinstance(extractor, str) else extractor
        self.matcher = create_matcher(matcher) if isinstance(matcher, str) else matcher

        self.extractor.eval().to(self.device)
        self.matcher.eval().to(self.device)

        # Visualizer
        self.mv = MatchVisualizer()

    def load_and_process_image(self, image_path: Union[str, Path], max_size: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray]:
        logger.info(f"Loading image: {image_path}")
        data = load_image_tensor(str(image_path), resize=max_size)

        image_tensor = data[0].to(self.device)
        image_cv = data[1]
        return image_tensor, image_cv

    def extract_image_features(self, image: torch.Tensor, suffix: str) -> Dict[str, Any]:
        logger.info(f"Extracting features for image{suffix}")

        preds = self.extractor.extract({"image": image})
        preds = extend_keys_with_suffix(preds, suffix)
        h, w = image.shape[-2:]
        preds[f"size{suffix}"] = torch.tensor([w, h])
        return preds

    @staticmethod
    def compute_match_statistics(
        matches: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        num_matches = len(matches["mkpts0"])
        if num_matches == 0:
            stats = {
                "num_matches": 0,
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
            }
        else:
            stats = {
                "num_matches": num_matches,
                "avg_score": float(np.mean(matches["mscores"])),
                "max_score": float(np.max(matches["mscores"])),
                "min_score": float(np.min(matches["mscores"])),
            }

        # Log statistics
        logger.info("Match statistics:")
        logger.info(f"  Number of matches: {stats['num_matches']}")
        logger.info(f"  Score: Min: {stats['min_score']:.2f}, Max: {stats['max_score']:.2f}, Avg: {stats['avg_score']:.2f}")

        return stats

    def match_pairs(
        self,
        img0_path: Any,
        img1_path: Any,
        max_size: Optional[int] = None,
        max_keypoints: int = -1,
        min_conf: float = 0.0,
        visualize: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Load and process images
        if isinstance(img0_path, (str, Path)):
            image0_name = Path(img0_path).stem
            image1_name = Path(img1_path).stem
            image0, image0_cv = self.load_and_process_image(img0_path, max_size)
            image1, image1_cv = self.load_and_process_image(img1_path, max_size)
        else:
            image0 = img0_path
            image1 = img1_path
            image0_name = kwargs.get("image0_name", "img0")
            image1_name = kwargs.get("image1_name", "img1")
            image0_cv = kwargs.get("image0_cv", None)
            image1_cv = kwargs.get("image1_cv", None)

        # Match images
        if "image0" in self.matcher.required_inputs:
            m_inputs = {"image0": image0, "image1": image1}
            matches = self.matcher.match(m_inputs)
            matches = to_numpy(matches)
        else:
            # Extract features
            features0 = self.extract_image_features(image0, "0")
            features1 = self.extract_image_features(image1, "1")

            # Match features
            logger.info(f"Matching img0: {len(features0['kpts0'][0])}, img1: {len(features1['kpts1'][0])}")

            inputs = {**features0, **features1}

            matches = self.matcher.match(inputs)
            matches = to_numpy(matches)

        # Apply threshold to matches
        filtered_matches = self.best_matches(matches, min_conf, max_keypoints)

        # Compute statistics
        stats = self.compute_match_statistics(filtered_matches)

        results = {
            "matches": filtered_matches,
            "stats": stats,
        }

        # Visualize matches
        if visualize:
            vis_res = self.visualize(
                filtered_matches,
                image0_cv,
                image1_cv,
                output_dir,
                image0_name,
                image1_name,
            )
            results.update(vis_res)

        return results

    def visualize(
        self,
        matches: Dict[str, np.ndarray],
        image0_cv: np.ndarray,
        image1_cv: np.ndarray,
        output_dir: Optional[Union[str, Path]],
        img0_name: str = "img0",
        img1_name: str = "img1",
    ) -> Optional[Path]:
        # Visualize matches
        logger.info("Creating and saving match visualization")
        composite_image = self.mv.visualize_matches(
            image0_cv,
            image1_cv,
            kpts0=matches["kpts0"],
            kpts1=matches["kpts1"],
            mkpts0=matches["mkpts0"],
            mkpts1=matches["mkpts1"],
            scores=matches["mscores"],
        )

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"matches_{img0_name}_{img1_name}.png"
            self.mv.save(str(output_file))
            logger.info(f"Visualization saved to {output_file}")
            return {
                "visualization_path": output_file,
                "composite_image": composite_image,
            }
        else:
            return {"composite_image": composite_image}

    def best_matches(
        self,
        matches: Dict[str, np.ndarray],
        min_conf: float = 0.0,
        max_keypoints: int = -1,
    ) -> Dict[str, np.ndarray]:
        # Filter matches based on minimum confidence score
        if min_conf > 0.0:
            mscores = matches["mscores"]
            m_valid = mscores > min_conf
            for k in ["mkpts0", "mkpts1", "mscores"]:
                if len(matches[k]) == 0:
                    continue

                matches[k] = matches[k][m_valid]

        # Select top k matches based on confidence score
        if max_keypoints > 0:
            mscores = matches["mscores"]
            m_topk = np.argsort(mscores)[::-1][:max_keypoints]
            for k in ["mkpts0", "mkpts1", "mscores"]:
                if len(matches[k]) == 0:
                    continue

                matches[k] = matches[k][m_topk]

        return matches


@click.command()
@click.argument("img0_path", type=click.Path(exists=True), default=default_img0_path)
@click.argument("img1_path", type=click.Path(exists=True), default=default_img1_path)
@click.option(
    "--extractor",
    default="superpoint",
    help="Feature extractor name or instance",
)
@click.option("--matcher", default="nn", help="Feature matcher name or instance")
@click.option(
    "--max_size",
    default=None,
    type=int,
    help="Maximum size to resize the images",
)
@click.option(
    "--max_keypoints",
    default=-1,
    type=int,
    help="Maximum number of keypoints to extract",
)
@click.option("--min_conf", default=0.0, type=float, help="Minimum confidence threshold")
@click.option("--viz", is_flag=True, help="Visualize the matches")
@click.option(
    "--output_dir",
    default="output",
    type=click.Path(),
    help="Directory to save the visualization",
)
@click.option("--use_gpu/--no_gpu", default=True, help="Use GPU if available")
def image_match(
    img0_path,
    img1_path,
    extractor,
    matcher,
    max_size,
    max_keypoints,
    min_conf,
    viz,
    output_dir,
    use_gpu,
):
    logger = setup_logger(app_name="image_matching", console_level="INFO", file_level="DEBUG")

    # Image matching
    im = ImageMatcher(extractor, matcher, use_gpu)

    #
    results = im.match_pairs(
        img0_path,
        img1_path,
        max_size=max_size,
        min_conf=min_conf,
        max_keypoints=max_keypoints,
        visualize=viz,
        output_dir=output_dir,
    )

    if "visualization_path" in results:
        click.echo(f"Visualization saved at: {results['visualization_path']}")
    else:
        click.echo(f"Matches found: {results['stats']['num_matches']}")


if __name__ == "__main__":
    image_match()
