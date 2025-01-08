import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

import click
import cv2
import numpy as np
import torch
from loguru import logger

from imm.estimators import create_homography_estimator
from imm.extractors import __all__ as SUPPORTED_EXTRACTORS
from imm.matchers import __all__ as SUPPORTED_MATCHERS
from imm.tools.match import Matching
from imm.utils.device import detect_device
from imm.utils.warnings import suppress_warnings


class Image:
    def __init__(self, image_path: Union[str, Path], id: int = -1, max_size: Optional[int] = None):
        self.id = id
        self.matches: Dict[int, int] = {}
        self.data = self._load_image(str(image_path) if isinstance(image_path, Path) else image_path)

        if max_size is not None:
            self.data = self._resize(max_size)

    @classmethod
    def from_numpy(cls, image_array: np.ndarray, id: int = -1, max_size: Optional[int] = None) -> "Image":
        if not isinstance(image_array, np.ndarray):
            raise ValueError("image_array must be a numpy array.")
        instance = cls.__new__(cls)
        instance.id = id
        instance.matches = {}
        instance.data = image_array

        if max_size is not None:
            instance.data = instance._resize(max_size)

        return instance

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        return image

    def _resize(self, max_size: int) -> np.ndarray:
        h, w = self.data.shape[:2]
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(self.data, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def hw(self) -> Tuple[int, int]:
        return self.data.shape[:2]

    def corners(self) -> np.ndarray:
        h, w = self.hw()
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    def as_tensor(self, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        tensor = torch.from_numpy(self.data).permute(2, 0, 1).float() / 255.0
        return tensor.to(device)

    def image(self) -> np.ndarray:
        return self.data


class ImageStitcher:
    def __init__(
        self,
        extractor: str = "superpoint",
        matcher: str = "superglue_outdoor",
        backend: str = "opencv",
        max_size: Optional[int] = None,
        min_matches: int = 10,
        max_keypoints: int = 1600,
        device: str = "cpu",
    ):
        self.max_size = max_size
        self.min_matches = min_matches
        self.device = device

        # Matching
        self.matcher = Matching(
            matcher_name=matcher, extractor_name=extractor, max_keypoints=max_keypoints, device=device
        )

        # Homography
        self.homography_estimator = create_homography_estimator(backend=backend)

    def _find_matches(self, image0: Image, image1: Image) -> Dict[str, np.ndarray]:
        matches = self.matcher.match_images(image0.as_tensor(self.device), image1.as_tensor(self.device))
        return matches

    def _compute_homography(self, matches: Dict[str, np.ndarray]) -> np.ndarray:
        mkpts0, mkpts1 = matches["mkpts0"], matches["mkpts1"]
        res = self.homography_estimator.estimate(mkpts0, mkpts1)
        return res["H"]

    def _remove_black_regions(self, stitched_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            stitched_image = stitched_image[y : y + h, x : x + w]
        return stitched_image

    def _apply_homography(self, image: np.ndarray, H: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        return cv2.warpPerspective(image, H, output_size)

    def _draw_borders(self, image: np.ndarray, color: Tuple[int, int, int], thickness: int = 5) -> np.ndarray:
        h, w = image.shape[:2]
        bordered_image = cv2.rectangle(image, (0, 0), (w - 1, h - 1), color, thickness)
        return bordered_image

    def _compute_pairwise_matches(self, images: List[Image]) -> Dict[Tuple[int, int], int]:
        """Compute the number of matches between every pair of images."""
        pairwise_matches = {}
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                matches = self._find_matches(images[i], images[j])
                pairwise_matches[(i, j)] = len(matches["mkpts0"])
                pairwise_matches[(j, i)] = len(matches["mkpts0"])
        return pairwise_matches

    def _find_optimal_order(self, images: List[Image], pairwise_matches: Dict[Tuple[int, int], int]) -> List[int]:
        """Find the optimal order to stitch images using a greedy approach."""
        remaining_images = set(range(len(images)))
        order = []

        # Start with the image that has the most matches with others
        start_image = max(
            remaining_images, key=lambda x: sum(pairwise_matches[(x, y)] for y in remaining_images if y != x)
        )
        order.append(start_image)
        remaining_images.remove(start_image)

        # Greedily select the next image with the most matches to the current stitched image
        while remaining_images:
            last_image = order[-1]
            next_image = max(remaining_images, key=lambda x: pairwise_matches[(last_image, x)])
            order.append(next_image)
            remaining_images.remove(next_image)

        return order

    def stitch_images(self, images: List[Image], draw: bool = False) -> np.ndarray:
        # Compute pairwise matches
        pairwise_matches = self._compute_pairwise_matches(images)

        # Find the optimal order to stitch images
        optimal_order = self._find_optimal_order(images, pairwise_matches)
        logger.info(f"Optimal stitching order: {optimal_order}")

        # Reorder images based on the optimal order
        ordered_images = [images[i] for i in optimal_order]

        # Start with the first image in the optimal order
        stitched = Image.from_numpy(ordered_images[0].image())
        remaining_images = ordered_images[1:]

        # Define a list of colors for borders
        border_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        if draw:
            stitched.data = self._draw_borders(stitched.data, random.choice(border_colors))

        while remaining_images:
            logger.info(f"Remaining images to stitch: {len(remaining_images)}")
            found_match = False

            # Search for the next image with enough matches
            for i, next_image in enumerate(remaining_images):
                matches = self._find_matches(stitched, next_image)
                if len(matches["mkpts0"]) >= self.min_matches:
                    logger.info(f"Found a match with image {next_image.id} ({len(matches['mkpts0'])} matches)")
                    found_match = True
                    break

            if not found_match:
                logger.warning("No more images with sufficient matches found. Stopping stitching.")
                break

            # Compute homography and stitch the found image
            H = self._compute_homography(matches)

            corners1 = stitched.corners().reshape(-1, 1, 2)
            corners2 = cv2.perspectiveTransform(next_image.corners().reshape(-1, 1, 2), H)
            all_corners = np.vstack((corners1, corners2))
            [x_min, y_min], [x_max, y_max] = (
                np.int32(all_corners.min(axis=0).flatten()),
                np.int32(all_corners.max(axis=0).flatten()),
            )
            output_size = (x_max - x_min, y_max - y_min)

            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
            stitched_image = self._apply_homography(stitched.image(), translation @ H, output_size)

            if draw:
                next_image.data = self._draw_borders(
                    next_image.data, border_colors[len(stitched.matches) % len(border_colors)]
                )

            stitched_image[-y_min : -y_min + next_image.hw()[0], -x_min : -x_min + next_image.hw()[1]] = (
                next_image.image()
            )
            stitched_image = self._remove_black_regions(stitched_image)
            stitched = Image.from_numpy(stitched_image)

            # Remove the stitched image from the queue
            remaining_images.pop(i)

        logger.info("Stitching completed")
        return stitched.image()

    def __call__(self, image_paths: List[str], visualize: bool = False, draw: bool = False) -> np.ndarray:
        image_objects = [Image(image_path, id=i, max_size=self.max_size) for i, image_path in enumerate(image_paths)]
        stitched_image = self.stitch_images(image_objects, draw=draw)

        if visualize:
            self.visualize(stitched_image)

        return stitched_image

    def visualize(self, stitched_image: np.ndarray) -> None:
        window_name = "Stitched Image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, stitched_image)

        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
        cv2.destroyAllWindows()


@click.command()
@click.option("--input", required=True, help="Folder containing the images to be stitched.")
@click.option("--output", required=True, help="Path to save the stitched image.")
@click.option(
    "--extractor", default="superpoint", type=click.Choice(SUPPORTED_EXTRACTORS), help="Feature extractor to use."
)
@click.option(
    "--matcher", default="superglue_outdoor", type=click.Choice(SUPPORTED_MATCHERS), help="Feature matcher to use."
)
@click.option(
    "--backend", default="opencv", type=click.Choice(["opencv", "pycolmap", "poselib"]), help="Homography backend."
)
@click.option("--max_size", default=None, type=int, help="Maximum size (width or height) for resizing images.")
@click.option("--min_matches", default=500, type=int, help="Minimum number of matches to consider a valid pair.")
@click.option("--max_keypoints", default=-1, type=int, help="Maximum number of keypoints to detect.")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.option("--visualize", is_flag=True, help="Visualize the stitched image.")
@click.option("--draw", is_flag=True, help="Draw borders around each image in the stitched result.")
@suppress_warnings()
def main(
    input: str,
    output: str,
    extractor: str,
    matcher: str,
    backend: str,
    max_size: Optional[int],
    min_matches: int,
    max_keypoints: int,
    force_cpu: bool,
    visualize: bool,
    draw: bool,
) -> None:
    logger.info("Starting image stitching process")

    # Device
    device = detect_device(force_cpu)

    image_paths = sorted(
        [os.path.join(input, f) for f in os.listdir(input) if f.lower().endswith(("png", "jpg", "jpeg", "bmp", "tiff"))]
    )
    if not image_paths:
        logger.error("No images found in the specified folder")
        raise ValueError("No images to stitch")

    stitcher = ImageStitcher(
        extractor=extractor,
        matcher=matcher,
        backend=backend,
        max_size=max_size,
        min_matches=min_matches,
        max_keypoints=max_keypoints,
        device=device,
    )
    stitched_image = stitcher(image_paths, visualize=visualize, draw=draw)

    if os.path.isdir(output):
        output = os.path.join(output, "stitched_image.jpg")

    cv2.imwrite(output, stitched_image)
    logger.info(f"Stitched image saved to {output}")


if __name__ == "__main__":
    main()
