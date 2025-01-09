import os
import random
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def load_images(path: str, max_size: Optional[int] = None) -> List[np.ndarray]:
    files: List[str] = os.listdir(path)

    image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_files: List[str] = sorted([f for f in files if os.path.splitext(f)[1].lower() in image_extensions])

    image_paths: List[str] = [os.path.join(path, f) for f in image_files]

    list_image: List[np.ndarray] = []
    for image_path in image_paths:
        image: Optional[np.ndarray] = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image {image_path}. Skipping...")
            continue

        if max_size is not None:
            h: int
            w: int
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale: float = max_size / max(h, w)
                new_size: Tuple[int, int] = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size)

        list_image.append(image)

    return list_image


def to_tensor(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).to(device)


class ImageStitcher:
    def __init__(
        self,
        extractor: str = "superpoint",
        matcher: str = "superglue_outdoor",
        backend: str = "opencv",
        max_keypoints: int = 1600,
        device: str = "cpu",
    ) -> None:
        self.device: str = device

        # Matching
        self.matcher: Matching = Matching(
            matcher_name=matcher, extractor_name=extractor, max_keypoints=max_keypoints, device=device
        )

        # Homography
        self.homography_estimator: Any = create_homography_estimator(backend=backend)

        logger.info(f"ImageStitcher initialized with backend={backend}, extractor={extractor}, matcher={matcher}")

    def _remove_black_area(self, panorama: np.ndarray, h_dst: int, conners: np.ndarray) -> np.ndarray:
        """Remove black area in panorama image"""
        # Min max of x,y coorners
        [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
        t: List[int] = [-xmin, -ymin]
        conners = conners.astype(int)

        # conners[0][0][0] is the X coordinate of top-left point of warped image
        # If it has value<0, warp image is merged to the left side of destination image
        # otherwise is merged to the right side of destination image
        if conners[0][0][0] < 0:
            n: int = abs(-conners[1][0][0] + conners[0][0][0])
            panorama = panorama[t[1] : h_dst + t[1], n:, :]
        else:
            if conners[2][0][0] < conners[3][0][0]:
                panorama = panorama[t[1] : h_dst + t[1], 0 : conners[2][0][0], :]
            else:
                panorama = panorama[t[1] : h_dst + t[1], 0 : conners[3][0][0], :]
        return panorama

    def blending_mask(
        self, height: int, width: int, barrier: int, smoothing_window: int, left_biased: bool = True
    ) -> np.ndarray:
        assert barrier < width
        mask: np.ndarray = np.zeros((height, width))

        offset: int = int(smoothing_window / 2)
        try:
            if left_biased:
                mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                    np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
                )
                mask[:, : barrier - offset] = 1
            else:
                mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                    np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
                )
                mask[:, barrier + offset :] = 1
        except BaseException:
            if left_biased:
                mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset).T, (height, 1))
                mask[:, : barrier - offset] = 1
            else:
                mask[:, barrier - offset : barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset).T, (height, 1))
                mask[:, barrier + offset :] = 1

        return cv2.merge([mask, mask, mask])

    def blending(self, dst_img_rz: np.ndarray, src_img_warped: np.ndarray, dst_w: int, side: str) -> np.ndarray:
        h: int
        w: int
        h, w, _ = dst_img_rz.shape
        smoothing_window: int = int(dst_w / 8)
        barrier: int = dst_w - int(smoothing_window / 2)

        # Create mask
        mask1: np.ndarray = self.blending_mask(h, w, barrier, smoothing_window=smoothing_window, left_biased=True)
        mask2: np.ndarray = self.blending_mask(h, w, barrier, smoothing_window=smoothing_window, left_biased=False)

        if side == "left":
            dst_img_rz = cv2.flip(dst_img_rz, 1)
            src_img_warped = cv2.flip(src_img_warped, 1)
            dst_img_rz = dst_img_rz * mask1
            src_img_warped = src_img_warped * mask2
            pano: np.ndarray = src_img_warped + dst_img_rz
            pano = cv2.flip(pano, 1)
        else:
            dst_img_rz = dst_img_rz * mask1
            src_img_warped = src_img_warped * mask2
            pano = src_img_warped + dst_img_rz

        return pano

    def compute_homography(self, src_img: np.ndarray, dst_img: np.ndarray, ransacRep: float = 5.0) -> np.ndarray:
        # Convert to tensor
        src_img_tensor: torch.Tensor = to_tensor(src_img, self.device)
        dst_img_tensor: torch.Tensor = to_tensor(dst_img, self.device)

        # Match features
        matches: Dict[str, np.ndarray] = self.matcher.match_images(src_img_tensor, dst_img_tensor)

        # estimate homography
        res: Dict[str, np.ndarray] = self.homography_estimator.estimate(matches["mkpts0"], matches["mkpts1"])

        return res["H"]

    def warp_two_images(self, src_img: np.ndarray, dst_img: np.ndarray) -> np.ndarray:
        """Warp two images and blend them together"""

        # Compute homography matrix
        H: np.ndarray = self.compute_homography(src_img, dst_img)

        # Get sizes
        src_h: int
        src_w: int
        src_h, src_w = src_img.shape[:2]
        dst_h: int
        dst_w: int
        dst_h, dst_w = dst_img.shape[:2]

        # Conners of src and dst image
        pts1: np.ndarray = np.float32([[0, 0], [0, src_h], [src_w, src_h], [src_w, 0]]).reshape(-1, 1, 2)
        pts2: np.ndarray = np.float32([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]]).reshape(-1, 1, 2)

        # Apply homography on src image
        pts1_: np.ndarray = cv2.perspectiveTransform(pts1, H)
        pts: np.ndarray = np.concatenate((pts1_, pts2), axis=0)

        # Find min max of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [_, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t: List[int] = [-xmin, -ymin]

        # top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
        # otherwise side=right
        # source image is merged to the left side or right side of destination image
        if pts[0][0][0] < 0:
            side: str = "left"
            width_pano: int = dst_w + t[0]
        else:
            width_pano: int = int(pts1_[3][0][0])
            side: str = "right"
        height_pano: int = ymax - ymin

        # Translation  (https://stackoverflow.com/a/20355545)
        Ht: np.ndarray = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        src_img_warped: np.ndarray = cv2.warpPerspective(src_img, Ht.dot(H), (width_pano, height_pano))

        # Generating size of dst_img_rz which has the same size as src_img_warped
        dst_img_rz: np.ndarray = np.zeros((height_pano, width_pano, 3))
        if side == "left":
            dst_img_rz[t[1] : src_h + t[1], t[0] : dst_w + t[0]] = dst_img
        else:
            dst_img_rz[t[1] : src_h + t[1], :dst_w] = dst_img

        # Blending the two images into a panorama
        pano: np.ndarray = self.blending(dst_img_rz, src_img_warped, dst_w, side)

        # Remove black area
        pano = self._remove_black_area(pano, dst_h, pts)
        return pano

    def multi_stitching(self, list_images: List[np.ndarray]) -> np.ndarray:
        """Stitching multiple images into a panorama"""

        # Assuming the list of images is sorted from left to right
        # Split the list of images into two halves
        # Stitch the left half and the right half separately

        n: int = int(len(list_images) / 2 + 0.5)
        left: List[np.ndarray] = list_images[:n]
        right: List[np.ndarray] = list_images[n - 1 :]
        right.reverse()

        # Stitch the left half
        while len(left) > 1:
            dst_img: np.ndarray = left.pop()
            src_img: np.ndarray = left.pop()
            left_pano: np.ndarray = self.warp_two_images(src_img, dst_img)
            left_pano = left_pano.astype("uint8")
            left.append(left_pano)

        # Stitch the right half
        while len(right) > 1:
            dst_img: np.ndarray = right.pop()
            src_img: np.ndarray = right.pop()
            right_pano: np.ndarray = self.warp_two_images(src_img, dst_img)
            right_pano = right_pano.astype("uint8")
            right.append(right_pano)

        # If width_right_pano > width_left_pano
        # Select right_pano as destination.
        # Otherwise is left_pano
        if right_pano.shape[1] >= left_pano.shape[1]:
            panorama: np.ndarray = self.warp_two_images(left_pano, right_pano)
        else:
            panorama: np.ndarray = self.warp_two_images(right_pano, left_pano)

        return panorama

    def stitch_images(self, input: str, resize: Optional[int] = None) -> np.ndarray:
        """Stitching multiple images into a panorama"""

        # Load images
        list_images: List[np.ndarray] = load_images(input, resize)

        # Stitch images
        panorama: np.ndarray = self.multi_stitching(list_images)

        return panorama

    def __call__(self, input: str, resize: Optional[int] = None) -> np.ndarray:
        return self.stitch_images(input, resize)

    def visualize(self, stitched_image: np.ndarray) -> None:
        """Visualize the stitched image"""
        window_name: str = "Stitched Image"
        stitched_image = stitched_image.astype("uint8")
        cv2.imshow(window_name, stitched_image)

        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
        cv2.destroyAllWindows()


@click.command()
@click.option("--input", required=True, help="Path to input directory")
@click.option("--output", default="assets", help="Path to output directory")
@click.option(
    "--extractor", default="superpoint", type=click.Choice(SUPPORTED_EXTRACTORS), help="Feature extractor to use."
)
@click.option(
    "--matcher", default="superglue_outdoor", type=click.Choice(SUPPORTED_MATCHERS), help="Feature matcher to use."
)
@click.option(
    "--backend", default="opencv", type=click.Choice(["opencv", "pycolmap", "poselib"]), help="Homography backend."
)
@click.option("--resize", type=int, default=0, help="Enter 1 to resize the resolution to 4x lower.")
@click.option("--max_keypoints", default=-1, type=int, help="Maximum number of keypoints to detect.")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.option("--visualize", is_flag=True, help="Visualize the stitched image.")
@click.help_option("-h", "--help", help="Show this message and exit.")
@suppress_warnings()
def stitch(
    input: str,
    output: Optional[str],
    extractor: str,
    matcher: str,
    backend: str,
    resize: int,
    max_keypoints: int,
    force_cpu: bool,
    visualize: bool,
) -> None:
    """Stitch multiple images into a panorama"""
    logger.info("Stitching images...")

    # Device
    device: str = detect_device(force_cpu)

    # Create panorama
    stitcher: ImageStitcher = ImageStitcher(
        extractor=extractor,
        matcher=matcher,
        backend=backend,
        max_keypoints=max_keypoints,
        device=device,
    )

    # Stitch images
    panorama: np.ndarray = stitcher(input, resize)

    # Save the result
    output_path: str = os.path.join(output, "panorama.jpg") if output else "panorama.jpg"
    cv2.imwrite(output_path, panorama)
    logger.info(f"Panorama saved to {output_path}")

    if visualize:
        stitcher.visualize(panorama)

    logger.success("Done!")


if __name__ == "__main__":
    stitch()
