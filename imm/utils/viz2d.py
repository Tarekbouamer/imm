import enum
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# backend tk
plt.switch_backend("tkagg")


class VizType(enum.Enum):
    KEYPOINTS = 1
    MATCH = 2


def load_image(image: Union[np.ndarray, str, Path]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"Failed to load image from path: {image}")
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError("Input should be a file path or a numpy.ndarray")
    return image


class Viz2D:
    def __init__(self):
        self.results = None  # Class member to hold the drawn results

    def ensure_rgb(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        image = load_image(image)
        if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def draw_image(self, image: Union[np.ndarray, str, Path], title: str = "Image", show_image: bool = True):
        image = self.ensure_rgb(image)
        self.results = image

        if show_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")
            plt.show()

    def save(self, file_path: str):
        if self.results is not None:
            cv2.imwrite(file_path, self.results)
        else:
            logger.warning("No image to save.")

    def draw_composite_image(
        self,
        image0: Union[np.ndarray, str, Path],
        image1: Union[np.ndarray, str, Path],
        offset: int = 10,
    ) -> np.ndarray:
        image0 = self.ensure_rgb(image0)
        image1 = self.ensure_rgb(image1)

        h0, w0, _ = image0.shape
        h1, w1, _ = image1.shape

        max_height = max(h0, h1)
        total_width = w0 + w1 + offset

        composite_image = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

        composite_image[:h0, :w0, :] = image0
        composite_image[:h1, w0 + offset : w0 + offset + w1, :] = image1

        self.results = composite_image  # Store the drawn result
        return composite_image


class KeypointVisualizer(Viz2D):
    def __init__(self):
        super().__init__()

    def draw_keypoints(
        self,
        image: Union[np.ndarray, str, Path],
        keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        title: str = "kpts",
        default_color: Tuple[int, int, int] = (0, 0, 255),
        show_image: bool = True,
    ):
        image_with_keypoints = self.ensure_rgb(image).copy()

        if scores is not None:
            if len(keypoints) != len(scores):
                raise ValueError("Keypoints and scores must have the same length")
            cmap = plt.get_cmap("coolwarm")
            norm = plt.Normalize(0, 1)
            for kp, score in zip(keypoints, scores):
                color = cmap(norm(score))[:3]
                color = (
                    int(color[0] * 255),
                    int(color[1] * 255),
                    int(color[2] * 255),
                )
                cv2.circle(image_with_keypoints, (int(kp[0]), int(kp[1])), 3, color, -1)
        else:
            for kp in keypoints:
                cv2.circle(
                    image_with_keypoints,
                    (int(kp[0]), int(kp[1])),
                    3,
                    default_color,
                    -1,
                )

        # title with number of keypoints
        title = f"{title} ({len(keypoints)} keypoints)"
        self.draw_image(image_with_keypoints, title, show_image=show_image)


class MatchVisualizer(Viz2D):
    def __init__(self):
        super().__init__()

    def draw_matches(
        self,
        image0: Union[np.ndarray, str, Path],
        image1: Union[np.ndarray, str, Path],
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        matches: Optional[np.ndarray] = None,
        mscores: Optional[np.ndarray] = None,
        color_inliers: Optional[Tuple[int, int, int]] = (0, 0, 255),
        color_outliers: Optional[Tuple[int, int, int]] = (255, 0, 0),
        color_lines: Optional[Tuple[int, int, int]] = (0, 255, 0),
        title: str = "Matches",
        show_image: bool = True,
    ) -> np.ndarray:
        image0_rgb = self.ensure_rgb(image0)
        image1_rgb = self.ensure_rgb(image1)
        composite_image = self.draw_composite_image(image0_rgb, image1_rgb)

        # Draw all keypoints
        for kp in kpts0:
            cv2.circle(composite_image, (int(kp[0]), int(kp[1])), 3, color_outliers, -1)
        for kp in kpts1:
            cv2.circle(
                composite_image,
                (int(kp[0]) + image0_rgb.shape[1] + 10, int(kp[1])),
                3,
                color_outliers,
                -1,
            )

        # Draw mutual keypoints
        for i in range(len(mkpts0)):
            cv2.circle(
                composite_image,
                (int(mkpts0[i][0]), int(mkpts0[i][1])),
                3,
                color_inliers,
                -1,
            )
            cv2.circle(
                composite_image,
                (int(mkpts1[i][0]) + image0_rgb.shape[1] + 10, int(mkpts1[i][1])),
                3,
                color_inliers,
                -1,
            )

        # Draw match lines
        if mscores is not None:
            valid = np.where(matches != -1)[0]
            mscores = mscores[valid]
            mscores = mscores / mscores.max()  # Normalize scores to [0, 1]

            for i, score in enumerate(mscores):
                kp0 = mkpts0[i]
                kp1_offset = (mkpts1[i][0] + image0_rgb.shape[1] + 10, mkpts1[i][1])
                color = tuple(int(c * score) for c in color_lines)
                cv2.line(
                    composite_image,
                    (int(kp0[0]), int(kp0[1])),
                    (int(kp1_offset[0]), int(kp1_offset[1])),
                    color,
                    1,
                )

        title = f"{title} (kpts0: {len(kpts0)}, kpts1: {len(kpts1)}, matches: {len(mkpts0)})"
        self.draw_image(composite_image, title, show_image=show_image)
        return composite_image

    def draw_epipolar_line(
        self,
        image0: Union[np.ndarray, str, Path],
        image1: Union[np.ndarray, str, Path],
        F: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        title: str = "Epipolar Line",
        show_image: bool = True,
    ) -> np.ndarray:
        if F.shape != (3, 3):
            raise ValueError("Fundamental matrix F must be a 3x3 matrix.")

        image0_rgb = self.ensure_rgb(image0)
        image1_rgb = self.ensure_rgb(image1)
        composite_image = self.draw_composite_image(image0_rgb, image1_rgb)

        h0, w0 = image0_rgb.shape[:2]
        h1, w1 = image1_rgb.shape[:2]

        for pt0, pt1 in zip(kpts0, kpts1):
            cv2.circle(composite_image, (int(pt0[0]), int(pt0[1])), 5, (0, 0, 255), -1)

            pt1_h = np.array([pt0[0], pt0[1], 1]).reshape(3, 1)
            epip_line = F @ pt1_h
            a, b, c = epip_line.flatten()

            if b != 0:  # Avoid division by zero
                x0, x1 = 0, w1 - 1
                y0 = int((-a * x0 - c) / b)
                y1 = int((-a * x1 - c) / b)
                cv2.line(composite_image, (w0 + x0, y0), (w0 + x1, y1), (0, 255, 0), 1)

            cv2.circle(composite_image, (w0 + int(pt1[0]), int(pt1[1])), 5, (255, 0, 0), -1)

        self.draw_image(composite_image, title, show_image=show_image)
        return composite_image
