import enum
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from imm.estimators._conversions import compute_epipolar_lines

# backend tk
plt.switch_backend("tkagg")


class VizType(enum.Enum):
    KEYPOINTS = 1
    MATCH = 2


def line_to_endpoints(line: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if len(line) == 2:
        pt0, pt1 = line
    else:
        x0, y0, x1, y1 = line
        pt0, pt1 = (x0, y0), (x1, y1)

    pt0 = tuple(int(x) for x in pt0)
    pt1 = tuple(int(x) for x in pt1)

    return pt0, pt1


def load_image(image: Union[np.ndarray, str, Path]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"Failed to load image from path: {image}")
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError(f"Input should be a file path or a numpy.ndarray, got: {type(image)}")
    return image


class Viz2D:
    def __init__(self, image: Union[np.ndarray, str, Path] = None):
        # Load image
        self._image = self.ensure_rgb(image) if image is not None else None

        # Text to display
        self._metadata = {}

    def ensure_rgb(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        image = load_image(image)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def _check_image(self):
        if self._image is None:
            raise ValueError("No image is set.")

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @image.setter
    def image(self, image: Union[np.ndarray, str, Path]):
        self._image = self.ensure_rgb(image)

    def show_opencv(self, title: str = "Image"):
        #
        self._check_image()

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, self._image)

        while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
        cv2.destroyAllWindows()

    def show_matplotlib(self, title: str = "Image"):
        #
        self._check_image()

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

    def show(self, title: str = "Image", use_opencv: bool = False):
        # Extend the title with metadata
        if self._metadata:
            title = f"{title} - {', '.join([f'{k}: {v}' for k, v in self._metadata.items()])}"

        try:
            if use_opencv:
                self.show_opencv(title)
            else:
                self.show_matplotlib(title)
        except Exception as e:
            logger.error(f"Failed to display image: {e}")

        finally:
            if use_opencv:
                cv2.destroyAllWindows()
            else:
                plt.close()

    def save(self, file_path: str = None):
        if self.results is not None:
            cv2.imwrite(file_path, self._image)
        else:
            logger.warning("No image to set to be saved.")

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

        self._image = composite_image
        return composite_image

    def add_point(self, point: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 255), radius: int = 1):
        #
        self._check_image()

        assert len(point) == 2, f"Invalid point: {point}"

        point = tuple(int(x) for x in point)

        cv2.circle(self._image, point, radius, color, -1)

    def add_line(
        self, pt0: Tuple[int, int], pt1: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1
    ):
        #
        self._check_image()

        assert len(pt0) == 2, f"Invalid point: {pt0}"
        assert len(pt1) == 2, f"Invalid point: {pt1}"

        #
        pt0 = tuple(int(x) for x in pt0)
        pt1 = tuple(int(x) for x in pt1)

        cv2.line(self._image, pt0, pt1, color, thickness)

    def add_text(
        self,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 0, 255),
        font_scale: float = 0.5,
        thickness: int = 1,
    ):
        #
        self._check_image()

        cv2.putText(self._image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


class ImageVisualizer(Viz2D):
    def __init__(self, image: Union[np.ndarray, str, Path] = None):
        super().__init__(image)

    def draw_keypoints(
        self,
        keypoints: Union[np.ndarray, list],
        scores: Union[np.ndarray, list] = None,
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 1,
        image: Optional[Union[np.ndarray, str, Path]] = None,
    ):
        # Set image
        if image is not None:
            self._image = self.ensure_rgb(image)

        # Check image
        self._check_image()

        if len(keypoints) < 1:
            raise ValueError("No keypoints to draw.")

        # Color map
        if scores is not None:
            cmap = plt.get_cmap("coolwarm")
            norm = plt.Normalize(0, 1)

            for kpt, score in zip(keypoints, scores):
                # Convert score to color
                color = cmap(norm(score))[:3]
                color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                self.add_point(kpt, color=color, radius=radius)
        # Single color
        else:
            for kpt in keypoints:
                self.add_point(kpt, color=color, radius=radius)

        # Update metadata
        self._metadata["num_keypoints"] = len(keypoints)

    def draw_lines(
        self,
        lines: np.ndarray,
        line_color: Tuple[int, int, int] = (0, 255, 0),
        line_thickness: int = 1,
        end_point_color: Tuple[int, int, int] = (255, 0, 0),
        end_point_radius: int = 2,
        image: Optional[Union[np.ndarray, str, Path]] = None,
    ):
        # Set image
        if image is not None:
            self._image = self.ensure_rgb(image)

        # Check image
        self._check_image()

        if len(lines) < 1:
            raise ValueError("No lines to draw.")

        for line in lines:
            # Get endpoints
            pt0, pt1 = line_to_endpoints(line)

            # Draw line
            self.add_line(pt0, pt1, color=line_color, thickness=line_thickness)

            # Draw keypoints
            self.add_point(pt0, color=end_point_color, radius=end_point_radius)
            self.add_point(pt1, color=end_point_color, radius=end_point_radius)

        # Update metadata
        self._metadata["num_lines"] = len(lines)


def color_generator(num_colors: int, cycle: int = 20):
    cmap = plt.get_cmap("tab20")  # choices: tab10, viridis, plasma, inferno, magma, cividis, tab
    for i in range(num_colors):
        color = cmap(i % cycle)
        yield (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))


class TwoViewVisualizer(Viz2D):
    def __init__(self, image0: Union[np.ndarray, str, Path] = None, image1: Union[np.ndarray, str, Path] = None):
        super().__init__()

        image0 = self.ensure_rgb(image0)
        image1 = self.ensure_rgb(image1)

        # Draw composite image
        self._image = self.draw_composite_image(image0, image1)

        size_t = namedtuple("Size", ["width", "height"])

        self.size0 = size_t(image0.shape[1], image0.shape[0])
        self.size1 = size_t(image1.shape[1], image1.shape[0])

    def draw_point_matches(
        self,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        matches: Optional[np.ndarray] = None,
        mscores: Optional[np.ndarray] = None,
        color_inliers: Optional[Tuple[int, int, int]] = (0, 0, 255),
        color_outliers: Optional[Tuple[int, int, int]] = (255, 0, 0),
        color_lines: Optional[Tuple[int, int, int]] = (0, 255, 0),
        radius: int = 2,
        image0: Optional[Union[np.ndarray, str, Path]] = None,
        image1: Optional[Union[np.ndarray, str, Path]] = None,
    ) -> np.ndarray:
        # Set image
        if image0 is not None:
            image0 = self.ensure_rgb(image0)
            image1 = self.ensure_rgb(image1)
            self._image = self.draw_composite_image(image0, image1)

        # Check image
        self._check_image()

        # Draw keypoints
        for kpt0, kpt1 in zip(kpts0, kpts1):
            self.add_point(kpt0, color=color_outliers, radius=radius)
            self.add_point((kpt1[0] + self.size0.width + 10, kpt1[1]), color=color_outliers, radius=radius)

        # Draw mutual keypoints
        for mkpt0, mkpt1 in zip(mkpts0, mkpts1):
            self.add_point(mkpt0, color=color_inliers, radius=radius)
            self.add_point((mkpt1[0] + self.size0.width + 10, mkpt1[1]), color=color_inliers, radius=radius)

        # Draw match lines
        if matches is not None:
            valid = np.where(matches != -1)[0]
            mscores = mscores[valid]
            mscores = mscores / mscores.max()

            for i, score in enumerate(mscores):
                kp0 = mkpts0[i]
                kp1_offset = (mkpts1[i][0] + self.size0.width + 10, mkpts1[i][1])
                color = tuple(int(c * score) for c in color_lines)
                self.add_line(kp0, kp1_offset, color=color, thickness=1)

        # Update metadata
        self._metadata.update(
            {
                "kpts0": len(kpts0),
                "kpts1": len(kpts1),
                "matches": len(mkpts0),
            }
        )

    def draw_lines_matches(
        self,
        lines0: np.ndarray,
        lines1: np.ndarray,
        mlines0: np.ndarray,
        mlines1: np.ndarray,
        color_outliers: Optional[Tuple[int, int, int]] = (64, 64, 64),
        thickness: int = 4,
        end_point_radius: int = 3,
        image0: Optional[Union[np.ndarray, str, Path]] = None,
        image1: Optional[Union[np.ndarray, str, Path]] = None,
    ) -> np.ndarray:
        # Set image
        if image0 is not None:
            image0 = self.ensure_rgb(image0)
            image1 = self.ensure_rgb(image1)
            self._image = self.draw_composite_image(image0, image1)

        # Check image
        self._check_image()

        # Draw lines
        for line0, line1 in zip(lines0, lines1):
            # Get endpoints
            pt_x, pt_y = line_to_endpoints(line0)

            self.add_line(pt_x, pt_y, color=color_outliers, thickness=2)
            self.add_point(pt_x, color=color_outliers, radius=end_point_radius)
            self.add_point(pt_y, color=color_outliers, radius=end_point_radius)

            # Get endpoints
            pt_x, pt_y = line_to_endpoints(line1)

            # shift the x coordinate of the second line
            pt_x = (pt_x[0] + self.size0.width + 10, pt_x[1])
            pt_y = (pt_y[0] + self.size0.width + 10, pt_y[1])

            self.add_line(pt_x, pt_y, color=color_outliers, thickness=2)
            self.add_point(pt_x, color=color_outliers, radius=end_point_radius)
            self.add_point(pt_y, color=color_outliers, radius=end_point_radius)

        # Draw mutual lines
        cgen = color_generator(len(mlines0))

        # Draw mutual lines
        for mline0, mline1 in zip(mlines0, mlines1):
            # Get color
            color_inliers = next(cgen)

            # line0
            pt_x, pt_y = line_to_endpoints(mline0)
            self.add_line(pt_x, pt_y, color=color_inliers, thickness=thickness)
            self.add_point(pt_x, color=color_inliers, radius=end_point_radius)
            self.add_point(pt_y, color=color_inliers, radius=end_point_radius)

            # line1
            pt_x, pt_y = line_to_endpoints(mline1)

            # shift the x coordinate of the second line
            pt_x = (pt_x[0] + self.size0.width + 10, pt_x[1])
            pt_y = (pt_y[0] + self.size0.width + 10, pt_y[1])

            self.add_line(pt_x, pt_y, color=color_inliers, thickness=thickness)
            self.add_point(pt_x, color=color_inliers, radius=end_point_radius)
            self.add_point(pt_y, color=color_inliers, radius=end_point_radius)

        # Update metadata
        self._metadata.update(
            {
                "lines0": len(lines0),
                "lines1": len(lines1),
                "lines_matches": len(mlines0),
            }
        )

    def draw_epipolar_lines(
        self,
        F: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        color_inliers: Optional[Tuple[int, int, int]] = (0, 0, 255),
        radius: int = 3,
        line_color: Optional[Tuple[int, int, int]] = (128, 0, 128),  # purple
        line_thickness: int = 1,
    ) -> np.ndarray:
        # Check Fundamental matrix
        if F.shape != (3, 3):
            raise ValueError("Fundamental matrix F must be a 3x3 matrix.")

        # Check image
        self._check_image()

        w0, w1 = self.size0.width, self.size1.width

        # compute epip lines
        epip_lines = compute_epipolar_lines(F, kpts0)
        epip_lines = epip_lines.T

        for pt0, pt1, epip_line in zip(kpts0, kpts1, epip_lines):
            # pt0
            self.add_point(pt0, color=color_inliers, radius=radius)

            # pt1
            self.add_point((pt1[0] + w0 + 10, pt1[1]), color=color_inliers, radius=radius)

            # epipolar line
            a, b, c = epip_line.flatten()

            assert b != 0, "b should not be zero"

            x0, x1 = 0, w1 - 1
            y0 = int((-a * x0 - c) / b)
            y1 = int((-a * x1 - c) / b)
            cv2.line(self._image, (w0 + x0, y0), (w0 + x1, y1), line_color, line_thickness)
