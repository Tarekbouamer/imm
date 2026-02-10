import enum
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import Normalize

from imm.utils.io import read_image


class VizType(enum.Enum):
    KEYPOINTS = 1
    MATCH = 2


class Viz2D:
    """Base class for 2D visualization."""

    def __init__(self):
        self.results = None

    def draw_image(self, image: Union[np.ndarray, str, Path], title: str = "Image", show_image: bool = True):
        """Draw single image.

        Args:
            image: Image path or numpy array
            title: Image title
            show_image: Whether to display the image
        """
        image, _ = read_image(image)
        self.results = image

        if show_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")
            plt.show()

    def save(self, file_path: str):
        """Save visualization result to file.

        Args:
            file_path: Output file path
        """
        if self.results is not None:
            # Results are already in BGR format (OpenCV default)
            cv2.imwrite(file_path, self.results)
        else:
            logger.warning("No image to save.")

    def draw_composite_image(
        self,
        image1: Union[np.ndarray, str, Path],
        image2: Union[np.ndarray, str, Path],
        offset: int = 10,
    ) -> np.ndarray:
        """Draw two images side by side.

        Args:
            image1: First image
            image2: Second image
            offset: Pixel offset between images

        Returns:
            Composite image with both images side by side
        """
        image1, _ = read_image(image1)
        image2, _ = read_image(image2)

        h1, w1, _ = image1.shape
        h2, w2, _ = image2.shape

        max_height = max(h1, h2)
        total_width = w1 + w2 + offset

        composite_image = np.ones(
            (max_height, total_width, 3), dtype=np.uint8) * 255

        composite_image[:h1, :w1, :] = image1
        composite_image[:h2, w1 + offset: w1 + offset + w2, :] = image2

        self.results = composite_image  # Store the drawn result
        return composite_image


class KeypointVisualizer(Viz2D):
    """Visualizer for keypoints and features."""

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
        """Draw keypoints on image.

        Args:
            image: Image path or numpy array
            keypoints: Keypoint coordinates (N, 2)
            scores: Optional keypoint scores for color mapping
            title: Image title
            default_color: Color for keypoints when scores are not provided
            show_image: Whether to display the image
        """
        image_with_keypoints, _ = read_image(image)
        image_with_keypoints = image_with_keypoints.copy()

        if scores is not None:
            if len(keypoints) != len(scores):
                raise ValueError(
                    "Keypoints and scores must have the same length")
            cmap = plt.get_cmap("coolwarm")
            norm = Normalize(0, 1)
            for kp, score in zip(keypoints, scores):
                color = cmap(norm(score))[:3]
                color = (
                    int(color[0] * 255),
                    int(color[1] * 255),
                    int(color[2] * 255),
                )
                cv2.circle(image_with_keypoints,
                           (int(kp[0]), int(kp[1])), 3, color, -1)
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
    """Visualizer for keypoint matches between images."""

    def __init__(self):
        super().__init__()

    def draw_matches(
        self,
        image1: Union[np.ndarray, str, Path],
        image2: Union[np.ndarray, str, Path],
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        matches: Optional[np.ndarray] = None,
        mscores: Optional[np.ndarray] = None,
        color_inliers: Tuple[int, int, int] = (0, 0, 255),
        color_outliers: Tuple[int, int, int] = (255, 0, 0),
        color_lines: Tuple[int, int, int] = (0, 255, 0),
        offset: int = 10,
        title: str = "Matches",
        show_image: bool = True,
    ):
        """Draw matches between two images.

        Args:
            image1: First image
            image2: Second image
            kpts0: All keypoints in first image
            kpts1: All keypoints in second image
            mkpts0: Matched keypoints in first image
            mkpts1: Matched keypoints in second image
            matches: Match indices (optional)
            mscores: Match confidence scores (optional)
            color_inliers: Color for matched keypoints
            color_outliers: Color for unmatched keypoints
            color_lines: Color for match lines
            offset: Pixel offset between images
            title: Visualization title
            show_image: Whether to display the image

        Returns:
            Composite image with matches drawn
        """
        image1_rgb, _ = read_image(image1)
        image2_rgb, _ = read_image(image2)
        composite_image = self.draw_composite_image(
            image1_rgb, image2_rgb, offset=offset)

        # Draw all keypoints
        for kp in kpts0:
            cv2.circle(composite_image, (int(kp[0]), int(
                kp[1])), 3, color_outliers, -1)
        for kp in kpts1:
            cv2.circle(
                composite_image,
                (int(kp[0]) + image1_rgb.shape[1] + offset, int(kp[1])),
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
                (
                    int(mkpts1[i][0]) + image1_rgb.shape[1] + offset,
                    int(mkpts1[i][1]),
                ),
                3,
                color_inliers,
                -1,
            )

        if mscores is not None:
            valid = np.where(matches != -1)[0]
            mscores = mscores[valid]

            for i, score in enumerate(mscores):
                kp0 = mkpts0[i]
                kp1_offset = (
                    mkpts1[i][0] + image1_rgb.shape[1] + offset,
                    mkpts1[i][1],
                )
                color = tuple(int(c * score) for c in color_lines)
                cv2.line(
                    composite_image,
                    (int(kp0[0]), int(kp0[1])),
                    (int(kp1_offset[0]), int(kp1_offset[1])),
                    color,
                    1,
                )
        # title with number of keypoints of image 0 and image 1 and matches
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
            cv2.circle(composite_image,
                       (int(pt0[0]), int(pt0[1])), 5, (0, 0, 255), -1)

            pt1_h = np.array([pt0[0], pt0[1], 1]).reshape(3, 1)
            epip_line = F @ pt1_h
            a, b, c = epip_line.flatten()

            if b != 0:  # Avoid division by zero
                x0, x1 = 0, w1 - 1
                y0 = int((-a * x0 - c) / b)
                y1 = int((-a * x1 - c) / b)
                cv2.line(composite_image, (w0 + x0, y0),
                         (w0 + x1, y1), (0, 255, 0), 1)

            cv2.circle(composite_image,
                       (w0 + int(pt1[0]), int(pt1[1])), 5, (255, 0, 0), -1)

        self.draw_image(composite_image, title, show_image=show_image)
        return composite_image


class HomographyVisualizer(Viz2D):
    """Visualizer for homography transformations."""

    def __init__(self):
        super().__init__()

    def draw_homography_warp(
        self,
        image1: Union[np.ndarray, str, Path],
        image2: Union[np.ndarray, str, Path],
        H: np.ndarray,
        alpha: float = 0.5,
        title: str = "Homography Warp",
        show_image: bool = True,
    ) -> np.ndarray:
        """Warp image1 to image2 using homography and blend for visualization.

        Args:
            image1: Source image to warp
            image2: Target image for alignment
            H: 3x3 homography matrix
            alpha: Blending factor (0.0 = only warped, 1.0 = only target)
            title: Visualization title
            show_image: Whether to display the image

        Returns:
            Blended warped image
        """
        image1, _ = read_image(image1)
        image2, _ = read_image(image2)

        # Get target dimensions
        h, w = image2.shape[:2]

        # Warp image1 to align with image2
        warped = cv2.warpPerspective(image1, H, (w, h))

        # Blend warped image with target
        blended = cv2.addWeighted(warped, 1 - alpha, image2, alpha, 0)

        self.results = blended

        if show_image:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            plt.title("Source Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title("Warped Source")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            plt.title(f"{title} (α={alpha})")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        return blended


class EpipolarVisualizer(Viz2D):
    """Visualizer for epipolar geometry from relative pose."""

    def __init__(self):
        super().__init__()

    def draw_epipolar_lines(
        self,
        image0: Union[np.ndarray, str, Path],
        image1: Union[np.ndarray, str, Path],
        pts0: np.ndarray,
        pts1: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        camera0,
        camera1,
        inliers: Optional[np.ndarray] = None,
        n_lines: int = 4,
        title: str = "Epipolar Geometry",
        show_image: bool = True,
    ) -> np.ndarray:
        """Draw epipolar lines showing geometric constraints from relative pose.

        Args:
            image0: First image
            image1: Second image
            pts0: Keypoints in first image (N, 2)
            pts1: Keypoints in second image (N, 2)
            R: Rotation matrix (3x3) from camera0 to camera1
            t: Translation vector (3x1) from camera0 to camera1
            camera0: Camera intrinsics for first image
            camera1: Camera intrinsics for second image
            inliers: Boolean mask of inlier matches
            n_lines: Number of epipolar lines to draw
            title: Visualization title
            show_image: Whether to display the image

        Returns:
            Composite image with epipolar lines
        """
        image0, _ = read_image(image0)
        image1, _ = read_image(image1)

        # Compute fundamental matrix from essential matrix
        # E = [t]_x @ R, F = K1^-T @ E @ K0^-1
        E = self._skew(t.reshape(3)) @ R
        K0 = camera0.K()
        K1 = camera1.K()
        F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)

        # Compute rotation angle and translation
        angle_deg = self._rotation_angle(R)
        t_norm = t.reshape(3) / (np.linalg.norm(t) + 1e-8)

        # Select points to visualize
        if inliers is not None:
            inlier_idx = np.where(inliers)[0]
        else:
            inlier_idx = np.arange(len(pts0))

        # Create composite image
        composite = self.draw_composite_image(image0, image1, offset=10)
        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]

        # Select n_lines random inlier points to visualize
        if len(inlier_idx) > 0:
            # Pick n_lines random inliers (or all if fewer available)
            n_points = min(n_lines, len(inlier_idx))
            show_indices = np.random.choice(
                inlier_idx, n_points, replace=False)

            # Generate distinct colors for each line
            colors = self._generate_colors(n_points)

            for i, show_idx in enumerate(show_indices):
                pt0 = pts0[show_idx]
                pt1 = pts1[show_idx]
                color = colors[i]

                # Compute epipolar line in image1
                pt0_h = np.array([pt0[0], pt0[1], 1.0])
                line1 = F @ pt0_h

                # Draw epipolar line on image1
                self._draw_epiline(composite, line1, w0 + 10, w1, h1, color)

                # Draw the point in image0
                cv2.circle(composite, (int(pt0[0]), int(pt0[1])), 8, color, -1)
                cv2.circle(composite, (int(pt0[0]), int(
                    pt0[1])), 10, (255, 255, 255), 2)

                # Draw the corresponding matched point in image1
                pt1_x = int(pt1[0]) + w0 + 10
                pt1_y = int(pt1[1])
                cv2.circle(composite, (pt1_x, pt1_y), 8, color, -1)
                cv2.circle(composite, (pt1_x, pt1_y), 10, (255, 255, 255), 2)

        # Add text overlay with pose information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        y_offset = 30

        info_lines = [
            f"Rotation: {angle_deg:.2f} deg",
            f"Translation: [{t_norm[0]:.3f}, {t_norm[1]:.3f}, {t_norm[2]:.3f}]",
            f"Inliers: {len(inlier_idx)}/{len(pts0)}"
        ]

        for i, line_text in enumerate(info_lines):
            y_pos = y_offset + i * 30
            cv2.putText(composite, line_text, (10, y_pos), font,
                        font_scale, (0, 255, 0), thickness)
            cv2.putText(composite, line_text, (10, y_pos),
                        font, font_scale, (0, 0, 0), 1)

        self.results = composite

        if show_image:
            self.draw_image(composite, title, show_image=True)

        return composite

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def _rotation_angle(R: np.ndarray) -> float:
        """Compute rotation angle in degrees from rotation matrix."""
        trace = np.trace(R)
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        return np.degrees(angle_rad)

    @staticmethod
    def _generate_colors(n: int) -> list:
        """Generate n distinct colors using HSV color space."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)  # Distribute hues evenly
            color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            colors.append(tuple(int(c) for c in color_bgr[0][0]))
        return colors

    @staticmethod
    def _draw_epiline(img: np.ndarray, line: np.ndarray, x_offset: int, w: int, h: int, color: Tuple[int, int, int]):
        """Draw epipolar line on image, clipped to image boundaries."""
        a, b, c = line

        # Compute intersections with image boundaries
        points = []

        # Left boundary (x = 0)
        if abs(b) > 1e-6:
            y = (-c) / b
            if 0 <= y < h:
                points.append((0, int(y)))

        # Right boundary (x = w-1)
        if abs(b) > 1e-6:
            y = (-a * (w - 1) - c) / b
            if 0 <= y < h:
                points.append((w - 1, int(y)))

        # Top boundary (y = 0)
        if abs(a) > 1e-6:
            x = (-c) / a
            if 0 <= x < w:
                points.append((int(x), 0))

        # Bottom boundary (y = h-1)
        if abs(a) > 1e-6:
            x = (-b * (h - 1) - c) / a
            if 0 <= x < w:
                points.append((int(x), h - 1))

        # Draw line between first two valid intersection points
        if len(points) >= 2:
            pt1 = (points[0][0] + x_offset, points[0][1])
            pt2 = (points[1][0] + x_offset, points[1][1])
            cv2.line(img, pt1, pt2, color, 2)
