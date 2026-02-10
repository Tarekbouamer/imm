from typing import Dict, NamedTuple, Tuple, Union

import numpy as np

from imm.estimators._conversions import distort_points, to_homogeneous


class Camera:
    def __init__(self, params: np.ndarray, camera_model: str = "PINHOLE", eps: float = 1e-6):

        self.params = np.asarray(params, dtype=np.float32).reshape(-1)

        assert len(
            self.params) >= 6, f"params must have at least 6 elements (width, height, fx, fy, cx, cy), got {len(self.params)}"

        self.camera_model = camera_model
        self.eps = float(eps)

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]) -> "Camera":
        """Create a Camera object from COLMAP camera parameters."""
        if isinstance(camera, tuple):
            camera = camera._asdict()

        camera_model = camera["model"]
        params = np.asarray(camera["params"], dtype=np.float32).reshape(-1)

        if camera_model in ["OPENCV", "PINHOLE", "RADIAL"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif camera_model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if camera_model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(camera_model)

        data = np.r_[camera["width"], camera["height"],
                     fx, fy, cx, cy, params].astype(np.float32)
        return cls(data, camera_model)

    @classmethod
    def from_image(cls, image: np.ndarray) -> "Camera":
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        fx = fy = 0.5 * max(w, h)
        return cls(np.array([w, h, fx, fy, cx, cy, 0.0, 0.0], dtype=np.float32), "PINHOLE")

    @classmethod
    def from_dict(cls, data: Dict) -> "Camera":
        """Create a Camera object from a dictionary.

        Args:
            data: Dict with 'model', 'width', 'height', and 'params'.
                  Compatible with to_dict() output format.

        Returns:
            Camera: Camera instance.
        """
        model = data["model"]
        width = float(data["width"])
        height = float(data["height"])
        params = np.asarray(data["params"], dtype=np.float32).reshape(-1)

        assert len(
            params) >= 4, "requires params at least [fx, fy, cx, cy]"

        full = np.r_[width, height, params].astype(np.float32)
        return cls(full, model)

    @classmethod
    def from_K(cls, K: np.ndarray, width: int = None, height: int = None) -> "Camera":
        K = np.asarray(K, dtype=np.float32)
        if width is None:
            width = int(round(float(K[0, 2]) * 2.0))
        if height is None:
            height = int(round(float(K[1, 2]) * 2.0))

        assert width > 0 and height > 0, "Invalid width or height inferred from K."

        return cls(np.array([width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0],  dtype=np.float32), "PINHOLE")

    @property
    def model(self) -> str:
        return self.camera_model

    @property
    def width(self) -> int:
        return int(self.params[0])

    @property
    def height(self) -> int:
        return int(self.params[1])

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def fx(self) -> float:
        return self.params[2]

    @property
    def fy(self) -> float:
        return self.params[3]

    @property
    def cx(self) -> float:
        return self.params[4]

    @property
    def cy(self) -> float:
        return self.params[5]

    @property
    def dist(self) -> np.ndarray:
        return self.params[6:]

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0., self.cx], [0., self.fy, self.cy], [0., 0., 1.]], dtype=np.float32)

    def scale(self, factor: float) -> "Camera":
        w, h = float(self.params[0]), float(self.params[1])

        scaled_params = np.array(
            [
                w * factor,
                h * factor,
                self.fx * factor,
                self.fy * factor,
                self.cx * factor,
                self.cy * factor,
                *self.dist,
            ],
            dtype=np.float32,
        )
        return self.__class__(scaled_params, self.camera_model)

    def in_image(self, p2d: np.ndarray) -> np.ndarray:
        """Check if 2D points are within the image boundaries."""
        p2d = np.asarray(p2d)
        if p2d.shape[-1] != 2:
            raise ValueError(f"p2d must have last dim == 2, got {p2d.shape}")

        # size: (2,) -> (1,2) for broadcasting
        size = np.asarray(self.size)[None, :]

        valid = np.all((p2d >= 0) & (p2d <= (size - 1)), axis=-1)
        return valid

    def normalize(self, p2d: np.ndarray) -> np.ndarray:
        """Convert 2D pixel coordinates to normalized camera coordinates."""
        p2d = np.asarray(p2d)
        if p2d.ndim != 2 or p2d.shape[1] != 2:
            raise ValueError(f"p2d must have shape (N, 2), got {p2d.shape}")
        return np.stack(
            [
                (p2d[:, 0] - self.cx) / self.fx,
                (p2d[:, 1] - self.cy) / self.fy,
            ],
            axis=1,
        )

    def denormalize(self, p2d: np.ndarray) -> np.ndarray:
        """Convert normalized 2D camera coordinates to pixel coordinates."""
        p2d = np.asarray(p2d)
        return np.stack(
            [
                p2d[:, 0] * self.fx + self.cx,
                p2d[:, 1] * self.fy + self.cy,
            ],
            axis=1,
        )

    def project(self, p3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points into the camera plane and check for visibility."""
        p3d = np.asarray(p3d)
        if p3d.ndim != 2 or p3d.shape[1] != 3:
            raise ValueError(f"p3d must have shape (N, 3), got {p3d.shape}")

        z = p3d[..., -1]
        valid = z > self.eps

        z_safe = np.maximum(z, self.eps)
        p2d = p3d[..., :-1] / z_safe[..., None]

        return p2d, valid

    def distort(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Distort normalized 2D coordinates and return a validity mask."""
        pts = np.asarray(pts)
        if pts.shape[-1] != 2:
            raise ValueError(f"pts must have last dim == 2, got {pts.shape}")

        return distort_points(pts, self.dist)

    def image2cam(self, p2d: np.ndarray) -> np.ndarray:
        """Convert 2D pixel coordinates to 3D camera points with z=1."""
        assert self.params.size > 0
        p2d = np.asarray(p2d)
        p2d = self.normalize(p2d)
        return to_homogeneous(p2d)

    def cam2image(self, p3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.distort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    def to_dict(self) -> Dict:
        return {
            "model": self.camera_model,
            "width": self.width,
            "height": self.height,
            "params": self.params[2:].tolist(),  # Format B only
        }

    def __repr__(self):
        return f"Camera({self.camera_model}, {self.params})"
