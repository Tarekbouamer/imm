from typing import Dict, NamedTuple, Tuple, Union

import numpy as np

from ._conversions import convert_points_to_homogeneous


class Camera:
    def __init__(self, params: np.ndarray, camera_model: str = "PINHOLE"):
        assert isinstance(params, np.ndarray), f"params should be a numpy array, got {type(params)}"
        self.params = np.array(params)
        self.camera_model = camera_model

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]):
        if isinstance(camera, tuple):
            camera = camera._asdict()

        camera_model = camera["model"]
        params = camera["params"]

        if camera_model in ["OPENCV", "PINHOLE", "RADIAL"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif camera_model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if camera_model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(camera_model)

        data = np.r_[camera["width"], camera["height"], fx, fy, cx, cy, params]
        return cls(data, camera_model)

    @classmethod
    def from_image(cls, image: np.ndarray):
        print("image shape", image.shape)
        h, w = image.shape[:2]
        return cls(np.array([w, h, w, h, w // 2, h // 2, 0.0, 0.0]), "PINHOLE")

    @classmethod
    def from_dict(cls, data: Dict):
        """Create a Camera object from a dictionary."""
        # {model: str, width: int, height: int, params: List[float]}
        return cls(np.array(data["params"]), data["model"])

    @classmethod
    def from_K(cls, K: np.ndarray, width: int = None, height: int = None):
        if width is None:
            width = K[0, 2] * 2
        if height is None:
            height = K[1, 2] * 2

        return cls(np.array([width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0]), "PINHOLE")

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

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def scale(self, factor: float):
        scaled_params = np.array(
            [
                self.width * factor,
                self.height * factor,
                self.fx * factor,
                self.fy * factor,
                self.cx * factor,
                self.cy * factor,
                *self.dist,
            ]
        )
        return self.__class__(scaled_params, self.camera_model)

    def todict(self) -> Dict:
        """
        camera_dict = {
                'model': COLMAP_CAMERA_MODEL_NAME_OR_ID,
                'width': IMAGE_WIDTH,
                'height': IMAGE_HEIGHT,
                'params': EXTRA_CAMERA_PARAMETERS_LIST}
        """
        return {
            "model": self.camera_model,
            "width": self.width,
            "height": self.height,
            "params": self.params[2:6].tolist(),
        }

    def normalize(self, p2d: np.ndarray) -> np.ndarray:
        """Normalize pixel coordinates."""
        p2d = p2d.copy()
        p2d[:, 0] = (p2d[:, 0] - self.cx) / self.fx
        p2d[:, 1] = (p2d[:, 1] - self.cy) / self.fy
        return p2d

    def denormalize(self, p2d: np.ndarray) -> np.ndarray:
        """Denormalize pixel coordinates."""
        p2d = p2d.copy()
        p2d[:, 0] = p2d[:, 0] * self.fx + self.cx
        p2d[:, 1] = p2d[:, 1] * self.fy + self.cy
        return p2d

    def image2camera(self, p2d: np.ndarray) -> np.ndarray:
        """Convert 2D pixel coordinates to normalized camera coordinates."""

        p2d = self.normalize(p2d)
        return convert_points_to_homogeneous(p2d)

    def project(self, p3d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Project 3D points to camera coordinates."""

        assert p3d.shape[1] == 3, f"Expected 3D points, got {p3d.shape[1]}"

        z = p3d[:, 2]
        mask = z > eps
        p2d = p3d[mask, :2] / z[mask, None]

        return p2d, mask

    def camera2image(self, p2d: np.ndarray) -> np.ndarray:
        """Convert 2D camera coordinates to pixel coordinates."""
        p2d = self.denormalize(p2d)
        return p2d

    # to world coordinates
    def image2world(self, p2d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Convert 2D pixel coordinates to world coordinates.

        Args:
            p2d: 2D pixel coordinates (N, 2)
            R: Rotation matrix (3, 3)
            t: Translation vector (3,) or (3, 1)

        """
        p3d = self.image2camera(p2d)
        print("p3d", p3d.shape)
        print("R", R.shape)
        print("t", t.shape)
        p3d_r = R @ p3d.T
        print("p3d_r", p3d_r.shape)
        p3d_t = p3d_r + t
        print("p3d_t", p3d_t.shape)
        return p3d_t.T

    def __repr__(self):
        return f"Camera({self.camera_model}, {self.params})"
