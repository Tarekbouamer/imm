from typing import Dict, List, Literal, NamedTuple, Tuple, Union

import numpy as np


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
        h, w = image.shape[:2]
        return cls(np.array([w, h, w, w, w // 2, h // 2, 0.0, 0.0]), "PINHOLE")

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


    def to_dict(self) -> Dict:
        return {
            "model": self.camera_model,
            "width": self.width,
            "height": self.height,
            "params": self.params.tolist(),
        }

    def __repr__(self):
        return f"Camera({self.camera_model}, {self.params})"
