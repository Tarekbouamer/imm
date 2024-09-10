import numpy as np
import torch
from torchvision import transforms as tfn

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
tfn_image_net = tfn.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

tfn_grayscale = tfn.Grayscale()


class ToTensor:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, np_img):
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)

        np_img = np_img.transpose((2, 0, 1))
        tensor_img = torch.from_numpy(np_img).to(dtype=self.dtype)
        return tensor_img


def make_transforms(normalize="imagenet", gray=False):
    # to tensor
    tfl = []

    # grayscale
    if gray:
        tfl += [tfn_grayscale]

    # normalize
    if normalize == "imagenet":
        tfl += [tfn_image_net]
    else:
        pass

    # return transforms
    return tfn.Compose(tfl)
