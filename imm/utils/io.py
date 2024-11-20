from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from loguru import logger

INTER_MODES: dict = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
}


def find_images(root: Path, output_file: Optional[Path] = None) -> List[Path]:
    """Find images in the specified path and subdirectories"""

    # Search for image files in the specified path
    image_files = [
        file
        for ext in {"jpg", "jpeg", "png"}
        for file in list(root.rglob(f"*.{ext}")) + list(root.rglob(f"*.{ext.upper()}"))
    ]

    #
    if not image_files:
        logger.warning(f"No images found in {root}")
        return []

    if output_file is None:
        output_file = root / "image_list.txt"

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f:
            for image_file in image_files:
                relative_path = image_file.relative_to(root)
                f.write(f"{relative_path}\n")
        logger.info(f"Image list written to {output_file}")
    except Exception as e:
        logger.error(f"Error writing to {output_file}: {e}")

    return image_files


def pad_image_bottom_right(img, pad_size, ret_mask=False):
    """Pad image to padding_size with zeros"""

    assert isinstance(pad_size, int) and pad_size >= max(img.shape[-2:]), f"{pad_size} < {max(img.shape[-2:])}"

    # mask
    mask = None

    if img.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=img.dtype)
        padded[: img.shape[0], : img.shape[1]] = img
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[: img.shape[0], : img.shape[1]] = True
    elif img.ndim == 3:
        padded = np.zeros((img.shape[0], pad_size, pad_size), dtype=img.dtype)
        padded[:, : img.shape[1], : img.shape[2]] = img

        if ret_mask:
            mask = np.zeros((img.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, : img.shape[1], : img.shape[2]] = True
    else:
        raise ValueError("img.ndim must be 2 or 3")

    return padded, mask


def get_target_wh(w, h, target=None, resize_fn=max):
    """Get target width and height"""

    scale = target / resize_fn(h, w)
    w_new, h_new = int(round(w * scale)), int(round(h * scale))
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    """Get divisible width and height"""

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h

    return w_new, h_new


def read_image(path, gray=False):
    """Read image"""

    # try to read image
    try:
        image = cv2.imread(path)
    except Exception:
        raise ValueError(f"Can't read image from {path}")

    # convert to gray
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get size
    size = (image.shape[1], image.shape[0])

    return image, size


def load_image(
    image_path, resize=None, resize_fn="max", gray=False, df=1, padding=False, interp=cv2.INTER_AREA
):  # read image
    # read image
    cv_img, original_size = read_image(image_path, gray)
    mask = None

    # get new size
    if resize_fn != "None" and resize is not None:
        # max
        if resize_fn == "max":
            w_new, h_new = get_target_wh(original_size[0], original_size[1], resize, resize_fn=max)
        elif resize_fn == "min":
            w_new, h_new = get_target_wh(original_size[0], original_size[1], resize, resize_fn=min)
        else:
            w_new, h_new = resize
    else:
        w_new, h_new = original_size

    # devisible by df
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    # resize
    if w_new != original_size[0] or h_new != original_size[1]:
        cv_img = cv2.resize(cv_img, (w_new, h_new), interpolation=interp)

    # scale
    scale = np.array([original_size[0] / w_new, original_size[1] / h_new], dtype=np.float32)

    return cv_img, mask, scale, original_size


def toImageTensor(cv_img, padding=False):
    if cv_img.ndim < 3:
        cv_img = np.expand_dims(cv_img, axis=-1)

    #
    w_new, h_new, _ = cv_img.shape

    # -> CHW
    cv_img = cv_img.transpose((2, 0, 1))

    # padding
    if padding:
        pad_to = max(h_new, w_new)
        cv_img, mask = pad_image_bottom_right(cv_img, pad_to, ret_mask=True)
    else:
        mask = None

    # to Tensor
    tensor_img = torch.from_numpy(cv_img).to(dtype=torch.float32)

    if mask is not None:
        mask = torch.from_numpy(mask).to(dtype=torch.bool)

    # normalize
    tensor_img = tensor_img / 255.0

    return tensor_img, mask


def load_image_tensor(
    image_path,
    resize=None,
    resize_fn="max",
    gray=False,
    df=1,
    padding=False,
    interp=cv2.INTER_AREA,
    batched=False,
):
    # load image
    cv_img, mask, scale, original_size = load_image(
        image_path,
        resize=resize,
        resize_fn=resize_fn,
        gray=gray,
        df=df,
        padding=padding,
        interp=interp,
    )

    # -> tensor
    tensor_img, mask = toImageTensor(cv_img, padding=padding)
    scale = torch.tensor(scale, dtype=torch.float32)
    original_size = torch.tensor(original_size, dtype=torch.float32)

    # batched
    if batched:
        tensor_img = tensor_img.unsqueeze(0)
        scale = scale.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    #
    return tensor_img, cv_img, mask, scale, original_size
