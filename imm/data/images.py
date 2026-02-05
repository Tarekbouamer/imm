from pathlib import Path

from loguru import logger
from torch.utils.data import Dataset

from imm.utils.io import find_images, load_image_tensor


def relative_path(path, root):
    return path.relative_to(root).as_posix()


class ImagesFromList(Dataset):
    """Dataset for loading images from a directory."""

    def __init__(self, root: str, max_img_size: int = -1):
        # root
        self.root = root

        # collect image paths
        self.images_paths = sorted(find_images(root))

        # image names
        self.names = [relative_path(img_path, root)
                      for img_path in self.images_paths]

        #
        self.max_img_size = max_img_size

        logger.info("ImagesFromList:")
        logger.info(f"      Images: {len(self.images_paths)} in {root}")
        logger.info(f"      Max image size: {max_img_size}")

    def __len__(self):
        return len(self.images_paths)

    def get_names(self):
        return self.names

    def __getitem__(self, item):
        out = {}

        img_path = self.images_paths[item]
        img_name = self.names[item]

        # load image
        data = load_image_tensor(img_path, resize=self.max_img_size)
        image = data[0]
        image_cv = data[1]
        scale = data[3]
        original_size = data[4]

        # dict
        out["image"] = image
        out["name"] = img_name
        out["original_size"] = original_size
        out["scale"] = scale

        return out

    def __repr__(self):
        return (
            f"ImagesFromList(root={self.root}, num_images={len(self.images_paths)}, max_img_size={self.max_img_size})"
        )
