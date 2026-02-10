
from pathlib import Path

from loguru import logger
from torch.utils.data import Dataset

from imm.utils.io import find_images, load_image_tensor


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


class ImagesFromList(Dataset):
    """Dataset for loading images from a directory."""

    def __init__(self, root: str, resize: int = -1):
        # root
        self.root = Path(root)

        # collect image paths
        self.images_paths = sorted(find_images(self.root))

        if not self.images_paths:
            raise FileNotFoundError(f"No images found under {self.root}")

        # image names
        self.names = [relative_path(img_path, self.root)
                      for img_path in self.images_paths]

        #
        self.resize = resize

        logger.info("ImagesFromList:")
        logger.info(f"      Images: {len(self.images_paths)} in {self.root}")
        logger.info(f"      Max image size: {resize}")

    def __len__(self):
        return len(self.images_paths)

    def get_names(self) -> list[str]:
        return self.names

    def __getitem__(self, item: int) -> dict:
        if item < 0 or item >= len(self.images_paths):
            raise IndexError(
                f"Index {item} out of range for dataset with {len(self.images_paths)} images")

        img_path = self.images_paths[item]
        img_name = self.names[item]

        # load image
        image, _, _, scale, original_size = load_image_tensor(
            img_path, resize=self.resize)

        return {
            "image": image,
            "name": img_name,
            "path": str(img_path),
            "original_size": original_size,
            "scale": scale,
        }

    def __repr__(self):
        return (
            f"ImagesFromList(root={self.root}, num_images={len(self.images_paths)}, resize={self.resize})"
        )
