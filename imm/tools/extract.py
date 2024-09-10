import click
from imm.utils.device import to_numpy
from imm.utils.io import load_image_tensor
from loguru import logger

from imm.extractors._helper import create_extractor
from imm.settings import img0_path
from imm.utils.logger import setup_logger
from imm.utils.viz2d import KeypointVisualizer


@click.command()
@click.option("--model", default="superpoint", help="Extractor name")
@click.option("--img_path", default=img0_path, help="Path to the first image")
@click.option("--max_keypoints", default=1000, help="Maximum number of keypoints")
@click.help_option("--help", "-h")
def extract(model: str, img_path: str, max_keypoints: int):
    # logger
    setup_logger(app_name="imm")

    # load image
    data = load_image_tensor(img_path)
    image, image_cv = data[0], data[1]

    # create extractor
    detector = create_extractor(model, cfg={"max_keypoints": max_keypoints}, pretrained=True)
    detector.eval()
    detector.cpu()

    # extract
    preds = detector.extract({"image": image})
    preds = to_numpy(preds)

    #
    kpts, scores, descriptors = (
        preds["kpts"],
        preds["scores"],
        preds["desc"],
    )

    if isinstance(kpts, list):
        kpts = kpts[0]
        scores = scores[0]
        descriptors = descriptors[0]

    if len(kpts.shape) == 3:
        kpts = kpts[0]
        scores = scores[0]

    logger.info(f"Keypoints: {kpts.shape}")
    logger.info(f"Scores: {scores.shape}")
    logger.info(f"Descriptors: {descriptors.shape}")

    # visualize
    visualizer = KeypointVisualizer()
    visualizer.visualize_keypoints(image_cv, kpts, scores)


if __name__ == "__main__":
    #
    extract()
