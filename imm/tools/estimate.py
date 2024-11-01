from typing import Optional

import click
from loguru import logger

from imm.estimators import CV_H_SOLVERS, ESTIMATORS_2D, create_homography_estimator
from imm.settings import img0_path as default_img0_path
from imm.settings import img1_path as default_img1_path
from imm.tools.match import Matching, load_and_process_image
from imm.utils.device import detect_device
from imm.utils.logger import setup_logger
from imm.utils.warnings import suppress_warnings


@click.command()
@click.argument("img0_path", type=click.Path(exists=True), default=default_img0_path)
@click.argument("img1_path", type=click.Path(exists=True), default=default_img1_path)
@click.option("--estimator", default="homography", help="Estimator name", type=click.Choice(ESTIMATORS_2D))
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option("--backend", default="cv", type=click.Choice(["cv", "poselib", "pycolmap"]), help="Estimator backend")
@click.option("--solver", default="ransac", type=click.Choice(CV_H_SOLVERS.keys()), help="Homography solver")
@click.option("--thd", default=2.0, type=float, help="Reprojection error threshold")
@click.option("--max_iters", default=1000, type=int, help="Max iterations")
@click.option("--confidence", default=0.998, type=float, help="Confidence level")
@click.option("--max_size", default=None, type=int, help="Max image size")
@click.option("--output_dir", default="output", help="Output directory for logs and visualization")
@click.option("--force_cpu", is_flag=False, help="Force the use of CPU instead of GPU")
@click.help_option("--help", "-h")
@suppress_warnings()
def estimate(
    img0_path: str,
    img1_path: str,
    estimator: str,
    matcher: str,
    extractor: str,
    backend: str,
    solver: str,
    thd: float,
    max_iters: int,
    confidence: float,
    max_size: Optional[int],
    output_dir: str,
    force_cpu: bool,
):
    """Estimate the transformation between two images."""
    setup_logger(app_name="imm")

    logger.info(f"Starting image estimation process {estimator}-{backend}")

    # Device
    device = detect_device(force_cpu)

    # Load and process images
    image0 = load_and_process_image(img0_path, max_size, device)[0]
    image1 = load_and_process_image(img1_path, max_size, device)[0]

    # Match images
    matcher_model = Matching(matcher_name=matcher, device=device, extractor_name=extractor)
    preds = matcher_model.match_images(image0, image1)

    # Get the estimator
    if estimator == "homography":
        estimator = create_homography_estimator(backend, solver, thd, max_iters, confidence)
    else:
        raise ValueError(f"Unknown estimator: {estimator}", available=["homography"])

    # Estimate the transformation
    preds = estimator.estimate(preds["mkpts0"], preds["mkpts1"])

    sucess = preds["success"]
    H = preds["H"]
    inliers = preds["inliers"]

    if sucess:
        logger.info(f"Estimation successful: {H}")
        logger.info(f"Inliers: {inliers}")
    else:
        logger.error("Estimation failed")

    logger.success("Estimation completed successfully")


if __name__ == "__main__":
    estimate()
