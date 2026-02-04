from typing import Optional

import click
import numpy as np
from loguru import logger

from imm.estimators import (
    CV_H_SOLVERS,
    create_fundamental_estimator,
    create_homography_estimator,
    create_relative_pose_estimator,
)
from imm.estimators._camera import Camera
from imm.settings import img0_path as default_img0_path
from imm.settings import img1_path as default_img1_path
from imm.tools.match import Matching, load_and_process_image
from imm.utils.device import detect_device
from imm.utils.warnings import suppress_warnings

# Suppress warnings
suppress_warnings()


@click.group()
@click.help_option("--help", "-h")
def cli():
    """Estimation tools for image transformations.

    Commands:
    - homography: Estimate the homography transformation between two images.
    - relative_pose: Estimate the relative pose between two images
    - fundamental: Estimate the fundamental matrix between two images
    """
    pass


@cli.command()
@click.argument("img0_path", type=click.Path(exists=True), default=default_img0_path)
@click.argument("img1_path", type=click.Path(exists=True), default=default_img1_path)
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option(
    "--backend", default="opencv", type=click.Choice(["opencv", "poselib", "pycolmap"]), help="Estimator backend"
)
@click.option("--solver", default="ransac", type=click.Choice(CV_H_SOLVERS.keys()), help="Homography solver")
@click.option("--thd", default=2.0, type=float, help="Reprojection error threshold")
@click.option("--max_iters", default=1000, type=int, help="Max iterations")
@click.option("--confidence", default=0.998, type=float, help="Confidence level")
@click.option("--max_size", default=None, type=int, help="Max image size")
@click.option("--output_dir", default="output", help="Output directory for logs and visualization")
@click.option("--force_cpu", is_flag=True, help="Force the use of CPU instead of GPU")
@click.option("visualize", "--visualize", is_flag=True, help="Visualize the matches")
@click.help_option("--help", "-h")
def homography(
    img0_path: str,
    img1_path: str,
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
    visualize: bool,
):
    """Estimate the homography transformation between two images."""
    logger.info(f"Homography estimation using {backend} backend")

    # Device
    device = detect_device(force_cpu)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, max_size, device)
    image1, image1_cv = load_and_process_image(img1_path, max_size, device)

    # Match images
    matcher_model = Matching(matcher_name=matcher, device=device, extractor_name=extractor)
    m_preds = matcher_model.match_images(image0, image1)

    # Get the estimator
    h_estimator = create_homography_estimator(backend, solver, thd, max_iters, confidence)
    h_preds = h_estimator.estimate(m_preds["mkpts0"], m_preds["mkpts1"])

    if h_preds["success"]:
        H = h_preds["H"]
        inliers = h_preds["inliers"]

        # filter out the inliers
        mkpts0 = m_preds["mkpts0"][inliers]
        mkpts1 = m_preds["mkpts1"][inliers]

        m_valid = np.where(m_preds["matches"] > -1)[0]

        matches = m_preds["matches"][m_valid][inliers]
        mscores = m_preds["mscores"][m_valid][inliers]

        # Visualize the matches (lazy import so imm-gui / estimate without --visualize skip matplotlib)
        if visualize:
            from imm.utils.viz2d import MatchVisualizer

            vis = MatchVisualizer()
            vis.draw_matches(
                image0_cv,
                image1_cv,
                m_preds["kpts0"],
                m_preds["kpts1"],
                mkpts0,
                mkpts1,
                matches=matches,
                mscores=mscores,
            )

        logger.info(f"Estimation successful: {H}")

    else:
        logger.error("Estimation failed")

    logger.info("Estimation completed")


@cli.command()
@click.argument("img0_path", type=click.Path(exists=True), default=default_img0_path)
@click.argument("img1_path", type=click.Path(exists=True), default=default_img1_path)
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option(
    "--backend", default="opencv", type=click.Choice(["opencv", "poselib", "pycolmap"]), help="Estimator backend"
)
@click.option("--solver", default="ransac", type=click.Choice(["ransac", "usac_magsac"]), help="Pose solver")
@click.option("--threshold", default=1.0, type=float, help="Threshold value")
@click.option("--confidence", default=0.999, type=float, help="Confidence level")
@click.option("--max_iters", default=1000, type=int, help="Max iterations")
@click.option("--max_size", default=None, type=int, help="Max image size")
@click.option("--output_dir", default="output", help="Output directory for logs and visualization")
@click.option("--force_cpu", is_flag=True, help="Force the use of CPU instead of GPU")
@click.option("visualize", "--visualize", is_flag=True, help="Visualize the matches")
@click.help_option("--help", "-h")
def relative_pose(
    img0_path: str,
    img1_path: str,
    matcher: str,
    extractor: str,
    backend: str,
    solver: str,
    threshold: float,
    confidence: float,
    max_iters: int,
    max_size: Optional[int],
    output_dir: str,
    force_cpu: bool,
    visualize: bool,
):
    """Estimate the relative pose between two images."""
    logger.info(f"Relative pose estimation using {backend} backend")

    # Device
    device = detect_device(force_cpu)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, max_size, device)
    image1, image1_cv = load_and_process_image(img1_path, max_size, device)

    # h, w of the image0
    h, w = image0_cv.shape[:2]

    # Get image dimensions
    camera0 = Camera.from_image(image0_cv)
    camera1 = Camera.from_image(image1_cv)

    # Match images
    matcher_model = Matching(matcher_name=matcher, device=device, extractor_name=extractor)
    m_preds = matcher_model.match_images(image0, image1)

    # Get the estimator
    r_estimator = create_relative_pose_estimator(backend, solver, threshold, confidence, max_iters)

    # Estimate
    r_preds = r_estimator.estimate(m_preds["mkpts0"], m_preds["mkpts1"], camera0, camera1)

    if r_preds["success"]:
        #
        R, t, E = r_preds["R"], r_preds["t"], r_preds["E"]
        inliers = r_preds["inliers"]

        # Filter out the inliers
        mkpts0 = m_preds["mkpts0"][inliers]
        mkpts1 = m_preds["mkpts1"][inliers]

        m_valid = np.where(m_preds["matches"] > -1)[0]

        matches = m_preds["matches"][m_valid][inliers]
        mscores = m_preds["mscores"][m_valid][inliers]

        F = np.linalg.inv(camera1.K).T @ E @ np.linalg.inv(camera0.K)

        # Visualize the matches
        if visualize:
            vis = MatchVisualizer()

            # draw inlier matches
            vis.draw_matches(
                image0_cv,
                image1_cv,
                m_preds["kpts0"],
                m_preds["kpts1"],
                mkpts0,
                mkpts1,
                matches=matches,
                mscores=mscores,
            )

            # draw epipolar lines
            vis.draw_epipolar_line(
                image0_cv,
                image1_cv,
                F,
                kpts0=mkpts0,
                kpts1=mkpts1,
            )

        logger.info(f"Estimation successful: {R}, {t}")

    else:
        logger.error("Estimation failed")

    logger.success("Estimation completed")


@cli.command()
@click.argument("img0_path", type=click.Path(exists=True), default=default_img0_path)
@click.argument("img1_path", type=click.Path(exists=True), default=default_img1_path)
@click.option("--matcher", default="superglue_outdoor", help="Matcher name")
@click.option("--extractor", default="superpoint", help="Extractor name")
@click.option(
    "--backend", default="opencv", type=click.Choice(["opencv", "poselib", "pycolmap"]), help="Estimator backend"
)
@click.option("--solver", default="ransac", type=click.Choice(["ransac", "usac_magsac"]), help="Pose solver")
@click.option("--threshold", default=1.0, type=float, help="Threshold value")
@click.option("--confidence", default=0.999, type=float, help="Confidence level")
@click.option("--max_iters", default=1000, type=int, help="Max iterations")
@click.option("--max_size", default=None, type=int, help="Max image size")
@click.option("--output_dir", default="output", help="Output directory for logs and visualization")
@click.option("--force_cpu", is_flag=True, help="Force the use of CPU instead of GPU")
@click.option("visualize", "--visualize", is_flag=True, help="Visualize the matches")
@click.help_option("--help", "-h")
def fundamental(
    img0_path: str,
    img1_path: str,
    matcher: str,
    extractor: str,
    backend: str,
    solver: str,
    threshold: float,
    confidence: float,
    max_iters: int,
    max_size: Optional[int],
    output_dir: str,
    force_cpu: bool,
    visualize: bool,
):
    """Estimate the fundamental matrix between two images."""
    logger.info(f"Fundamental matrix estimation using {backend} backend")

    # Device
    device = detect_device(force_cpu)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, max_size, device)
    image1, image1_cv = load_and_process_image(img1_path, max_size, device)

    # Match images
    matcher_model = Matching(matcher_name=matcher, device=device, extractor_name=extractor)
    m_preds = matcher_model.match_images(image0, image1)

    # Get the estimator
    f_estimator = create_fundamental_estimator(backend, solver, threshold, confidence, max_iters)

    # Estimate
    f_preds = f_estimator.estimate(m_preds["mkpts0"], m_preds["mkpts1"])

    if f_preds["success"]:
        #
        F, inliers = f_preds["F"], f_preds["inliers"]

        # Filter out the inliers
        mkpts0 = m_preds["mkpts0"][inliers]
        mkpts1 = m_preds["mkpts1"][inliers]

        m_valid = np.where(m_preds["matches"] > -1)[0]

        matches = m_preds["matches"][m_valid][inliers]
        mscores = m_preds["mscores"][m_valid][inliers]

        # Visualize the matches (lazy import so imm-gui / estimate without --visualize skip matplotlib)
        if visualize:
            from imm.utils.viz2d import MatchVisualizer

            vis = MatchVisualizer()

            vis.draw_matches(
                image0_cv,
                image1_cv,
                m_preds["kpts0"],
                m_preds["kpts1"],
                mkpts0,
                mkpts1,
                matches=matches,
                mscores=mscores,
            )

            # draw epipolar lines
            vis.draw_epipolar_line(
                image0_cv,
                image1_cv,
                F,
                kpts0=mkpts0,
                kpts1=mkpts1,
            )

        logger.info(f"Estimation successful: {F}")

    else:
        logger.error("Estimation failed")

    logger.success("Estimation completed")


if __name__ == "__main__":
    cli()
