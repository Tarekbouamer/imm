import sys
from pathlib import Path
from typing import Optional

import click
import matplotlib
import numpy as np
from loguru import logger

from imm.cli._match import Matching, load_and_process_image
from imm.estimators import (
    CV_H_SOLVERS,
    create_fundamental_estimator,
    create_homography_estimator,
    create_relative_pose_estimator,
)
from imm.geometry import Camera
from imm.settings import img0_path as default_img0_path
from imm.settings import img1_path as default_img1_path
from imm.utils.device import detect_device
from imm.utils.logger import set_log_dir
from imm.utils.warnings import suppress_warnings
from imm.viz import EpipolarVisualizer, HomographyVisualizer, MatchVisualizer


@click.group()
@click.help_option("--help", "-h")
def cli():
    """Estimation tools for image transformations.

    Commands:
    - homography: Estimate the homography transformation between two images.
    - relative_pose: Estimate the relative pose between two images
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
@click.option("--reproj_thd", default=2.0, type=float, help="Reprojection error threshold")
@click.option("--max_iters", default=1000, type=int, help="Max iterations")
@click.option("--confidence", default=0.998, type=float, help="Confidence level")
@click.option("--max_keypoints", default=-1, type=int, help="Max keypoints to keep (-1 keeps all)")
@click.option("--resize", default=640, type=int, help="Resize to max dimension")
@click.option("--output", default="output", help="Directory for logs and visualization")
@click.option("--show", is_flag=True, help="Show homography warp visualization")
@click.option("--warp-alpha", default=0.5, type=float, help="Blending factor for warp visualization (0-1)")
@click.option("--force_cpu", is_flag=True, help="Force the use of CPU instead of GPU")
@click.help_option("--help", "-h")
def homography(
    img0_path: str,
    img1_path: str,
    matcher: str,
    extractor: str,
    backend: str,
    solver: str,
    reproj_thd: float,
    max_iters: int,
    confidence: float,
    max_keypoints: int,
    resize: Optional[int],
    output: str,
    show: bool,
    warp_alpha: float,
    force_cpu: bool,
):
    """Estimate the homography transformation between two images."""
    suppress_warnings()

    # Configure logging to output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(log_dir=output_dir, app_name="homography")

    logger.info(f"Homography estimation using {backend} backend")

    # Device
    device = detect_device(force_cpu)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, resize, device)
    image1, image1_cv = load_and_process_image(img1_path, resize, device)

    # Match images
    matching: Matching = Matching(matcher_name=matcher,
                                  device=device, extractor_name=extractor,
                                  max_keypoints=max_keypoints)
    m_preds = matching.match_images(image0, image1)

    # Get the estimator
    h_estimator = create_homography_estimator(
        backend, solver, reproj_thd, max_iters, confidence)

    # Estimate the transformation
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
            show_image=False,
        )

        # Save results and visualization
        output_dir = Path(output)

        # Save homography matrix
        h_file = output_dir / "homography.npy"
        np.save(h_file, H)
        logger.info(f"Homography saved to {h_file}")

        # Save visualization
        viz_file = output_dir / "homography_matches.png"
        vis.save(str(viz_file))
        logger.info(f"Visualization saved to {viz_file}")

        # Homography warp visualization
        warp_vis = HomographyVisualizer()
        blended = warp_vis.draw_homography_warp(
            image0_cv,
            image1_cv,
            H,
            alpha=warp_alpha,
            title="Homography Warp",
            show_image=show,
        )

        # Save warp visualization
        warp_file = output_dir / "homography_warp.png"
        warp_vis.save(str(warp_file))
        logger.info(f"Warp visualization saved to {warp_file}")

        logger.info(f"Estimation successful: {H}")

    else:
        logger.error("Estimation failed")
        sys.exit(1)

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
@click.option("--max_keypoints", default=-1, type=int, help="Max keypoints to keep (-1 keeps all)")
@click.option("--resize", default=640, type=int, help="Resize to max dimension")
@click.option("--output", default="output", help="Directory for logs and visualization")
@click.option("--show", is_flag=True, help="Show the matches")
@click.option("--n_lines", default=5, type=int, help="Number of epipolar lines to visualize")
@click.option("--force_cpu", is_flag=True, help="Force the use of CPU instead of GPU")
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
    max_keypoints: int,
    resize: Optional[int],
    output: str,
    show: bool,
    n_lines: int,
    force_cpu: bool,
):
    """Estimate the relative pose between two images."""
    suppress_warnings()

    # Configure logging to output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(log_dir=output_dir, app_name="relative_pose")

    logger.info(f"Relative pose estimation using {backend} backend")

    # Device
    device = detect_device(force_cpu)

    # Load and process images
    image0, image0_cv = load_and_process_image(img0_path, resize, device)
    image1, image1_cv = load_and_process_image(img1_path, resize, device)

    # h, w of the image0
    h, w = image0_cv.shape[:2]

    # Get image dimensions
    camera0 = Camera.from_image(image0_cv)
    camera1 = Camera.from_image(image1_cv)

    # Match images
    matching: Matching = Matching(matcher_name=matcher,
                                  device=device, extractor_name=extractor,
                                  max_keypoints=max_keypoints)
    m_preds = matching.match_images(image0, image1)

    # Estimator
    estimator = create_relative_pose_estimator(
        backend, solver, threshold, confidence, max_iters)
    r_preds = estimator.estimate(
        m_preds["mkpts0"], m_preds["mkpts1"], camera0, camera1)

    if r_preds["success"]:
        R, t = r_preds["R"], r_preds["t"]
        inliers = r_preds["inliers"]
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
            show_image=False,
        )

        # Epipolar geometry visualization
        # Epipolar visualization
        epi_vis = EpipolarVisualizer()
        epi_vis.draw_epipolar_lines(
            image0_cv,
            image1_cv,
            m_preds["mkpts0"],
            m_preds["mkpts1"],
            R,
            t,
            camera0,
            camera1,
            inliers=inliers,
            n_lines=n_lines,
            show_image=show,
        )

        # Save results and visualization
        output_dir = Path(output)

        # Save visualization
        viz_file = output_dir / "pose_epipolar.png"
        epi_vis.save(str(viz_file))
        logger.info(f"Epipolar visualization saved to {viz_file}")

        logger.info(f"Estimation successful: {R}, {t}")

        if inliers is not None:
            logger.info(
                f"Inliers: {inliers.sum()}/{len(inliers)}")

    else:
        logger.error("Estimation failed")
        sys.exit(1)

    logger.success("Estimation completed")


if __name__ == "__main__":
    cli()
