import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import version as torch_version

import imm


def get_environment_info() -> dict[str, Any]:
    """Get comprehensive system environment information.

    Returns:
        Dictionary with environment details
    """
    env = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "imm_version": imm.__version__,
        "torch_version": torch.__version__,
        "cpu_cores": os.cpu_count() or 1,
    }

    # Device information
    cuda_available = torch.cuda.is_available()
    env["cuda_available"] = cuda_available

    if cuda_available:
        env["cuda_version"] = torch_version.cuda
        env["gpu_count"] = torch.cuda.device_count()
        env["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(
                    torch.cuda.get_device_properties(
                        i).total_memory / 1024**3, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ]

    return env


def calculate_stats(values: Sequence[float | int]) -> dict[str, float]:
    """Calculate min/max/avg statistics.
    """
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}

    return {
        "avg": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def save_extraction_manifest(
    extractor_name: str,
    config: dict[str, Any],
    device: str,
    total_time: float,
    processed_images: list[str],
    manifest_path: Path,
    skipped_images: int = 0,
    failed_images: int = 0,
    keypoint_counts: Optional[list[int]] = None,
    processing_times_ms: Optional[list[float]] = None,
    resume_mode: bool = False,
    errors: Optional[list[dict[str, Any]]] = None,
    parent_manifest: Optional[str] = None,
) -> dict[str, Any]:
    """Create and save extraction manifest dictionary.

    Args:
        extractor_name: Name of the extractor model
        config: Extractor configuration
        device: Device used for processing
        total_time: Total processing time in seconds
        processed_images: List of processed image paths
        manifest_path: Full path to manifest JSON file
        skipped_images: Number of skipped images
        failed_images: Number of failed images
        keypoint_counts: List of keypoint counts per image
        processing_times_ms: List of processing times per image (ms)
        resume_mode: Whether resume mode was used
        errors: List of errors encountered
        parent_manifest: Path to parent manifest for workflow chaining

    Returns:
        Manifest dictionary
    """
    total_images = len(processed_images)

    manifest = {
        "stage": "extraction",
        "extractor": extractor_name,
        "config": config,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,
        "total_time_seconds": round(total_time, 2),
        "images_per_second": round(total_images / total_time, 2) if total_time > 0 else 0.0,
        "skipped_images": skipped_images,
        "failed_images": failed_images,
        "errors": errors or [],
        "environment": get_environment_info(),
        "resume_mode": resume_mode,
        "images": processed_images,
    }

    # Add parent manifest reference if provided
    if parent_manifest:
        manifest["parent_manifest"] = parent_manifest

    # Add statistics if available
    if keypoint_counts:
        manifest["stats"] = {
            **calculate_stats(keypoint_counts),
            "total_keypoints": sum(keypoint_counts),
        }
        if processing_times_ms:
            manifest["stats"]["avg_processing_time_ms"] = round(
                sum(processing_times_ms) / len(processing_times_ms), 2
            )

    # Save manifest
    save_manifest(manifest, manifest_path)

    return manifest


def create_matching_manifest(
    matcher_name: str,
    config: dict[str, Any],
    device: str,
    total_time: float,
    processed_pairs: list[tuple[str, str]],
    manifest_path: Path,
    extractor_name: Optional[str] = None,
    features_file: Optional[str] = None,
    skipped_pairs: int = 0,
    failed_pairs: int = 0,
    match_counts: Optional[list[int]] = None,
    match_confidences: Optional[list[float]] = None,
    processing_times_ms: Optional[list[float]] = None,
    resume_mode: bool = False,
    errors: Optional[list[dict[str, Any]]] = None,
    parent_manifest: Optional[str] = None,
) -> dict[str, Any]:
    """Create and save matching manifest dictionary.

    Args:
        matcher_name: Name of the matcher model
        config: Matcher configuration
        device: Device used for processing
        total_time: Total processing time in seconds
        processed_pairs: List of processed image pairs
        manifest_path: Full path to manifest JSON file
        extractor_name: Name of extractor (if sparse matching)
        features_file: Input features file (if sparse)
        skipped_pairs: Number of skipped pairs
        failed_pairs: Number of failed pairs
        match_counts: List of match counts per pair
        match_confidences: List of average match confidences per pair
        processing_times_ms: List of processing times per pair (ms)
        resume_mode: Whether resume mode was used
        errors: List of errors encountered
        parent_manifest: Path to parent manifest for workflow chaining

    Returns:
        Manifest dictionary
    """
    total_pairs = len(processed_pairs)

    manifest = {
        "stage": "matching",
        "matcher": matcher_name,
        "extractor": extractor_name,
        "config": config,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "total_pairs": total_pairs,
        "total_time_seconds": round(total_time, 2),
        "pairs_per_second": round(total_pairs / total_time, 2) if total_time > 0 else 0.0,
        "skipped_pairs": skipped_pairs,
        "failed_pairs": failed_pairs,
        "errors": errors or [],
        "environment": get_environment_info(),
        "features_file": features_file,
        "resume_mode": resume_mode,
        "pairs": processed_pairs,
    }

    # Add parent manifest reference if provided
    if parent_manifest:
        manifest["parent_manifest"] = parent_manifest

    # Add statistics if available
    if match_counts:
        manifest["stats"] = {
            **calculate_stats(match_counts),
            "pairs_with_no_matches": sum(1 for c in match_counts if c == 0),
        }
        if match_confidences:
            manifest["stats"]["avg_match_confidence"] = round(
                sum(match_confidences) / len(match_confidences), 2
            ) if match_confidences else 0.0
        if processing_times_ms:
            manifest["stats"]["avg_processing_time_ms"] = round(
                sum(processing_times_ms) / len(processing_times_ms), 2
            )

    # Save manifest
    save_manifest(manifest, manifest_path)

    return manifest


def create_estimation_manifest(
    estimator_name: str,
    backend: str,
    solver: str,
    config: dict[str, Any],
    device: str,
    total_time: float,
    pair_results: list[dict[str, Any]],
    manifest_path: Path,
    matches_file: Optional[str] = None,
    errors: Optional[list[dict[str, Any]]] = None,
    parent_manifest: Optional[str] = None,
) -> dict[str, Any]:
    """Create and save estimation manifest dictionary.

    Args:
        estimator_name: Name of the estimator
        backend: Backend used (opencv, poselib, pycolmap)
        solver: Solver method (ransac, lmeds, etc.)
        config: Estimator configuration
        device: Device used for processing
        total_time: Total processing time in seconds
        pair_results: List of per-pair results with metrics
        manifest_path: Full path to manifest JSON file
        matches_file: Input matches file
        errors: List of errors encountered
        parent_manifest: Path to parent manifest for workflow chaining

    Returns:
        Manifest dictionary
    """
    total_pairs = len(pair_results)
    successful = sum(1 for r in pair_results if r.get("success", False))
    failed = total_pairs - successful

    # Calculate statistics from successful estimations
    inlier_counts = [r["num_inliers"] for r in pair_results if r.get(
        "success") and "num_inliers" in r]
    inlier_ratios = [r["inlier_ratio"] for r in pair_results if r.get(
        "success") and "inlier_ratio" in r]
    reproj_errors = [r["reprojection_error"] for r in pair_results if r.get(
        "success") and "reprojection_error" in r]
    processing_times = [r["processing_time_ms"]
                        for r in pair_results if "processing_time_ms" in r]

    manifest = {
        "stage": "estimation",
        "estimator": estimator_name,
        "backend": backend,
        "solver": solver,
        "config": config,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "total_pairs": total_pairs,
        "total_time_seconds": round(total_time, 2),
        "pairs_per_second": round(total_pairs / total_time, 2) if total_time > 0 else 0.0,
        "successful_estimations": successful,
        "failed_estimations": failed,
        "success_rate": round(successful / total_pairs, 3) if total_pairs > 0 else 0.0,
        "errors": errors or [],
        "environment": get_environment_info(),
        "matches_file": matches_file,
        "pairs": pair_results,
    }

    # Add parent manifest reference if provided
    if parent_manifest:
        manifest["parent_manifest"] = parent_manifest

    # Add statistics
    stats = {}
    if inlier_counts:
        stats.update({f"{k}_inliers": v for k,
                     v in calculate_stats(inlier_counts).items()})
    if inlier_ratios:
        stats["avg_inlier_ratio"] = round(
            sum(inlier_ratios) / len(inlier_ratios), 3)
    if reproj_errors:
        stats["avg_reprojection_error"] = round(
            sum(reproj_errors) / len(reproj_errors), 3)
    if processing_times:
        stats["avg_processing_time_ms"] = round(
            sum(processing_times) / len(processing_times), 2)

    if stats:
        manifest["stats"] = stats

    # Save manifest
    save_manifest(manifest, manifest_path)

    return manifest


def save_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Save manifest dictionary to JSON file.
    """
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
