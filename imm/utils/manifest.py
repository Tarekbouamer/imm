import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

import imm


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive system environment information.

    Captures:
    - Python and PyTorch versions
    - Platform information
    - CPU model and core count
    - All available GPU devices with memory info
    - Optional package versions (opencv, poselib, pycolmap)

    Returns:
        Dictionary with environment details
    """
    env = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }

    # CPU information
    try:
        cpu_count = os.cpu_count() or 1
        env["cpu_cores"] = cpu_count
        # Try to get CPU model name
        try:
            if platform.system() == "Linux":
                cpu_model = subprocess.check_output(
                    "grep -m 1 'model name' /proc/cpuinfo | cut -d ':' -f 2 | xargs",
                    shell=True, text=True, stderr=subprocess.DEVNULL).strip()
                if cpu_model:
                    env["cpu_model"] = cpu_model
            elif platform.system() == "Darwin":
                cpu_model = subprocess.check_output(
                    "sysctl -n machdep.cpu.brand_string",
                    shell=True, text=True, stderr=subprocess.DEVNULL).strip()
                if cpu_model:
                    env["cpu_model"] = cpu_model
        except Exception:
            pass
    except Exception:
        pass

    # GPU information - all devices
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["gpu_count"] = torch.cuda.device_count()
        env["gpus"] = []
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            env["gpus"].append(gpu_info)

    # IMM package version
    env["imm_version"] = imm.__version__

    # Optional package versions
    optional_packages = {
        "opencv": "cv2",
        "poselib": "poselib",
        "pycolmap": "pycolmap",
    }

    for pkg_name, import_name in optional_packages.items():
        try:
            module = __import__(import_name)
            if hasattr(module, "__version__"):
                env[f"{pkg_name}_version"] = module.__version__
        except ImportError:
            pass

    return env


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB.

    Args:
        file_path: Path to the file

    Returns:
        File size in MB, or 0.0 if file doesn't exist
    """
    if not file_path.exists():
        return 0.0
    return round(file_path.stat().st_size / (1024 * 1024), 2)


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate min/max/avg statistics.

    Args:
        values: List of numeric values to analyze

    Returns:
        Dictionary with 'avg', 'min', 'max' keys
    """
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}

    return {
        "avg": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def create_extraction_manifest(
    extractor_name: str,
    config: Dict[str, Any],
    device: str,
    total_time: float,
    processed_images: List[str],
    output_file: str,
    skipped_images: int = 0,
    failed_images: int = 0,
    keypoint_counts: Optional[List[int]] = None,
    processing_times_ms: Optional[List[float]] = None,
    resume_mode: bool = False,
    errors: Optional[List[Dict[str, Any]]] = None,
    parent_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    """Create extraction manifest dictionary.

    Args:
        extractor_name: Name of the extractor model
        config: Extractor configuration
        device: Device used for processing
        total_time: Total processing time in seconds
        processed_images: List of processed image paths
        output_file: Output file path
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
        "output_file": Path(output_file).name,
        "output_size_mb": get_file_size_mb(Path(output_file)),
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

    return manifest


def create_matching_manifest(
    matcher_name: str,
    config: Dict[str, Any],
    device: str,
    total_time: float,
    processed_pairs: List[List[str]],
    output_file: str,
    extractor_name: Optional[str] = None,
    features_file: Optional[str] = None,
    skipped_pairs: int = 0,
    failed_pairs: int = 0,
    match_counts: Optional[List[int]] = None,
    match_confidences: Optional[List[float]] = None,
    processing_times_ms: Optional[List[float]] = None,
    resume_mode: bool = False,
    errors: Optional[List[Dict[str, Any]]] = None,
    parent_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    """Create matching manifest dictionary.

    Args:
        matcher_name: Name of the matcher model
        config: Matcher configuration
        device: Device used for processing
        total_time: Total processing time in seconds
        processed_pairs: List of processed image pairs
        output_file: Output file path
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
        "output_file": Path(output_file).name,
        "output_size_mb": get_file_size_mb(Path(output_file)),
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

    return manifest


def create_estimation_manifest(
    estimator_name: str,
    backend: str,
    solver: str,
    config: Dict[str, Any],
    device: str,
    total_time: float,
    pair_results: List[Dict[str, Any]],
    matches_file: Optional[str] = None,
    output_file: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    parent_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    """Create estimation manifest dictionary.

    Args:
        estimator_name: Name of the estimator
        backend: Backend used (opencv, poselib, pycolmap)
        solver: Solver method (ransac, lmeds, etc.)
        config: Estimator configuration
        device: Device used for processing
        total_time: Total processing time in seconds
        pair_results: List of per-pair results with metrics
        matches_file: Input matches file
        output_file: Output file (if applicable)
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

    # Add output file info if provided
    if output_file:
        output_path = Path(output_file)
        manifest["output_file"] = output_path.name
        manifest["output_size_mb"] = get_file_size_mb(output_path)

    return manifest


def save_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """Save manifest dictionary to JSON file.

    Saves the manifest with the naming convention: <stem>_manifest.json

    Args:
        manifest: Manifest dictionary to save
        output_path: Base output path (manifest file created as <stem>_manifest.json)
    """
    manifest_path = output_path.parent / f"{output_path.stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
