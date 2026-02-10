# IMM Features & Roadmap - Category-Based Checklist

Tasks organized by feature category, with completion status across all work streams.

---

## Priority 1: HDF5 Data Management & Optimization

### HDF5 Compression & Performance

**Problem:** HDF5 files can be large; need compression and optimization options for storage and I/O efficiency.

**Steps:**

1. Add compression options for HDF5 writers (gzip levels, lzf, blosc)
2. Configure chunk sizes for optimal read/write performance
3. Add compression level CLI flags to extraction and matching commands
4. Benchmark compression vs speed tradeoffs

**Benefits:** Reduced storage requirements, faster network transfers, configurable performance

---

### HDF5 Streaming & Append Mode

**Problem:** Current writers require full dataset in memory; need streaming for large-scale datasets.

**Steps:**

1. Implement streaming writers for memory-constrained environments
2. Add append mode for incremental dataset building
3. Support resumable writes (handle interruptions gracefully)
4. Add buffer size configuration for streaming operations

**Benefits:** Handle datasets larger than RAM, incremental processing, fault tolerance

---

### HDF5 Metadata & Indexing

**Problem:** No efficient way to query HDF5 contents without loading entire file.

**Steps:**

1. Add HDF5 index/metadata for fast querying (image names, feature counts, timestamps)
2. Create CLI inspection tool (`imm-inspect`) for querying HDF5 contents
3. Support filter/search operations (by name pattern, feature count range, etc.)
4. Add summary statistics generation (total images, keypoint distribution, file size breakdown)

**Benefits:** Fast dataset inspection, easier debugging, better data management

---

### Save Extracted Features & Matched Pairs Outside Manifest

**Problem:** Extracted features and matched pairs are stored in HDF5, but file paths and pair relationships need to be preserved in external manifests for reproducibility and data lineage.

**Steps:**

1. Add CSV/JSON manifest output for extraction (image paths, feature counts, timestamps)
2. Save pair lists for matching operations (input pairs, successful matches, failed matches)
3. Include metadata: extraction config, model version, processing time
4. Support manifest formats compatible with common evaluation tools
5. Add validation to verify manifest consistency with HDF5 contents

**Benefits:** Reproducible experiments, easier dataset sharing, clear data provenance, compatibility with external tools

---

## Consistency & Quality

_All organizational and structural refactoring tasks completed. Feature-specific items tracked in their respective sections below._

---

## Models

- [x] `ModelBase` base class for all models
- [x] `FeatureModel` base class for extractors
- [x] `MatcherModel` base class for matchers
- [x] Image transforms (`tfn_grayscale`, `tfn_image_net`)

---

## Geometry

- [x] `Camera` class for camera models and intrinsics
- [x] Geometric metrics (AUC, pose errors, epipolar distance, reprojection error)
- [x] Pose error computation
- [x] Epipolar error computation
- [x] Reprojection error computation
- [x] Homography error computation
- [x] Precision-recall curve computation
- [x] Pose AUC computation

---

## API

- [x] Stable public API module (`imm/api/`)
- [x] Factory functions (`create_extractor`, `create_matcher`)
- [x] Model weight download API (`download_model_weights`)
- [x] Extractor registry (`EXTRACTORS_REGISTRY`)
- [x] Matcher registry (`MATCHERS_REGISTRY`)
- [ ] Standardize output schemas for all operations (JSON + numpy arrays)
- [ ] Request/response models (Pydantic) for type safety and validation
- [ ] Unified error/exception taxonomy for predictable failure handling
- [ ] Input validation utilities (image size, format, dtype checks)
- [ ] Output validation for extractors/matchers (shape, range checks)

---

## CLI

- [x] CLI feature extraction (`imm-extract`) for single image and datasets
- [x] CLI matching (`imm-match`) for image pairs
- [x] CLI estimation (`imm-estimate`) for homography and relative pose
- [x] CLI download models (`imm-download`) with --all and --name options
- [x] CLI export (`imm-export`) for format conversion (dummy)
- [x] CLI optimize (`imm-optimize`) for ONNX/TorchScript/quantization/pruning (dummy)
- [x] Consistent CLI flags and help text across all tools
- [x] Click group refactoring for `imm-match` (three subcommands: pair, images, features)
- [x] Parse pairs file utility (`parse_pairs_file()`) for space-separated pairs
- [x] Dataset-based matching from pairs file (ImagePairsDataset, FeaturesPairsDataset)
- [x] Output directory structure (matches.h5 + matches_manifest.json)
- [ ] Implement detector threshold (det_thd) in extraction
- [ ] Add --dry-run flag to preview operations without execution
- [ ] Add --verbose flag for detailed logging
- [ ] Pipeline CLI for chaining extract → match → estimate

---

## Data I/O

### Datasets

- [x] `ImagesFromList` dataset (single images)
- [x] `ImagePairsDataset` (image pairs)
- [x] `FeaturesPairsDataset` (feature pairs from HDF5)

### Writers

- [x] `H5Writer` base writer for HDF5
- [x] `FeaturesWriter` for feature export
- [x] `MatchesWriter` for match export
- [x] `AsycMatchesWriter` for async match export
- [ ] HDF5 optimization tasks (see Priority 1)

---

## Core / Base Elements

### Registry System

- [x] Registry decorator-based model registration
- [x] Factory pattern for model creation and loading
- [x] Pretrained weight caching to hub/
- [x] Model weight loading with automatic download

---

## Device & Infrastructure

- [x] Device auto-detection (`detect_device`)
- [x] Lazy visualization imports to avoid optional dependency issues
- [x] Support for optional backends (OpenCV, PoseLib, PyColmap)
- [ ] Centralized device/dtype/precision configuration
- [ ] Automatic garbage collection triggers
- [ ] CUDA graph capture for fixed-size inputs

---

## Download & Model Management

- [x] Automatic model weight download from hub
- [x] Model weight caching to hub/ directory
- [x] CLI download all models (`imm-download --all`)
- [x] CLI download specific model by name (`imm-download --name superpoint`)
- [x] Custom download path override (`imm-download --path /custom/dir`)
- [x] Download progress reporting (percentage, speed, ETA)
- [x] Python API for downloading models (`download_model_weights`)

---

## Extraction

- [x] Extractors registry with pretrained weight loading
- [x] Dataset loader for extraction (`ImagesFromList`)
- [x] Dataset extraction with progress/ETA (`tqdm` in `extract_dataset`)
- [x] Resume/skip logic (skip already extracted images, --override flag to force re-extraction)
- [x] Error recovery in extraction (continue on failures, log errors)
- [ ] Add image preprocessing options (resize methods, padding strategies)
- [ ] Support for custom image loaders (WebP, HEIF formats)
- [ ] Extractor warm-up run to avoid first-image slowdown
- [ ] Benchmark script for extractor speed/accuracy comparison
- [ ] Distributed extraction/matching across multiple machines

---

## Matching

- [x] Sparse and dense matcher support via `required_inputs`
- [x] Matcher registry with pretrained weight loading
- [x] Match filtering by score threshold
- [x] Dataset/list-of-pairs matching with progress/ETA (`match_sequence_images`, `match_sequence_features`)
- [x] Dataset-level matching from pairs file (via Click subcommands)
- [ ] Add match confidence filtering thresholds
- [ ] Implement reciprocal/mutual nearest neighbor check
- [ ] Add Lowe's ratio test option for matchers
- [ ] Cross-check / mutual consistency options exposed in CLI
- [ ] gRPC service for high-throughput matching
- [ ] Loop closure detection module

---

## Estimation

- [x] Estimator factory utilities
- [x] Success flag in estimator outputs
- [x] Homography estimation (OpenCV, PoseLib)
- [x] PnP estimation (OpenCV, PoseLib)
- [x] Relative pose estimation (OpenCV, PoseLib)
- [x] PyColmap optional dependency integration
- [ ] Homography estimation via PyColmap
- [ ] PnP estimation via PyColmap
- [ ] Relative pose estimation via PyColmap
- [ ] RANSAC parameter auto-tuning based on match count
- [ ] Multi-model fitting (fit multiple hypotheses)
- [ ] Degeneracy detection (planar scenes, pure rotation)
- [ ] Robust estimation with outlier analysis and statistics
- [ ] Relative pose chain solver (incremental pose tracking)
- [ ] Colmap integration / export format support

---

## Visualization

- [x] `Viz2D` base visualization class
- [x] `KeypointVisualizer` for keypoint visualization
- [x] `MatchVisualizer` for match visualization
- [x] `HomographyVisualizer` for homography warp visualization
- [x] Homography warp preview with 3-panel display (source, warped, blended)
- [x] Configurable blending alpha for warp visualization

---

## UI

### Gradio UI

- [x] Gradio UI (`imm-gui`) for interactive image matching
- [ ] Interactive parameter tuning UI
- [ ] Model selection dropdown
- [ ] Export results from UI

---

## Parallelization & Distribution

- [x] Parallel execution (DataLoader workers for datasets)
- [x] Progress + ETA reporting (via `tqdm`)
- [x] Resume/skip logic for extraction (read HDF5, skip already extracted images)
- [x] Error recovery in extraction (continue on single image failures, log errors)
- [x] Async image loading with prefetching for extraction (pin_memory, prefetch_factor, persistent_workers)
- [x] Async image loading with prefetching for matching (pin_memory, prefetch_factor, persistent_workers)

---

## Artifacts & Output Formats

### Extraction Outputs

- [x] HDF5 export (`features.h5`) with keypoints, descriptors, scores per image
- [x] Extraction manifest JSON with config, stats, environment
- [x] Unified environment info (CPU, GPU, package versions including imm)
- [x] Stage identifier and parent manifest reference for workflow chaining

### Matching Outputs

- [x] HDF5 export (`matches.h5`) with matches, mkpts0/1, mscores per pair
- [x] Matching manifest JSON with config, pair stats, environment
- [x] Unified environment info (CPU, GPU, package versions including imm)
- [x] Stage identifier and parent manifest reference for workflow chaining

### Estimation Outputs

- [x] Estimation manifest JSON with config, stats, environment
- [x] Output file size tracking in manifests
- [x] Unified environment info (CPU, GPU, package versions including imm)
- [x] Stage identifier and parent manifest reference for workflow chaining

---

## Model Optimization & Export

- [ ] ONNX export for production inference
- [ ] TorchScript export for C++ deployment
- [ ] Quantization strategies (int8, fp16) with validation
- [ ] Mixed precision inference (AMP)
- [ ] TensorRT optimization for NVIDIA GPUs

---

## API & Configuration

- [ ] Add device/dtype/seed recording for all runs

---

## Utilities & Tools

- [ ] Pair generator utilities (exhaustive, sequential, geometric)
- [ ] Image quality assessment (blur detection, exposure)
- [ ] Result merger for distributed runs
- [ ] HDF5 inspection tool (CLI for querying contents)

---

## Docker

- [x] Dockerfile for GPU-enabled image
- [ ] Docker Compose for UI + API server

---

## Examples

- [x] Example scripts (`demos/`)
- [ ] Example notebooks demonstrating end-to-end workflows
- [ ] Example scripts for common use cases (registration)
- [ ] Tutorial notebooks (extraction, matching, pose estimation)

---

## Testing & Quality

- [ ] Unit tests for all estimators (homography, PnP, pose)
- [ ] Integration tests for extract → match → estimate pipelines
- [ ] Add pytest fixtures for test data
- [ ] CI/CD pipeline with automated testing
