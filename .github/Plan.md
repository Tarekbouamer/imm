# IMM Features & Roadmap - Category-Based Checklist

Tasks organized by feature category, with completion status across all work streams.

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
- [ ] Implement detector threshold (det_thd) in extraction
- [ ] Add batch_size and num_workers CLI flags for extraction
- [ ] Add config file support (YAML/JSON) for all CLI tools
- [ ] Add --dry-run flag to preview operations without execution
- [ ] Add --verbose flag for detailed logging
- [ ] Dataset-level metrics aggregation CLI (`imm-metrics`)
- [ ] Pipeline CLI for chaining extract → match → estimate
- [ ] Fundamental/essential matrix CLI commands

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
- [ ] Add compression options for HDF5 (gzip levels)
- [ ] Streaming writers for memory-constrained environments
- [ ] Append mode for incremental dataset building
- [ ] HDF5 index/metadata for fast querying

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
- [ ] Model caching to avoid repeated loading
- [ ] LRU cache for feature/match results
- [ ] Memory pool management for large batches
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
- [ ] Implement dataset-level matching from pairs file
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
- [ ] Fundamental matrix estimation (explicit estimator)
- [ ] RANSAC parameter auto-tuning based on match count
- [ ] Multi-model fitting (fit multiple hypotheses)
- [ ] Degeneracy detection (planar scenes, pure rotation)
- [ ] Robust estimation report export (inliers, stats, HTML)
- [ ] Relative pose chain solver (incremental pose tracking)
- [ ] Triangulation for 3D point reconstruction
- [ ] Bundle adjustment integration

### OpenCV Backend

- [x] Homography estimation via OpenCV
- [x] PnP estimation via OpenCV
- [x] Essential matrix estimation via OpenCV

### PoseLib Backend

- [x] Homography estimation via PoseLib
- [x] PnP estimation via PoseLib
- [x] Essential matrix estimation via PoseLib
- [ ] Robust estimation with PoseLib outlier handling

### PyColmap Backend

- [x] Pycolmap optional dependency integration
- [ ] Homography estimation via PyColmap
- [ ] PnP estimation via PyColmap
- [ ] Essential matrix estimation via PyColmap
- [ ] Colmap integration / export format support

---

## Visualization

- [x] `Viz2D` base visualization class
- [x] `KeypointVisualizer` for keypoint visualization
- [x] `MatchVisualizer` for match visualization
- [ ] Epipolar line visualization
- [ ] Homography warp preview
- [ ] Headless visualization export for CI/automation

---

## UI

### Gradio UI

- [x] Gradio UI (`imm-gui`) for interactive image matching
- [ ] Interactive parameter tuning UI
- [ ] Model selection dropdown
- [ ] Batch processing interface
- [ ] Export results from UI

### Rerun UI

- [ ] Rerun integration for debugging and 3D visualization
- [ ] Camera pose trajectory visualization
- [ ] Point cloud from triangulation
- [ ] Keypoint tracks across frames
- [ ] Match correspondences in 3D
- [ ] Outlier detection visualization

### Web UI & Reports

- [ ] Gallery auto-generation from runs (HTML/static site)
- [ ] Match visualization report export (HTML or PDF)
- [ ] Web-based result browser for extraction/matching runs

---

## Parallelization & Distribution

- [x] Parallel execution (DataLoader workers for datasets)
- [x] Progress + ETA reporting (via `tqdm`)
- [x] Resume/skip logic for extraction (read HDF5, skip already extracted images)
- [x] Error recovery in extraction (continue on single image failures, log errors)
- [x] Async image loading with prefetching for extraction (pin_memory, prefetch_factor, persistent_workers)
- [x] Async image loading with prefetching for matching (pin_memory, prefetch_factor, persistent_workers)
- [ ] Multi-GPU extraction (DataParallel/DistributedDataParallel for feature extraction)
- [ ] Memory-aware batching (dynamic batch size based on GPU memory)
- [ ] Distributed extraction/matching across multiple machines (split image list, merge outputs)
- [ ] Multiprocessing orchestration for dataset-level tasks (beyond DataLoader)
- [ ] Job queue integration (Ray, Celery, etc.)

---

## Artifacts & Output Formats

### Extraction Outputs

- [x] HDF5 export (`features.h5`) with keypoints, descriptors, scores per image
- [x] Extraction manifest JSON with config, stats, environment

### Matching Outputs

- [x] HDF5 export (`matches.h5`) with matches, mkpts0/1, mscores per pair
- [x] Matching manifest JSON with config, pair stats, environment

---

## Model Optimization & Export

- [ ] ONNX export for production inference
- [ ] TorchScript export for C++ deployment
- [ ] Quantization strategies (int8, fp16) with validation
- [ ] Model pruning
- [ ] Mixed precision training/inference (AMP)
- [ ] TensorRT optimization for NVIDIA GPUs
- [ ] Model distillation for faster inference
- [ ] Dynamic batching for variable-size inputs

---

## API & Configuration

- [ ] Provide single configuration layer (YAML/JSON) with CLI/env overrides
- [ ] Add device/dtype/seed recording for all runs

---

## Utilities & Tools

- [ ] Dataset statistics calculator (image sizes, formats, distributions)
- [ ] Pair generator utilities (exhaustive, sequential, geometric)
- [ ] Image quality assessment (blur detection, exposure)
- [ ] Dataset splitter (train/val/test)
- [ ] Result merger for distributed runs
- [ ] HDF5 inspection tool (CLI for querying contents)

---

## Monitoring & Observability

- [ ] Structured logging to JSON for monitoring/analysis
- [ ] Performance metadata per model (latency, memory)
- [ ] Real-time progress dashboard (web-based)
- [ ] Prometheus metrics exporter
- [ ] OpenTelemetry tracing integration

---

## Reliability & Caching

- [ ] Test coverage for edge cases (no matches, insufficient points, corrupted images, OOM)
- [ ] Graceful degradation for corrupted images
- [ ] Timeout handling for slow operations
- [ ] Memory overflow protection (auto-reduce batch size)
- [ ] Retry logic for transient failures
- [ ] Checkpointing for long-running jobs
- [ ] Dead letter queue for failed items

---

## Server & REST API

- [ ] REST API service with FastAPI (`/extract`, `/match`, `/estimate` endpoints)
- [ ] gRPC service for high-throughput matching
- [ ] Health and metrics endpoints for monitoring
- [ ] Request/response validation and error handling

---

## Docker

- [x] Dockerfile for GPU-enabled image
- [ ] Docker Compose for UI + API server
- [ ] Prebuilt Docker images in registry (CPU/GPU variants)

---

## Examples

- [x] Example scripts (`demos/`)
- [ ] Example notebooks demonstrating end-to-end workflows
- [ ] Example scripts for common use cases (SfM, localization, registration)
- [ ] Incremental SfM pipeline wrapper
- [ ] Tutorial notebooks (extraction, matching, pose estimation)

---

## Testing & Quality

- [ ] Unit tests for all estimators (homography, PnP, pose)
- [ ] Integration tests for extract → match → estimate pipelines
- [ ] Benchmark suite for extractors/matchers on standard datasets
- [ ] Regression tests for performance tracking
- [ ] Add pytest fixtures for test data
- [ ] Add property-based testing (Hypothesis)
- [ ] CI/CD pipeline with automated testing
- [ ] Code coverage reporting (>80% target)
- [ ] Memory leak detection tests
- [ ] Load testing for production scenarios
