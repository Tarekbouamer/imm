# IMM Features & Roadmap - Category-Based Checklist

Tasks organized by feature category, with completion status across all work streams.

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

---

## Download & Model Management

- [x] Automatic model weight download from hub
- [x] Model weight caching to hub/ directory
- [ ] CLI download all models (`imm-download --all`)
- [ ] CLI download specific model by name (`imm-download --name superpoint`)
- [ ] Custom download path support (default: hub/, override with `--path`)
- [ ] Download progress reporting (percentage, speed, ETA)
- [ ] Python API for downloading models

---

## Extraction

- [ ] FeatureModel contract documentation (extract method, output schema)
- [x] Extractors registry with pretrained weight loading
- [x] Dataset loader for batch extraction (`ImagesFromList`)
- [x] Dataset extraction with progress/ETA (`tqdm` in `extract_dataset`)
- [x] HDF5 feature export via `FeaturesWriter`
- [ ] Feature caching policies beyond HDF5 (LMDB, SQLite)
- [ ] Benchmark script for extractor speed/accuracy comparison
- [ ] Batch extraction progress resume/restart
- [ ] Distributed extraction/matching across multiple machines

---

## Matching

- [ ] MatcherModel contract documentation (match method, required_inputs)
- [x] Sparse and dense matcher support via `required_inputs`
- [x] Matcher registry with pretrained weight loading
- [x] Match filtering by score threshold
- [x] Dataset/list-of-pairs matching with progress/ETA (`match_sequence_images`, `match_sequence_features`)
- [x] Match export via writers (`MatchesWriter`, `AsycMatchesWriter`)
- [ ] Cross-check / mutual consistency options exposed in CLI
- [ ] Batch matching benchmarks and metrics reports
- [ ] Match visualization report export (HTML or PDF)
- [ ] gRPC service for high-throughput matching
- [ ] Loop closure detection module

---

## Estimation

- [ ] EstimatorModel contract documentation (estimate method, success flag)
- [x] Estimator factory utilities
- [x] Success flag in estimator outputs
- [ ] Fundamental matrix estimation (explicit estimator)
- [ ] Fundamental/essential matrix CLI commands
- [ ] Batch estimation over datasets with summary metrics
- [ ] Robust estimation report export (inliers, stats, HTML)
- [ ] Relative pose chain solver (incremental pose tracking)

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

- [x] PyColmap optional dependency integration
- [ ] Homography estimation via PyColmap
- [ ] PnP estimation via PyColmap
- [ ] Essential matrix estimation via PyColmap
- [ ] Colmap integration / export format support

---

## Visualization

### Gradio

- [x] Keypoint visualization (`KeypointVisualizer`)
- [x] Match visualization (`MatchVisualizer`)
- [ ] Match visualization report export (HTML or PDF)
- [ ] Gallery auto-generation from runs

### Matplotlib

- [ ] Epipolar line visualization
- [ ] Homography warp preview

### Rerun

- [ ] Headless visualization export for CI/automation

---

## Parallelization & Distribution

- [x] Parallel execution (DataLoader workers for datasets)
- [x] Progress + ETA reporting (via `tqdm`)
- [ ] Multiprocessing orchestration for dataset-level tasks (beyond DataLoader)
- [ ] Distributed extraction/matching across multiple machines
- [ ] Job queue integration (Ray, Celery, etc.)
- [ ] Progress tracking and fault recovery for long-running jobs

---

## Export & Data Management

- [x] HDF5 export for extracted features
- [x] Match export via writers (`MatchesWriter`, `AsycMatchesWriter`)
- [ ] Run manifest generation (JSON)
- [ ] Standardized JSON export for matches and estimations
- [ ] Central model zoo manifest (YAML/JSON) with versioning
- [ ] Run management system (query, replay, export)

---

## Model Optimization & Export

- [ ] ONNX export for production inference
- [ ] TorchScript export for C++ deployment
- [ ] Quantization strategies (int8, fp16) with validation
- [ ] Model pruning guidelines per device (mobile, edge, server)

---

## API & Configuration

- [ ] Define a stable public API module (`imm/api/`) with consistent extract/match/estimate interfaces
- [ ] Standardize output schemas for all operations (JSON + numpy arrays)
- [ ] Add request/response models (Pydantic) for type safety and validation
- [ ] Add unified error/exception taxonomy for predictable failure handling
- [ ] Clean public Python API layer with documentation
- [ ] Introduce run manifest (JSON) capturing config, device, dtype, seed, model versions
- [ ] Provide single configuration layer (YAML/JSON) with CLI/env overrides
- [ ] Add device/dtype/seed recording for all runs
- [ ] Add runtime configuration profiles (CPU/GPU presets, batch size hints)

---

## Monitoring & Observability

- [ ] Standard metric output format (AUC, pose errors, epipolar distance, recall, inlier ratio)
- [ ] Structured logging to JSON for monitoring/analysis
- [ ] Performance metadata per model (latency, memory, batch size hints)

---

## Reliability & Caching

- [ ] Add dataset-level resume/skip logic based on manifest + output checksums
- [ ] Add capability registry documenting sparse/dense, min/max input size, required inputs
- [ ] Batch extraction/matching progress resume/restart
- [ ] Test coverage for edge cases (no matches, insufficient points, etc.)

---

## Server & REST API

- [ ] REST API service with FastAPI (`/extract`, `/match`, `/estimate` endpoints)
- [ ] gRPC service for high-throughput matching
- [ ] Health and metrics endpoints for monitoring
- [ ] Request/response validation and error handling

---

## Deployment & Infrastructure

- [x] Dockerfile for GPU-enabled image
- [ ] Rerun integration for visualization and debugging

---

## Docker

- [ ] Docker Compose for UI + API server
- [ ] Prebuilt Docker images in registry (CPU/GPU variants)

---

## CLI Tools

- [x] CLI feature extraction (`imm-extract`) for single image and datasets
- [x] CLI matching (`imm-match`) for image pairs
- [x] CLI estimation (`imm-estimate`) for homography and relative pose
- [x] Gradio UI (`imm-gui`)
- [ ] Dataset-level metrics aggregation CLI (`imm-metrics`)
- [ ] Pipeline CLI for chaining extract → match → estimate

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
- [ ] Benchmark suite comparing extractors/matchers on standard datasets
- [ ] Regression tests for performance tracking
