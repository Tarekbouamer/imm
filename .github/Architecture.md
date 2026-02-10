# IMM Architecture

## Block-by-Block Architecture (Inner to Outer)

### Layer 1: Core Models (Innermost)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Core Models & Algorithms                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PyTorch Models (Pretrained Neural Networks)                        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Feature Extractors                                           │   │
│  │  ├─ SuperPoint (keypoint + descriptor)                      │   │
│  │  ├─ D2Net (dense detection)                                 │   │
│  │  ├─ R2D2 (rotation invariant)                               │   │
│  │  ├─ DISK (scale-invariant)                                  │   │
│  │  ├─ XFeat (efficient)                                       │   │
│  │  └─ CAPS (context-aware)                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Feature Matchers                                             │   │
│  │  ├─ SuperGlue (learned matching)                            │   │
│  │  ├─ LoFTR (local feature transformer)                       │   │
│  │  ├─ LightGlue (efficient learner)                           │   │
│  │  ├─ AsPN (anchor-based sparse)                              │   │
│  │  ├─ MatchFormer (transformer-based)                         │   │
│  │  └─ DKM (dense keypoint matching)                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Geometric Estimators                                         │   │
│  │  ├─ OpenCV (solvePnP, findHomography, findEssentialMat)    │   │
│  │  ├─ PoseLib (robust estimation library)                     │   │
│  │  └─ PyColmap (structure-from-motion backend)                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 2: Base Classes & Interfaces

```
                              ↑ wraps
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                     Base Classes & Contracts                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ModelBase (Foundation)                                             │
│    ├─ Common interface for all models                              │
│    ├─ Device management                                            │
│    ├─ Forward pass wrapper                                         │
│    └─ Inference mode context manager                               │
│                                                                      │
│  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │ FeatureModel             │  │ Output Schema:                 │  │
│  │ (extract interface)      │  │  {                             │  │
│  │                          │  │    'keypoints': tensor,        │  │
│  │ Contract:                │  │    'descriptors': tensor,      │  │
│  │  extract(image)          │  │    'scores': tensor,           │  │
│  │  → (kpts, desc, scores)  │  │    'scales': tensor (opt)      │  │
│  │                          │  │  }                             │  │
│  └──────────────────────────┘  └────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │ MatcherModel             │  │ Output Schema:                 │  │
│  │ (match interface)        │  │  {                             │  │
│  │                          │  │    'matches': indices,         │  │
│  │ Contract:                │  │    'match_confidence': float,  │  │
│  │  match(kpts1, desc1,     │  │    'scores': tensor (opt)      │  │
│  │         kpts2, desc2)    │  │  }                             │  │
│  │  → matches               │  │                                │  │
│  │                          │  │ Input Modes:                   │  │
│  │ required_inputs:         │  │  • sparse: kpts + desc         │  │
│  │  - 'sparse' or 'dense'   │  │  • dense: raw images           │  │
│  └──────────────────────────┘  └────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │ EstimatorModel           │  │ Output Schema:                 │  │
│  │ (estimate interface)     │  │  {                             │  │
│  │                          │  │    'transform': matrix,        │  │
│  │ Contract:                │  │    'success': bool,            │  │
│  │  estimate(matches,       │  │    'inliers': boolean mask,    │  │
│  │           points1,       │  │    'metrics': dict (opt)       │  │
│  │           points2)       │  │  }                             │  │
│  │  → (pose, success)       │  │                                │  │
│  │                          │  │ Backends:                      │  │
│  │ Always return:           │  │  • opencv                      │  │
│  │  - success flag          │  │  • poselib                     │  │
│  │  - metrics               │  │  • pycolmap                    │  │
│  └──────────────────────────┘  └────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 3: Registry & Factory System

```
                              ↑ uses
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    Registry & Factory System                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model Registration & Discovery                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ @register_extractor decorator                              │    │
│  │ @register_matcher decorator                                │    │
│  │ @register_estimator decorator                              │    │
│  │                                                            │    │
│  │ Central Registry:                                          │    │
│  │  extractors = {'superpoint': SuperPoint, ...}             │    │
│  │  matchers = {'superglue': SuperGlue, ...}                 │    │
│  │  estimators = {'homography': HomographyEstimator, ...}    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Factory Methods & Model Loading                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ load_model_weights(model_name, device)                     │    │
│  │   ├─ Check model hub cache                                 │    │
│  │   ├─ Download if missing (from gdown/hub)                 │    │
│  │   ├─ Load to device (GPU/CPU)                             │    │
│  │   └─ Return initialized model                              │    │
│  │                                                            │    │
│  │ get_model_config(model_name)                              │    │
│  │   ├─ Returns model configuration                           │    │
│  │   ├─ Input/output specs                                    │    │
│  │   └─ Capability registry                                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Weight Caching System                                             │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ hub/ directory structure:                                   │    │
│  │  ├─ superpoint.pth                                         │    │
│  │  ├─ superglue_indoor.pth                                   │    │
│  │  ├─ superglue_outdoor.pth                                  │    │
│  │  ├─ loftr_indoor_ds.pth                                    │    │
│  │  └─ [...all pretrained weights]                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 4: Pipeline Processing

```
                              ↑ orchestrates
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    Feature Pipeline Processors                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Extraction Pipeline                                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Extractor class:                                           │    │
│  │  ├─ Load model via factory                                │    │
│  │  ├─ Process single image                                  │    │
│  │  ├─ Batch processing with DataLoader                      │    │
│  │  ├─ Normalize outputs to schema                           │    │
│  │  └─ Export features (HDF5, JSON)                          │    │
│  │                                                            │    │
│  │ Methods:                                                  │    │
│  │  extract(image) → features dict                           │    │
│  │  extract_dataset(dataset) → batch results                 │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Matching Pipeline                                                 │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Matcher class:                                             │    │
│  │  ├─ Detect sparse/dense mode                              │    │
│  │  ├─ Route to appropriate matcher backend                  │    │
│  │  ├─ Filter matches by threshold                           │    │
│  │  ├─ Apply cross-check if enabled                          │    │
│  │  └─ Export matches (JSON, HDF5)                           │    │
│  │                                                            │    │
│  │ Methods:                                                  │    │
│  │  match(img1, img2) → matches dict                         │    │
│  │  match_sequence(image_list) → pairwise matches            │    │
│  │  match_features(kpts1, desc1, kpts2, desc2) → matches    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Estimation Pipeline                                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Estimator class:                                           │    │
│  │  ├─ Select backend (opencv/poselib/pycolmap)             │    │
│  │  ├─ Validate match count                                  │    │
│  │  ├─ Run estimation with selected method                   │    │
│  │  ├─ Extract inliers and metrics                           │    │
│  │  └─ Export results (pose, homography, F-matrix)           │    │
│  │                                                            │    │
│  │ Methods:                                                  │    │
│  │  estimate(matches, pts1, pts2) → pose dict                │    │
│  │  estimate_batch(dataset) → results                        │    │
│  │  estimate_homography(...) → homography                    │    │
│  │  estimate_pose(...) → camera pose                         │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 5: Data Management & Storage

```
                              ↑ manages
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Management & Storage Layer                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Feature Export (FeaturesWriter)                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ HDF5 Format:                                               │    │
│  │  dataset_name/                                             │    │
│  │    ├─ image001.jpg/keypoints         → (N, 2)            │    │
│  │    ├─ image001.jpg/descriptors       → (N, D)            │    │
│  │    ├─ image001.jpg/scores            → (N,)              │    │
│  │    └─ image001.jpg/metadata.json     → config             │    │
│  │                                                            │    │
│  │ JSON Format:                                              │    │
│  │  {                                                        │    │
│  │    "image001.jpg": {                                      │    │
│  │      "keypoints": [[x1,y1], ...],                        │    │
│  │      "descriptors": [...],                                │    │
│  │      "scores": [...]                                      │    │
│  │    }                                                       │    │
│  │  }                                                        │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Match Export (MatchesWriter)                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Synchronous Writer:                                        │    │
│  │  - Real-time match writing                                │    │
│  │  - Blocking I/O                                           │    │
│  │                                                            │    │
│  │ Asynchronous Writer:                                      │    │
│  │  - Background I/O thread                                  │    │
│  │  - Queue-based buffering                                  │    │
│  │  - Non-blocking pipeline                                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Manifest System (imm/utils/manifest.py)                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Pure Function API:                                         │    │
│  │  - create_extraction_manifest()                            │    │
│  │      → Returns dict with config, stats, environment        │    │
│  │  - create_matching_manifest()                              │    │
│  │      → Returns dict with matcher config, pair stats        │    │
│  │  - create_estimation_manifest()                            │    │
│  │      → Returns dict with success rates, inlier stats       │    │
│  │  - save_manifest(manifest_dict, output_path)               │    │
│  │      → Saves to <output>_manifest.json                     │    │
│  │                                                            │    │
│  │ Manifest Contents:                                         │    │
│  │  - Model config (extractor/matcher/estimator)             │    │
│  │  - Processing stats (avg keypoints, matches, inliers)      │    │
│  │  - Environment info (Python, PyTorch, CUDA, GPU)           │    │
│  │  - Errors and failures                                     │    │
│  │  - Throughput metrics (images/sec, pairs/sec)              │    │
│  │                                                            │    │
│  │ Usage in CLI:                                             │    │
│  │  manifest = create_extraction_manifest(...)               │    │
│  │  save_manifest(manifest, output_path)                      │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  HDF5 Readers (Datasets)                                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Thread-safe HDF5 access:                                   │    │
│  │  - Lazy open per thread in dataset readers                 │    │
│  │  - Reuse handles within thread for __getitem__ calls       │    │
│  │  - Avoid cross-thread file handle sharing                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Caching & Performance                                             │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Feature Cache (optional):                                  │    │
│  │  - LMDB backend for fast access                            │    │
│  │  - SQLite for structured queries                           │    │
│  │                                                            │    │
│  │ Model Cache:                                              │    │
│  │  - GPU memory cache for frequently used models             │    │
│  │  - LRU eviction policy                                     │    │
│  │  - Automatic model offloading                              │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 6: Core Services & Infrastructure

```
                              ↑ provides
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                  Core Services & Infrastructure                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Device Management                                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ detect_device():                                           │    │
│  │  ├─ Check CUDA availability                               │    │
│  │  ├─ Check GPU memory                                      │    │
│  │  ├─ Fallback to CPU                                       │    │
│  │  └─ Return optimal device string                          │    │
│  │                                                            │    │
│  │ Device routing:                                           │    │
│  │  - GPU: cuda:0, cuda:1, ...                               │    │
│  │  - CPU: cpu                                               │    │
│  │  - MPS (Apple): mps                                       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Parallelization                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ DataLoader-based parallelization:                         │    │
│  │  - num_workers: parallel data loading                      │    │
│  │  - batch_size: tunable batch processing                    │    │
│  │  - prefetch_factor: reduce I/O blocking                    │    │
│  │                                                            │    │
│  │ Progress tracking:                                        │    │
│  │  - tqdm integration                                       │    │
│  │  - ETA calculation                                        │    │
│  │  - Real-time statistics                                   │    │
│  │                                                            │    │
│  │ Future: Ray, Celery for distributed processing            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Metrics & Monitoring                                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Supported metrics:                                         │    │
│  │  ├─ AUC@5°, AUC@10°, AUC@20° (pose accuracy)             │    │
│  │  ├─ Inlier ratio (RANSAC inliers / total matches)         │    │
│  │  ├─ Epipolar error (match quality)                        │    │
│  │  ├─ Recall (ground truth matches found)                   │    │
│  │  └─ Precision (valid matches / total matches)             │    │
│  │                                                            │    │
│  │ Logging:                                                  │    │
│  │  - Structured JSON logging                                │    │
│  │  - Model performance metadata                             │    │
│  │  - Runtime statistics                                     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Error Handling & Reliability                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Success flags in all outputs:                             │    │
│  │  - Estimation always returns success boolean              │    │
│  │  - Graceful degradation (fallback backends)               │    │
│  │                                                            │    │
│  │ Edge case handling:                                       │    │
│  │  - Insufficient keypoints                                 │    │
│  │  - Degenerate geometry                                    │    │
│  │  - No valid matches                                       │    │
│  │                                                            │    │
│  │ Resumable operations:                                     │    │
│  │  - Manifest-based progress tracking                       │    │
│  │  - Skip already processed items                           │    │
│  │  - Checkpoint recovery                                    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 7: API Layer (FastAPI/gRPC)

```
                              ↑ exposes
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                      API Servers & Endpoints                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  REST API (FastAPI)                                                 │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ POST /extract                                              │    │
│  │  ├─ Input: image (bytes), model name (string)             │    │
│  │  ├─ Processing: Extractor pipeline                         │    │
│  │  └─ Output: {keypoints, descriptors, scores}              │    │
│  │                                                            │    │
│  │ POST /match                                               │    │
│  │  ├─ Input: image1, image2, model name                     │    │
│  │  ├─ Processing: Extract + Match pipelines                 │    │
│  │  └─ Output: {matches, confidence}                         │    │
│  │                                                            │    │
│  │ POST /estimate                                            │    │
│  │  ├─ Input: matches, points1, points2, method              │    │
│  │  ├─ Processing: Estimation pipeline                       │    │
│  │  └─ Output: {pose, success, inliers, metrics}            │    │
│  │                                                            │    │
│  │ GET /health                                               │    │
│  │  └─ Output: {status, gpu_available, models_loaded}       │    │
│  │                                                            │    │
│  │ GET /metrics                                              │    │
│  │  └─ Output: {requests_count, avg_latency, models_info}   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  gRPC API (High-Performance)                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ service ImageMatching {                                    │    │
│  │   rpc Extract(ImageRequest) returns (Features);            │    │
│  │   rpc Match(MatchRequest) returns (Matches);               │    │
│  │   rpc Estimate(EstimationRequest) returns (Pose);          │    │
│  │   rpc StreamExtract(stream Image) returns (stream Feat);   │    │
│  │ }                                                          │    │
│  │                                                            │    │
│  │ Benefits:                                                 │    │
│  │  - Binary protocol (faster than JSON)                     │    │
│  │  - Streaming support                                      │    │
│  │  - Bidirectional communication                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Request/Response Validation                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Pydantic Models (type safety):                             │    │
│  │  - ImageRequest validation                                │    │
│  │  - MatchRequest validation                                │    │
│  │  - Response schema enforcement                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 8: Visualization & UI

```
                              ↑ displays
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    Visualization & UI Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Gradio Web Interface (Primary UI)                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Components:                                                │    │
│  │  ├─ Image Upload                                          │    │
│  │  ├─ Model Selection Dropdown                              │    │
│  │  ├─ Parameter Sliders                                     │    │
│  │  ├─ Real-time Preview                                     │    │
│  │  └─ Result Display                                        │    │
│  │                                                            │    │
│  │ Tabs:                                                     │    │
│  │  1. Extract: Show keypoints overlay on image              │    │
│  │  2. Match: Show matching lines between images             │    │
│  │  3. Estimate: Show transformation applied                 │    │
│  │  4. Gallery: Browse previous results                      │    │
│  │                                                            │    │
│  │ Features:                                                 │    │
│  │  - Live preview of extracted keypoints                    │    │
│  │  - Confidence score thresholding                          │    │
│  │  - Model comparison                                       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Matplotlib Visualizers                                             │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ KeypointVisualizer:                                        │    │
│  │  ├─ Draw keypoints as circles on image                    │    │
│  │  ├─ Color by scale/confidence                             │    │
│  │  └─ Export as PNG/PDF                                     │    │
│  │                                                            │    │
│  │ MatchVisualizer:                                          │    │
│  │  ├─ Draw match lines between images                       │    │
│  │  ├─ Color inliers vs outliers                             │    │
│  │  └─ Epipolar line overlay (optional)                      │    │
│  │                                                            │    │
│  │ Future:                                                   │    │
│  │  ├─ Homography warp overlay                               │    │
│  │  ├─ Epipolar geometry visualization                       │    │
│  │  └─ Report generation (HTML/PDF)                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Rerun Visualization (3D)                                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Camera Pose Visualization:                                │    │
│  │  ├─ 3D camera trajectory                                  │    │
│  │  ├─ Frustum visualization                                 │    │
│  │  └─ Point cloud from triangulation                        │    │
│  │                                                            │    │
│  │ Debugging:                                                │    │
│  │  - Keypoint tracks across frames                          │    │
│  │  - Match correspondences in 3D                            │    │
│  │  - Outlier detection visualization                        │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 9: CLI Tools

```
                              ↑ commands
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                        Command-Line Tools                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  imm-extract (Single/Dataset Extraction)                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ $ imm-extract image.jpg --model superpoint --device cuda   │    │
│  │ $ imm-extract dataset/ --model superpoint --output feat.h5 │    │
│  │                                                            │    │
│  │ Options:                                                  │    │
│  │  --model: Extractor name (superpoint, disk, etc.)         │    │
│  │  --max_keypoints: Maximum keypoints per image             │    │
│  │  --resize: Max image dimension                            │    │
│  │  --output: HDF5 output path                               │    │
│  │  --batch_size, --num_workers: DataLoader parallelization  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  imm-match (Click Group with 3 Subcommands)                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Subcommand: pair                                           │    │
│  │   Match single image pair                                 │    │
│  │   $ imm-match pair img0.jpg img1.jpg \                     │    │
│  │       --matcher superglue_outdoor \                        │    │
│  │       --extractor superpoint \                             │    │
│  │       --output output/ --show                              │    │
│  │                                                            │    │
│  │ Subcommand: images                                        │    │
│  │   Match multiple image pairs from directory               │    │
│  │   $ imm-match images assets/phototourism/ \               │    │
│  │       --pairs pairs.txt \                                  │    │
│  │       --matcher loftr_outdoor_ds \                         │    │
│  │       --output matches.h5 \                                │    │
│  │       --batch_size 4 --num_workers 4                       │    │
│  │                                                            │    │
│  │ Subcommand: features                                      │    │
│  │   Match from pre-extracted features                       │    │
│  │   $ imm-match features output/features.h5 \               │    │
│  │       --pairs pairs.txt \                                  │    │
│  │       --matcher superglue_outdoor \                        │    │
│  │       --output matches.h5 --batch_size 8                   │    │
│  │                                                            │    │
│  │ Common Options:                                           │    │
│  │  --matcher: Matcher name (sparse: superglue, lightglue;   │    │
│  │             dense: loftr, matchformer, aspanformer, dkm)  │    │
│  │  --extractor: Extractor for sparse matchers (optional     │    │
│  │               for pair/images, not used in features)       │    │
│  │  --match_thd: Match score threshold                        │    │
│  │  --resize: Max image dimension (pair/images only)          │    │
│  │  --output: Directory for matches.h5 + manifest.json        │    │
│  │                                                            │    │
│  │ Pairs File Format (space-separated):                      │    │
│  │   image0.jpg image1.jpg                                   │    │
│  │   image2.jpg image3.jpg                                   │    │
│  │   # comments allowed                                      │    │
│  │                                                            │    │
│  │ Output Structure:                                         │    │
│  │   output/                                                 │    │
│  │     ├─ matches.h5           (HDF5 with match data)        │    │
│  │     └─ matches_manifest.json (config, stats, metadata)    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  imm-estimate (Geometric Estimation)                                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ $ imm-estimate matches.json --method homography            │    │
│  │                                                            │    │
│  │ Methods: homography, relative_pose, pnp                    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  imm-download (Model Weight Downloader)                             │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ $ imm-download --all                                       │    │
│  │ $ imm-download --name superpoint                           │    │
│  │ $ imm-download --path /custom/dir                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  imm-gui (Gradio UI)                                                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ $ imm-gui [--server 0.0.0.0:7860]                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  imm-export (Format Conversion - Stub)                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ $ imm-export features.h5 --format colmap                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  imm-optimize (Model Optimization - Stub)                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ $ imm-optimize --model superpoint --format onnx            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  imm-metrics (planned)                                              │
│  $ imm-metrics results/ --metrics auc recall                        │
│                                                                      │
│  Pipeline CLI (planned)                                             │
│  $ imm-pipeline extract match estimate dataset/ output/             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 10: Deployment & Containerization

```
                              ↑ deploys
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    Deployment & Containerization                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Docker Images                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ CPU Image:                                                 │    │
│  │  FROM python:3.10                                          │    │
│  │  ├─ PyTorch CPU build                                      │    │
│  │  ├─ OpenCV, kornia                                         │    │
│  │  └─ All extractors, matchers, estimators                   │    │
│  │                                                            │    │
│  │ GPU Image (CUDA 11.8):                                     │    │
│  │  FROM nvidia/cuda:11.8-runtime-ubuntu22.04                │    │
│  │  ├─ PyTorch CUDA build                                     │    │
│  │  ├─ cuDNN, TensorRT support                                │    │
│  │  └─ All backends + GPU optimization                        │    │
│  │                                                            │    │
│  │ Size:                                                      │    │
│  │  - CPU: ~2.5 GB                                            │    │
│  │  - GPU: ~4.5 GB                                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Docker Compose Stack                                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ services:                                                  │    │
│  │   imm-api:                                                 │    │
│  │     image: imm-gpu:latest                                  │    │
│  │     ports: [8000:8000]                                    │    │
│  │     environment: [CUDA_VISIBLE_DEVICES=0]                │    │
│  │                                                            │    │
│  │   imm-ui:                                                  │    │
│  │     image: imm-gpu:latest                                  │    │
│  │     command: imm-gui --server 0.0.0.0:7860               │    │
│  │     ports: [7860:7860]                                    │    │
│  │     depends_on: [imm-api]                                 │    │
│  │                                                            │    │
│  │   redis (optional):                                        │    │
│  │     image: redis:7                                         │    │
│  │     ports: [6379:6379]                                    │    │
│  │                                                            │    │
│  │   postgres (optional):                                     │    │
│  │     image: postgres:15                                    │    │
│  │     volumes: [db_data:/var/lib/postgresql/data]           │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Kubernetes Manifests (Future)                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Planned:                                                   │    │
│  │  - Deployment spec with replicas                          │    │
│  │  - GPU node affinity                                      │    │
│  │  - Service and Ingress                                    │    │
│  │  - StatefulSet for persistent cache                       │    │
│  │  - HPA (Horizontal Pod Autoscaling)                        │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Resource Requirements                                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Minimum:                                                   │    │
│  │  - CPU: 2 cores, RAM: 4 GB                                 │    │
│  │  - Disk: 10 GB (for model weights)                         │    │
│  │                                                            │    │
│  │ Recommended:                                              │    │
│  │  - GPU: NVIDIA A100 or RTX 4090                            │    │
│  │  - CPU: 8+ cores, RAM: 16+ GB                              │    │
│  │  - Disk: 100 GB (SSD for caching)                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Information Flow Through Layers

### Typical Request Flow (Outer → Inner)

```
HTTP Request (Layer 9: API)
    ↓
Route Handler & Validation (Layer 7)
    ↓
Call Appropriate Service (Layer 6)
    ↓
Pipeline Processor (Layer 4)
    ↓
Factory Load Model (Layer 3)
    ↓
Base Class Forward Pass (Layer 2)
    ↓
Core PyTorch Model (Layer 1)
    ↓
Return Results
    ↓
Cache Results (Layer 5)
    ↓
Serialize Response (Layer 7)
    ↓
HTTP Response (Layer 9)
```

### Data Processing Flow (Inner → Outer)

```
Raw Model Outputs (Layer 1)
    ↓
Normalize to Schema (Layer 2: FeatureModel)
    ↓
Pipeline Aggregation (Layer 4)
    ↓
Export & Serialize (Layer 5)
    ↓
Cache & Manifest (Layer 5)
    ↓
Format for API (Layer 7)
    ↓
Render in UI (Layer 8)
    ↓
Display to User (Layer 9)
```

## Architectural Principles

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Interface Contracts**: Layers communicate through well-defined interfaces (FeatureModel, MatcherModel, EstimatorModel)
3. **Extensibility**: New models can be added by implementing base classes and registering
4. **Caching Strategy**: Multi-level caching (model weights, features, results)
5. **Error Propagation**: Success flags propagate through entire pipeline
6. **Observability**: Metrics and logging at each layer
7. **Scalability**: Stateless design enables horizontal scaling
