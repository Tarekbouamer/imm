# ImMatch (IMM) 🖼️

**ImMatch (IMM)** is a versatile library for image matching and feature extraction in computer vision applications. It provides algorithms to detect, describe, and match keypoints between images, as well as estimate geometric relationships, making it ideal for tasks such as visual localization, augmented reality, and 3D reconstruction.

## Table of Contents 📑

- [Prerequisites](#prerequisites)
- [Installation 🖥️](#installation-)
- [Running with Docker 🐳](#running-with-docker-)
- [Supported Algorithms](#supported-algorithms)
  - [Extractors](#supported-extractors)
  - [Matchers](#supported-matchers)
  - [Estimators](#supported-estimators)
- [Usage](#usage)
  - [Feature Extraction](#feature-extraction)
  - [Feature Matching](#feature-matching)
  - [Geometric Estimation](#geometric-estimation)
<!-- - [Additional Information](#additional-information)
- [Acknowledgements](#acknowledgements)
- [License](#license) -->

## Prerequisites

Before installing ImMatch, ensure you have the following:

- Python 3.8 or later
- Conda
- CUDA Toolkit

## Installation 🖥️

1. Clone the Repository

```bash
git clone https://github.com/Tarekbouamer/imm.git
cd imm
```

2. Set Up a Conda Environment

```bash
conda create -n imm python=3.8 -y
conda activate imm
```

3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

4. Install ImMatch

```bash
pip install .[optional]
```

## Running with Docker 🐳

For those who prefer using Docker, we provide a Dockerfile to set up ImMatch in a containerized environment.

1. Build the Docker image:

```bash
docker build -t imm:latest .
```

2. Run the Docker container:

```bash
docker run -it --gpus all imm:latest
```

## Supported Algorithms

ImMatch supports a wide range of feature extractors, matchers, and geometric estimators. Here's an overview:

### Supported Extractors

| Extractor Name | Type | Description |
|----------------|------|-------------|
| SuperPoint     |      |             |
| D2-Net         |      |             |
| R2D2           |      |             |
| DISK           |      |             |

### Supported Matchers

| Matcher Name | Type | Description |
|--------------|------|-------------|
| SuperGlue    |      |             |
| LoFTR        |      |             |

### Supported Estimators

| Estimator Name     | Description |
|--------------------|-------------|
| Fundamental Matrix |             |
| Essential Matrix   |             |
| Homography         |             |
| PnP                |             |

## Usage

ImMatch provides command-line tools for feature extraction, matching, and geometric estimation. Here are the main commands:

### Feature Extraction

Use the `imm-extract` script to extract features from an image:

```bash
imm-extract --model MODEL_NAME --img_path PATH_TO_IMAGE --max_keypoints MAX_KEYPOINTS
```

Example:

```bash
imm-extract --model superpoint --img_path /path/to/your/image.jpg --max_keypoints 1200
```

### Feature Matching

Use the `imm-match` script to match features between two images:

```bash
imm-match IMG0_PATH IMG1_PATH [OPTIONS]
```

Options:

- `--matcher`: Matcher name (default: "superglue_outdoor")
- `--extractor`: Extractor name (default: "superpoint")
- `--max_size`: Max image size (optional)
- `--output_dir`: Output directory for logs and visualization (default: "output")
- `--threshold`: Matching score threshold (default: 0.1)
- `--visualize/--no-visualize`: Enable or disable visualization (default: True)
- `--use_gpu/--no-gpu`: Use GPU if available (default: True)

Example:

```bash
imm-match path/to/image1.jpg path/to/image2.jpg --matcher superglue_outdoor --extractor superpoint --max_size 1000 --output_dir my_results --threshold 0.2 --visualize --use_gpu
```

<!-- ### Geometric Estimation

Use the `imm-estimate` script to perform geometric estimation:

```bash
imm-estimate IMG0_PATH IMG1_PATH --estimator ESTIMATOR_NAME [OPTIONS]
```

Example:

```bash
imm-estimate path/to/image1.jpg path/to/image2.jpg --estimator ransac --model fundamental_matrix --threshold 1.0 --confidence 0.99 --max_iters 1000
```

For more detailed information on usage and available options, use the `--help` flag with each command:

```bash
imm-extract --help
imm-match --help
imm-estimate --help
``` -->

<!-- ## Additional Information

For more details on using specific extractors, matchers, and estimators, including their parameters and best practices, please refer to the [ImMatch documentation](https://github.com/Tarekbouamer/imm/docs) (replace with actual documentation link when available).

For bug reports, feature requests, or contributions, please visit the [ImMatch GitHub repository](https://github.com/Tarekbouamer/imm). -->

<!-- ## Acknowledgements

We would like to thank the open-source community and the authors of the algorithms implemented in ImMatch for their invaluable contributions to the field of computer vision.

## License

[Include license information here] -->
