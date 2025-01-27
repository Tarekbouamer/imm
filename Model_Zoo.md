# Model Zoo

ImMatch supports a wide range of feature extractors, matchers, and geometric estimators. Here's an overview:

## Extractors

### Keypoint Extractors

| Model         | Variants | Training Datasets | Links|
|-------------- |----------|-------------------|------|
| Caps          |`caps_sp` | MegaDepth         | [Paper](https://arxiv.org/pdf/2004.13324) / [Code](https://github.com/qianqianwang68/caps) |
| D2-Net        |`d2net_ots` `d2net_tf` `d2net_tf_no_phototourism`   | MegaDepth | [Paper](https://arxiv.org/abs/1905.03561) / [Code](https://github.com/mihaidusmanu/d2-net) |
| DISK          |`disk_depth` `disk_epipolar` | MegaDepth | [Paper](https://arxiv.org/abs/2006.13566) / [Code](https://github.com/cvlab-epfl/disk) |
| R2D2          |`r2d2_WASF_N16` `r2d2_WASF_N8_big` `r2d2_WAF_N16` `faster2d2_WASF_N16` `faster2d2_WASF_N8_big`    | 1M Oxford and Paris Revisited         | [Paper](https://arxiv.org/abs/1906.06195) / [Code](https://github.com/naver/r2d2) |
| Superpoint    |`superpoint` | COCO-MS | [Paper](https://arxiv.org/abs/1712.07629) / [Code](https://github.com/magicleap/SuperPointPretrainedNetwork) |
| XFeat         |`xfeat_spare` `xfeat_dense` | ScanNet / MegaDepth | [Paper](https://arxiv.org/abs/2404.19174) / [Code](https://github.com/verlab/accelerated_features) |

### Line Segment Extractors

| Model         | Variants | Training Datasets | Links|
|-------------- |----------|-------------------|------|
| LSD           |`lsd`     | NA         | [Paper](https://www.ipol.im/pub/art/2012/gjmr-lsd/) / [Python](https://github.com/iago-suarez/pytlsd) |
| DeepLSD       | `deep_lsd` | Wireframe/ MegaDepth  | [Paper](https://arxiv.org/abs/2212.07766) / [Code](https://github.com/cvg/DeepLSD) |

### Wireframe Extractors

| Model         | Variants | Training Datasets | Links|
|-------------- |----------|-------------------|------|
| Wireframes     |`sp_lsd_wireframe` | MegaDepth | [Paper](https://arxiv.org/pdf/2304.02008) / [Code](https://github.com/cvg/glue-factory/tree/main) |

## Matchers

### Keypoint Matchers

| Model         | Variants      | Training Datasets | Links|
|-------------- |---------------|-------------------|------|
| Aspanformer  |`aspanformer_indoor` `aspanformer_outdoor` | ScanNet / MegaDepth| [Paper](https://arxiv.org/abs/2208.14201) / [Code](https://github.com/apple/ml-aspanformer)|
| Efficient LoFTR |`efficient_loftr_indoor_ds` `efficient_loftr_outdoor_ds` | ScanNet / MegaDepth | [Paper](https://arxiv.org/abs/2403.04765) / [Code](https://github.com/zju3dv/efficientloftr) |
| Lightglue    |`lightglue_disk` `lightglue_aliked` `lightglue_sift` | MegaDepth | [Paper](https://arxiv.org/abs/2404.19174) / [Code](https://github.com/cvg/LightGlue) |
| Lighterglue  |`lighterglue`  | ScanNet / MegaDepth | [Paper](https://arxiv.org/abs/2404.19174) / [Code](https://github.com/verlab/accelerated_features/blob/main/README.md) |
| LoFTR        |`loftr_indoor_ds_new` `loftr_indoor_ds` `loftr_outdoor_ds` | ScanNet / MegaDepth | [Paper](https://arxiv.org/pdf/2104.00680) / [Code](https://github.com/zju3dv/LoFTR) |
| Matchformer  |`matchformer_largela` `matchformer_largesea` `matchformer_litela` `matchformer_litesea` | ScanNet / MegaDepth | [Paper](https://arxiv.org/pdf/2203.09645) / [Code](https://github.com/jamycheung/MatchFormer) |
| NN           |`nn`           | NA         | NA |
| Superglue    |`superglue_indoor` `superglue_outdoor` | ScanNet / MegaDepth | [Paper](https://arxiv.org/abs/1911.11763) / [Code](https://github.com/magicleap/SuperGluePretrainedNetwork) |

### Line Segment Matchers

| Model         | Variants      | Training Datasets | Links|
|-------------- |---------------|-------------------|------|
| GlueStick       | `gluestick` | MegaDepth | [Paper](https://arxiv.org/pdf/2304.02008) / [Code](https://github.com/cvg/glue-factory/tree/main)|

(For more visualization on the supported algorithms, check out the [Gallery](Gallery.md).)

## Estimators

| Estimator          | PoseLib          | PyColmap         | OpenCV           |
|--------------------|:----------------:|:----------------:|:----------------:|
| Fundamental Matrix |:white_check_mark:|:white_check_mark:|:white_check_mark:|
| Relative Pose      |:white_check_mark:|:white_check_mark:|:white_check_mark:|
| Homography         |:white_check_mark:|:white_check_mark:|:white_check_mark:|
| PnP                |:white_check_mark:|:white_check_mark:|:white_check_mark:|
