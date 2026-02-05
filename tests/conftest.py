from collections import namedtuple

# Test image paths
img0_path = "assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg"
img1_path = "assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg"
img0_path_indoor = "assets/scannet/scene0783_00_480.jpg"
img1_path_indoor = "assets/scannet/scene0783_00_1530.jpg"

# Test configuration namedtuple
emii = namedtuple("emii", ["extractor", "matcher", "img0", "img1"])

# Test matrix for matchers
MATCHERS_LIST = [
    # Extractors + matchers
    emii("superpoint", "nn", img0_path, img1_path),
    emii("caps_sp", "nn", img0_path, img1_path),
    emii("d2net_tf_no_phototourism", "nn", img0_path, img1_path),
    emii("d2net_ots", "nn", img0_path, img1_path),
    emii("d2net_tf", "nn", img0_path, img1_path),
    emii("disk_depth", "nn", img0_path, img1_path),
    emii("disk_epipolar", "nn", img0_path, img1_path),
    emii("faster2d2_WASF_N8_big", "nn", img0_path, img1_path),
    emii("faster2d2_WASF_N16", "nn", img0_path, img1_path),
    emii("r2d2_WAF_N16", "nn", img0_path, img1_path),
    emii("r2d2_WASF_N8_big", "nn", img0_path, img1_path),
    emii("r2d2_WASF_N16", "nn", img0_path, img1_path),
    emii("xfeat_sparse", "nn", img0_path, img1_path),
    emii("xfeat_dense", "nn", img0_path, img1_path),
    # Superpoint + glue family
    emii("superpoint", "superglue_indoor", img0_path_indoor, img1_path_indoor),
    emii("superpoint", "superglue_outdoor", img0_path, img1_path),
    emii("superpoint", "lightglue_superpoint", img0_path, img1_path),
    emii("xfeat_sparse", "lighterglue", img0_path, img1_path),
    # Loftr family
    emii(None, "loftr_indoor_ds", img0_path_indoor, img1_path_indoor),
    emii(None, "loftr_indoor_ds_new", img0_path_indoor, img1_path_indoor),
    emii(None, "loftr_outdoor_ds", img0_path, img1_path),
    emii(None, "efficient_loftr", img0_path, img1_path),
    # Matchformer family
    emii(None, "matchformer_largela", img0_path, img1_path),
    emii(None, "matchformer_largesea", img0_path_indoor, img1_path_indoor),
    emii(None, "matchformer_litela", img0_path_indoor, img1_path_indoor),
    emii(None, "matchformer_litesea", img0_path, img1_path),
    # Aspanformer family
    emii(None, "aspanformer_indoor", img0_path_indoor, img1_path_indoor),
    emii(None, "aspanformer_outdoor", img0_path, img1_path),
]
