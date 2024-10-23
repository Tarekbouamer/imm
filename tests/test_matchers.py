from collections import namedtuple

import numpy as np
import torch

from imm.extractors._helper import create_extractor
from imm.matchers._helper import MATCHERS_REGISTRY, create_matcher
from imm.settings import img0_path, img1_path
from imm.utils.device import to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.warnings import suppress_warnings

# Named tuple for providing test cases
emii = namedtuple('emii', ['extractor', 'matcher', 'img0', 'img1'])

# List of test cases (extractor, matcher, image0, image1)
# TODO: add all cases, meaning all extractors indoors and outdoors
# TODO: to insure a fair comparison, we need to use indoor and outdoor images depending on the matcher training data
TEST_MATCHERS = [
    # emii("superpoint", "lightglue_superpoint", img0_path, img1_path),
    # emii("disk_depth", "lightglue_disk", img0_path, img1_path),
    # emii("superpoint", "lightglue_aliked", img0_path, img1_path),
    # emii("superpoint", "lightglue_sift", img0_path, img1_path),
    emii("xfeat_sparse", "lighterglue", img0_path, img1_path),
    emii("superpoint", "nn", img0_path, img1_path),
    emii("superpoint", "superglue_indoor", img0_path, img1_path),
    emii("superpoint", "superglue_outdoor", img0_path, img1_path),
    emii("superpoint", "loftr_indoor_ds_new", img0_path, img1_path),
    emii("superpoint", "loftr_indoor_ds", img0_path, img1_path),
    emii("superpoint", "loftr_outdoor_ds", img0_path, img1_path),
    emii("superpoint", "aspanformer_indoor", img0_path, img1_path),
    emii("superpoint", "aspanformer_outdoor", img0_path, img1_path),
    emii("superpoint", "efficient_loftr", img0_path, img1_path),
    emii("superpoint", "matchformer_largela", img0_path, img1_path),
    # emii("superpoint", "matchformer_largesea", img0_path, img1_path),
    emii("superpoint", "matchformer_litela", img0_path, img1_path),
    emii("superpoint", "matchformer_litesea", img0_path, img1_path),
]


def extract_features(extractor_model, image, suffix, device="cpu"):
    """Extract features (keypoints, descriptors, and scores if available) from an image using an extractor."""
    extractor = create_extractor(extractor_model)
    extractor.eval().to(device)
    preds = extractor.extract({"image": image})

    # Get image size as height and width
    h, w = image.shape[-2:]
    size = torch.tensor([h, w])

    # Ensure we have descriptors and scores, use dummy scores if not available
    features = {
        f"kpts{suffix}": preds.get("kpts", None),
        f"desc{suffix}": preds.get("desc", None),
        f"scores{suffix}": preds.get("scores", torch.ones(len(preds.get("kpts", [])))),
        f"size{suffix}": size,
    }

    for key, value in features.items():
        if not isinstance(value, torch.Tensor):
            features[key] = torch.stack(value)

    return features


def extract_matches(matcher_model, image0, image1, extractor_model=None, device="cpu"):
    """Extract matches between two images using the specified matcher."""
    matcher = create_matcher(
        matcher_model, pretrained=True)
    matcher.eval().to(device)

    # Check if the matcher requires direct image input or pre-extracted features
    if "image0" in matcher.required_inputs:
        m_input = {"image0": image0, "image1": image1}
        matches = matcher.match(m_input)
    else:
        # Create extractor and extract features if required by matcher
        features0 = extract_features(extractor_model, image0, "0", device)
        features1 = extract_features(extractor_model, image1, "1", device)
        matches = matcher.match({**features0, **features1})

    return to_numpy(matches)


def validate_matches(matches):
    """Validate the extracted matches."""
    assert matches is not None and len(matches) > 0, "No matches found"


@suppress_warnings()
def test_all_registered_matchers():
    """Test all matchers provided in the TEST_MATCHERS list."""
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # N
    N = len(TEST_MATCHERS)
    print(f"Running tests for {N} matchers")

    # Iterate over all test cases in TEST_MATCHERS
    for it, test_case in enumerate(TEST_MATCHERS):
        print(
            f"{it+1}/{N} - Testing matcher: {test_case.matcher} with extractor: {test_case.extractor}")
        try:
            # Load images
            image0 = load_image_tensor(test_case.img0)[0].to(device)
            image1 = load_image_tensor(test_case.img1)[0].to(device)

            # Extract matches for the given matcher and extractor
            matches = extract_matches(
                test_case.matcher, image0, image1, test_case.extractor, device)

            # Validate the matches
            validate_matches(matches)
            print(f"Matcher {test_case.matcher} passed\n")

        except Exception as e:
            print(f"Matcher {test_case.matcher} failed: {e}")
            raise  # Stop the process immediately if any model fails

    print("All matchers tested successfully")
