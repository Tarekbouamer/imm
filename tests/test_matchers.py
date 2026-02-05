import torch

from imm.extractors._helper import create_extractor
from imm.matchers._helper import create_matcher
from tests.conftest import MATCHERS_LIST
from imm.utils.device import detect_device, to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.warnings import suppress_warnings


def extract_features(extractor, image, suffix, device="cpu"):
    """Extract features"""
    extractor = create_extractor(extractor)
    extractor.eval().to(device)
    preds = extractor.extract({"image": image})

    # Get image size
    h, w = image.shape[-2:]
    size = torch.tensor([h, w])

    # Suffix
    preds = {f"{k}{suffix}": v for k, v in preds.items() if k in [
        "kpts", "desc", "scores"]}
    preds[f"size{suffix}"] = size

    # Flatten
    for key, value in preds.items():
        if not isinstance(value, torch.Tensor):
            preds[key] = torch.stack(value)

    return preds


def match_features(matcher, extractor, image0, image1, device="cpu"):
    """Match features of pair of images"""
    matcher = create_matcher(matcher, pretrained=True)
    matcher.eval().to(device)

    # Check if the matcher requires direct image input or pre-extracted features
    if "image0" in matcher.required_inputs:
        m_input = {"image0": image0, "image1": image1}
        matches = matcher.match(m_input)
    else:
        # Create extractor and extract features if required by matcher
        features0 = extract_features(extractor, image0, "0", device)
        features1 = extract_features(extractor, image1, "1", device)
        matches = matcher.match({**features0, **features1})

    return to_numpy(matches)


def validate_matches(preds):
    """Validate the extracted matches."""
    matches = preds["matches"]
    mscores = preds["mscores"]

    kpts0 = preds["kpts0"]
    kpts1 = preds["kpts1"]

    mkpts0 = preds["mkpts0"]
    mkpts1 = preds["mkpts1"]

    # Check if the matches are valid
    assert matches is not None and len(matches) > 1, "Invalid matches"
    assert (
        len(matches) == len(mscores) == len(kpts0)
    ), f"Mismatch between matches: {len(matches)}, mscores: {len(mscores)}, and kpts0: {len(kpts0)}"

    assert len(mkpts0) == len(
        mkpts1), f"Mismatch between mkpts0: {len(mkpts0)} and mkpts1: {len(mkpts1)}"

    for k, v in preds.items():
        print(k, v.shape)


@suppress_warnings()
def test_all_registered_matchers():
    """Test all matchers provided in the TEST_MATCHERS list."""
    # Device selection
    device = detect_device()

    # N
    N = len(MATCHERS_LIST)
    print(f"Running tests for {N} matchers")

    for it, test_case in enumerate(MATCHERS_LIST):
        print(
            f"{it+1}/{N} - Testing matcher: {test_case.matcher} with extractor: {test_case.extractor}")
        try:
            # Load images
            image0 = load_image_tensor(test_case.img0, 640)[0].to(device)
            image1 = load_image_tensor(test_case.img1, 640)[0].to(device)

            # Match features
            preds = match_features(
                test_case.matcher, test_case.extractor, image0, image1, device=device)

            # Validate matches
            validate_matches(preds)
            print(f"Matcher {test_case.matcher} passed\n")

        except Exception as e:
            print(f"Matcher {test_case.matcher} failed: {e}")
            raise

    print("All matchers tested successfully")


if __name__ == "__main__":
    test_all_registered_matchers()
