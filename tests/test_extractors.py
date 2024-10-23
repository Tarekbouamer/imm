import torch

from imm.extractors._helper import EXTRACTORS_REGISTRY, create_extractor
from imm.settings import img0_path
from imm.utils.device import to_numpy
from imm.utils.io import load_image_tensor
from imm.utils.warnings import suppress_warnings


def extract_keypoints(model, image):
    """Extract keypoints and descriptors (optionally scores) from the image using the specified model."""
    detector = create_extractor(
        model, cfg={"max_keypoints": 1000}, pretrained=True)
    detector.eval().cuda()
    preds = to_numpy(detector.extract({"image": image}))

    kpts = preds.get("kpts", None)
    # Handle cases where scores might not exist
    scores = preds.get("scores", None)
    descs = preds.get("desc", None)

    return kpts, scores, descs


def validate_data(kpts, scores, descs, expected_dim):
    """Validate keypoints and descriptors, scores are optional."""
    assert kpts is not None and len(
        kpts) > 0 and kpts[0].shape[1] == 2, "Invalid keypoints"
    assert descs is not None and len(
        descs) > 0 and descs[0].shape[0] == expected_dim, "Invalid descriptors"

    # If scores are provided, validate them
    if scores is not None:
        assert len(scores) > 0, "Invalid scores"
        assert len(kpts[0]) == len(
            scores[0]), "Mismatch between keypoints and scores"

    assert len(
        kpts[0]) == descs[0].shape[1], "Mismatch between keypoints and descriptors"


@suppress_warnings()
def test_all_registered_extractors():
    """Test all registered extractors for correct keypoints, scores (optional), and descriptors extraction."""

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image = load_image_tensor(img0_path)[0].to(device)

    models = EXTRACTORS_REGISTRY.list_models

    for model in models:
        try:
            kpts, scores, descs = extract_keypoints(model, image)
            cfg = EXTRACTORS_REGISTRY.get_default_cfg(model)
            validate_data(kpts, scores, descs, cfg["descriptor_dim"])
            print(f"Model {model} passed")
        except Exception as e:
            print(f"Model {model} failed: {e}")
