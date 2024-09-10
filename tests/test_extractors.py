# from imm.utils.device import to_numpy
# from imm.utils.io import load_image_tensor
# from imm.extractors._helper import create_extractor, EXTRACTORS_REGISTRY
# from imm.settings import img0_path
# from loguru import logger


# def extract_keypoints(model, image):
#     """Extract keypoints, scores, and descriptors from the image using the specified model."""
#     try:
#         logger.info(f"Creating extractor for model: {model}")
#         detector = create_extractor(model, cfg={"max_keypoints": 1000}, pretrained=True)
#         detector.eval().cuda()

#         logger.info("Extracting keypoints, scores, and descriptors...")
#         preds = detector.extract({"image": image})
#         preds = to_numpy(preds)

#         kpts = preds["kpts"]
#         scores = preds["scores"]
#         descs = preds["desc"]

#         logger.info(
#             f"Extracted {len(kpts[0])} keypoints, {len(scores[0])} scores, {len(descs[0])} descriptors"
#         )
#         return kpts, scores, descs
#     except Exception as e:
#         logger.error(f"Error extracting keypoints for model {model}: {e}")
#         raise


# def validate_keypoints(kpts):
#     """Validate the keypoints data."""
#     assert kpts is not None, "No keypoints found"
#     assert len(kpts) > 0, "No keypoints found"
#     assert isinstance(kpts, list), f"Keypoints should be a list, got {type(kpts)}"
#     assert kpts[0].shape[1] == 2, f"Keypoints should be (N, 2), got {kpts[0].shape}"


# def validate_scores(scores):
#     """Validate the scores data."""
#     assert scores is not None, "No scores found"
#     assert len(scores) > 0, "No scores found"
#     assert isinstance(scores, list), f"Scores should be a list, got {type(scores)}"


# def validate_descriptors(descs, expected_dim):
#     """Validate the descriptors data."""
#     assert descs is not None, "No descriptors found"
#     assert len(descs) > 0, "No descriptors found"
#     assert isinstance(descs, list), f"Descriptors should be a list, got {type(descs)}"
#     assert (
#         descs[0][0].shape[0] == descs[0][1].shape[0]
#     ), f"Descriptors should have the same dimension, got {descs[0][0].shape} and {descs[0][1].shape}"
#     assert (
#         descs[0].shape[0] == expected_dim
#     ), f"Descriptors should have {expected_dim} dimensions, got {descs[0].shape[0]}"


# def test_all_registered_extractors():
#     """Test all registered extractors for correct keypoints, scores, and descriptors extraction."""
#     logger.info("Loading image...")
#     data = load_image_tensor(img0_path)
#     image = data[0].cuda()

#     models = EXTRACTORS_REGISTRY.list_models
#     logger.info(f"Testing all models, found {len(models)} models")

#     for model in models:
#         logger.info(f"Testing model: {model}")
#         try:
#             kpts, scores, descs = extract_keypoints(model, image)
#             cfg = EXTRACTORS_REGISTRY.get_default_cfg(model)
#             validate_keypoints(kpts)
#             validate_scores(scores)
#             validate_descriptors(descs, cfg["descriptor_dim"])

#             assert len(kpts[0]) == len(scores[0]), "Number of keypoints and scores do not match"
#             assert (
#                 kpts[0].shape[0] == descs[0].shape[1]
#             ), "Number of keypoints and descriptors do not match"

#             logger.success(f"Model {model} passed")
#         except AssertionError as e:
#             logger.error(f"Validation failed for model {model}: {e}")
#         except Exception as e:
#             logger.error(f"An error occurred while testing model {model}: {e}")

#     logger.success("All models tested")
