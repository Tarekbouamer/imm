# import torch
# from loguru import logger

# from imm.registry.register import ModelRegistry


def _cfg(**kwargs):
    return {**kwargs}


def extend_keys_with_suffix(data, suffix="0"):
    return {k + suffix: v for k, v in data.items()}


# # def create_extractor(name, cfg=None, pretrained=True, **kwargs):
# #     logger.info(f"Creating extractor {name}")

# #     print(EXTRACTORS_REGISTRY.list_models)

# #     # check if the model available
# #     if EXTRACTORS_REGISTRY.is_model(name) is False:
# #         raise ValueError(f"Extractor {name} is not available")

# #     print(EXTRACTORS_REGISTRY)

# #     # get the model
# #     model = EXTRACTORS_REGISTRY.create_model(name, cfg=cfg, pretrained=pretrained, **kwargs)

# #     return model


# def create_matcher(name, cfg=None, pretrained=True, **kwargs):
#     logger.info(f"Creating matcher {name}")

#     # check if the model available
#     if MATCHERS_REGISTRY.is_model(name) is False:
#         raise ValueError(f"Matcher {name} is not available")

#     # get the model
#     model = MATCHERS_REGISTRY.create_model(name, cfg=cfg, pretrained=pretrained, **kwargs)

#     return model


# def scale_keyppoints_to_original(data: dict) -> dict:
#     keypoints0 = []
#     keypoints1 = []

#     for kps0, kps1, scale0, scale1 in zip(
#         data["kpts0"], data["kpts1"], data["scale0"], data["scale1"]
#     ):  # noqa: E501
#         kps0 = kps0 * scale0
#         kps1 = kps1 * scale1

#         keypoints0.append(kps0)
#         keypoints1.append(kps1)

#     data["kpts0"] = keypoints0
#     data["kpts1"] = keypoints1

#     return data


# def make_mutual(data: dict):
#     # mutuals
#     mkpts0 = []
#     mkpts1 = []
#     matches_scores = []

#     for kps0, kps1, matches, scores in zip(
#         data["kpts0"], data["kpts1"], data["matches"], data["matches_scores"]
#     ):  # noqa: E501
#         #
#         valid = torch.where(matches != -1)[0]

#         # valid matches
#         mkpts0.append(kps0[valid])
#         mkpts1.append(kps1[matches[valid]])
#         matches_scores.append(scores[valid])

#     data["mkpts0"] = mkpts0
#     data["mkpts1"] = mkpts1
#     data["matches_scores"] = matches_scores

#     return data
