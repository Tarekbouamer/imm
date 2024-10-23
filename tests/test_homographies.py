# import numpy as np
# import cv2
# from loguru import logger
# from imm.estimators import (
#     CvHomographyEstimator,
#     PoseLibHomographyEstimator,
# )

# def generate_homography_test_data(num_points=1000):
#     pts0 = np.array([[141, 131], [480, 159], [493, 630], [64, 601]]).astype(np.float32)

#     H_true = np.array(
#         [[0.434, -0.419, 291.709], [0.146, 0.441, 161.37], [-0.0003, -0.00009, 1]]
#     ).astype(np.float32)

#     pts1 = np.array([[318, 256], [534, 372], [316, 670], [73, 473]]).astype(np.float32)
#     return pts0, pts1, H_true

# def test_homographies(estimator, pts0, pts1, H_true, atol_matrix=2.0, atol_points=2.0):
#     result = estimator.estimate(pts0, pts1)

#     assert result["success"], f"{estimator.__class__.__name__} estimation failed"
#     assert result["H"] is not None, f"{estimator.__class__.__name__} matrix is None"

#     # Verify if the transformation is correct by using the inverse of H_est
#     H_est_inv = np.linalg.inv(result["H"])
#     pts0_est = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H_est_inv).reshape(-1, 2)

#     assert np.allclose(
#         pts0, pts0_est, atol=atol_points
#     ), f"{estimator.__class__.__name__} points do not match original points"

#     logger.success(f"{estimator.__class__.__name__} test passed")

# def run_homography_tests():
#     pts0, pts1, H_true = generate_homography_test_data()
#     estimators = [CvHomographyEstimator(), PoseLibHomographyEstimator()]

#     for estimator in estimators:
#         test_homographies(estimator, pts0, pts1, H_true)

# if __name__ == "__main__":
#     run_homography_tests()
