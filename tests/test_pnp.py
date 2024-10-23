# import numpy as np
# import cv2
# from loguru import logger
# from imm.estimators import (
#     PoseLibPnPEstimator,
#     PycolmapPnPEstimator,
# )

# def generate_pnp_test_data(num_points=10):


#     pts3d = np.random.rand(num_points, 3).astype(np.float32) * 100
#     R_true = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]
#     t_true = np.array([[10], [20], [30]], dtype=np.float32)
#     K = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)
#     pts2d, _ = cv2.projectPoints(pts3d, R_true, t_true, K, None)

#     print(pts2d)

#     camera = {
#         "model": "PINHOLE",
#         "width": 640,
#         "height": 480,
#         "params": [320, 240, 640, 480],
#     }
#     return pts2d.reshape(-1, 2), pts3d, camera

# def test_pnp_estimator(estimator, pts2d, pts3d, camera, atol_points=1.0):
#     result = estimator.estimate(pts2d, pts3d, camera)

#     assert result["success"], f"{estimator.__class__.__name__} PnP estimation failed"
#     assert result["qvec"] is not None, "Quaternion vector is None"
#     assert result["tvec"] is not None, "Translation vector is None"
#     assert len(result["inliers"]) > 0, "No inliers found"

#     logger.success(f"{estimator.__class__.__name__} test passed")

# def run_pnp_tests():
#     pts2d, pts3d, camera = generate_pnp_test_data()
#     estimators = [PoseLibPnPEstimator(), PycolmapPnPEstimator()]

#     for estimator in estimators:
#         test_pnp_estimator(estimator, pts2d, pts3d, camera)

# if __name__ == "__main__":
#     run_pnp_tests()
