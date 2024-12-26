import click
import cv2
import numpy as np

from imm.estimators import create_pnp_estimator


def quaternion_to_rotation_vector(qvec):
    qvec = qvec.ravel()
    qvec = qvec / np.linalg.norm(qvec)
    angle = 2 * np.arccos(qvec[0])
    axis = qvec[1:] / np.sin(angle / 2)
    return axis * angle


def process_frame(img, estimator, cmx, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        height, width = img.shape[:2]

        camera = {
            "model": "SIMPLE_PINHOLE",
            "width": width,
            "height": height,
            "params": [cmx[0, 0], cmx[1, 1], cmx[0, 2], cmx[1, 2]],
        }

        ret = estimator.estimate(corners2, objp, camera, dist=dist)

        if ret["success"] is False:
            raise ValueError("Estimation failed.")
            exit(1)

        qvec = ret["qvec"]
        qvec = quaternion_to_rotation_vector(qvec)
        tvec = ret["tvec"]

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        axis_img, _ = cv2.projectPoints(axis, qvec, tvec, cmx, dist)
        img = draw(img, corners2, axis_img)
        cv2.putText(
            img,
            f"Rotation Vector: {np.round(qvec.ravel(), 2)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f"Translation Vector: {np.round(tvec.ravel(), 2)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)
    return img


@click.command()
@click.argument("input_path", type=click.Path(exists=True), default="assets/chessboard.jpg")
@click.argument("calibration_file", type=click.Path(exists=True), default="assets/calib.npz")
@click.option("--backend", type=click.Choice(["opencv", "poselib", "pycolmap"]), default="poselib", help="Backend to use.")
@click.help_option("--help", "-h")
def handle_input(input_path, calibration_file, backend):
    calib_data = np.load(calibration_file)
    cmx = calib_data["cmx"]
    dist = calib_data["dist"]

    estimator = create_pnp_estimator(backend=backend)

    if input_path.lower().endswith((".mp4", ".avi")):
        cap = cv2.VideoCapture(input_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, estimator, cmx, dist)
            cv2.imshow("Video", processed_frame)
            if cv2.waitKey(1000) & 0xFF == ord("q"):
                break
        cap.release()
    else:
        img = cv2.imread(input_path)
        if img is not None:
            processed_image = process_frame(img, estimator, cmx, dist)
            cv2.imshow("Image", processed_image)
            cv2.waitKey(4000)
        else:
            print("Error loading image. Check the image path.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    handle_input()
