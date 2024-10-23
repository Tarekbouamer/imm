import gradio as gr
from typing import Dict, List, Union, Tuple, Optional
import cv2
import numpy as np
import torch
import json
from datetime import datetime

from imm.extractors import __all__ as extractor_names
from imm.matchers import __all__ as matcher_names
from imm.image_matching import ImageMatcher
from imm.utils.warnings import suppress_warnings


COLOR_SCHEMES = {
    "Default": [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
    "Rainbow": [(0, 255, 255), (0, 255, 0), (255, 0, 255)],
    "Heat": [(0, 0, 255), (255, 0, 0), (0, 165, 255)],
}


class ImageMatchingGradioApp:
    """
    A Gradio app for visualizing and analyzing image matching using different feature extractors and matchers.
    """

    def __init__(self):
        self.custom_css = """
        .container {max-width: 1200px; margin: auto;}
        .output-image {width: 100%; max-height: 600px; object-fit: contain;}
        .input-image {height: 300px; object-fit: contain;}
        .advanced-options {border: 1px solid #ddd; padding: 10px; margin-top: 10px;}
        """
        self.image_matcher: Optional[ImageMatcher] = None
        self.history: List[List[Union[str, int, float]]] = []

    def build_interface(self) -> gr.Blocks:
        """Constructs and returns the Gradio interface."""
        with gr.Blocks(css=self.custom_css) as demo:
            gr.Markdown("""# 🌟 Welcome to the Advanced Image Matching Visualization Tool""")
            gr.Markdown(""" ### This Gradio application provides an intuitive platform for visualizing and analyzing image matching with various feature extractors and matchers. It's designed for researchers, developers, and enthusiasts in computer vision.
                        ## **Quick Start Guide:**
                        1. **📤 Upload Images**: Select two images for imm.
                        2. **🛠️ Configure Settings**: Set your preferences for extractors, matchers, and other parameters.
                        3. **🔍 Analyze**: Hit "Match Images" to see how algorithms match features between your images.
                        4. **📥 Export Results**: Check historical data and export findings for further analysis.
                        ### Explore cutting-edge image matching techniques with ease and gain insights into different algorithms' performance!
                        """)

            with gr.Row():
                with gr.Column():
                    image0_input = gr.Image(
                        type="numpy",
                        label="Image 0",
                        elem_classes="input-image",
                    )
                    image0_info = gr.JSON(label="Image 0 Info")
                with gr.Column():
                    image1_input = gr.Image(
                        type="numpy",
                        label="Image 1",
                        elem_classes="input-image",
                    )
                    image1_info = gr.JSON(label="Image 1 Info")

            with gr.Row():
                with gr.Column():
                    extractor_input = gr.Dropdown(
                        choices=extractor_names,
                        value="superpoint",
                        label="Feature Extractor",
                    )
                    matcher_input = gr.Dropdown(
                        choices=matcher_names,
                        value="superglue_outdoor",
                        label="Matcher Algorithm",
                    )
                with gr.Column():
                    max_size_input = gr.Slider(200, 2048, value=640, step=1, label="Max Image Size")
                    max_keypoints_input = gr.Slider(0, 4096, value=1024, step=1, label="Max Keypoints")
                with gr.Column():
                    min_conf_input = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Min Confidence")
                    use_gpu_input = gr.Checkbox(value=False, label="Use GPU (if available)")

            with gr.Accordion("Advanced Options", open=False):
                with gr.Row(elem_classes="advanced-options"):
                    with gr.Column():
                        force_cpu_input = gr.Checkbox(value=False, label="Force CPU (override GPU)")
                        overlay_lines_input = gr.Checkbox(value=True, label="Overlay matching lines")
                    with gr.Column():
                        line_thickness_input = gr.Slider(1, 10, value=1, step=1, label="Line Thickness")
                        point_size_input = gr.Slider(1, 10, value=2, step=1, label="Keypoint Size")
                    with gr.Column():
                        alpha_input = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Overlay Alpha")
                        color_scheme_input = gr.Radio(
                            ["Default", "Rainbow", "Heat"],
                            value="Default",
                            label="Color Scheme",
                        )

            with gr.Row():
                match_button = gr.Button("Match Images", variant="primary")

            with gr.Tabs():
                with gr.TabItem("Results"):
                    with gr.Row():
                        composite_image_output = gr.Image(
                            type="numpy",
                            label="Matching Visualization",
                            elem_classes="output-image",
                        )
                    with gr.Row():
                        with gr.Column():
                            stats_output = gr.JSON(label="Matching Statistics")
                        with gr.Column():
                            error_output = gr.Textbox(label="Errors/Warnings", lines=3)

                with gr.TabItem("History"):
                    history_output = gr.Dataframe(
                        headers=[
                            "Timestamp",
                            "Extractor",
                            "Matcher",
                            "# Matches",
                            "Avg. Confidence",
                        ],
                        label="Matching History",
                    )
                    export_history_button = gr.Button("Export History")

            image0_input.change(
                self.update_image_info,
                inputs=[image0_input],
                outputs=[image0_info],
            )
            image1_input.change(
                self.update_image_info,
                inputs=[image1_input],
                outputs=[image1_info],
            )

            match_button.click(
                fn=self.match_images_wrapper,
                inputs=[
                    image0_input,
                    image1_input,
                    extractor_input,
                    matcher_input,
                    max_size_input,
                    max_keypoints_input,
                    min_conf_input,
                    # output_dir_input,
                    use_gpu_input,
                    force_cpu_input,
                    overlay_lines_input,
                    line_thickness_input,
                    point_size_input,
                    alpha_input,
                    color_scheme_input,
                ],
                outputs=[
                    composite_image_output,
                    stats_output,
                    error_output,
                    history_output,
                ],
            )

            export_history_button.click(
                fn=self.export_history,
                outputs=[gr.File(label="Download History")],
            )

        return demo

    @staticmethod
    def update_image_info(image: np.ndarray) -> Dict[str, Union[str, float]]:
        """Updates the information about the uploaded image, such as shape, data type, min, and max values."""
        if image is not None:
            return {
                "Shape": image.shape,
                "dtype": str(image.dtype),
                "min": float(np.min(image)),
                "max": float(np.max(image)),
            }
        return {}

    @staticmethod
    def cv_to_tensor(image: np.ndarray) -> torch.Tensor:
        """Converts an image from OpenCV format to a PyTorch tensor, normalizing the pixel values."""
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Normalize image
        image = image / 255.0
        return image

    def match_images(
        self,
        image0_cv: np.ndarray,
        image1_cv: np.ndarray,
        extractor: str,
        matcher: str,
        max_size: int,
        max_keypoints: int,
        min_conf: float,
        use_gpu: bool,
        force_cpu: bool,
        overlay_lines: bool,
        line_thickness: int,
        point_size: int,
        alpha: float,
        color_scheme: str,
    ) -> Tuple[np.ndarray, Dict[str, Union[int, float]]]:
        """Matches two images using the specified extractor and matcher, and returns the matched image and statistics."""
        # Initialize ImageMatcher with selected extractor and matcher
        use_gpu = use_gpu and not force_cpu
        self.image_matcher = ImageMatcher(extractor, matcher, use_gpu)

        # Convert images from OpenCV format to PyTorch tensors
        image0 = self.cv_to_tensor(image0_cv)
        image1 = self.cv_to_tensor(image1_cv)

        # Cuda
        if use_gpu:
            image0 = image0.cuda()
            image1 = image1.cuda()

        # Perform image matching
        results = self.image_matcher.match_pairs(
            image0,
            image1,
            max_size=max_size,
            min_conf=min_conf,
            max_keypoints=max_keypoints,
            image0_cv=image0_cv,
            image1_cv=image1_cv,
        )

        mkpts0 = results["matches"]["mkpts0"]
        mkpts1 = results["matches"]["mkpts1"]
        mscores = results["matches"]["mscores"]

        num_mkpts0 = len(mkpts0)
        num_mkpts1 = len(mkpts1)
        score_mean = float(np.mean(mscores)) if len(mscores) > 0 else 0
        score_std = float(np.std(mscores)) if len(mscores) > 0 else 0

        stats = {
            "# keypoints in image0": num_mkpts0,
            "# keypoints in image1": num_mkpts1,
            "# matches": len(mscores),
            "Mean score": score_mean,
            "Std score": score_std,
        }

        # Custom visualization based on new parameters
        composite_image = self.custom_visualization(
            image0_cv,
            image1_cv,
            results,
            overlay_lines,
            line_thickness,
            point_size,
            alpha,
            color_scheme,
        )

        # Update history
        self.update_history(extractor, matcher, len(mscores), score_mean)

        return composite_image, stats

    def custom_visualization(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        results: Dict,
        overlay_lines: bool,
        line_thickness: int,
        point_size: int,
        alpha: float,
        color_scheme: str,
        gap: int = 10,
    ) -> np.ndarray:
        """Creates a custom visualization of the matched images."""
        image1_rgb = self.ensure_rgb(image1)
        image2_rgb = self.ensure_rgb(image2)

        # Create composite image
        composite_image = self.draw_composite_image(image1_rgb, image2_rgb, gap)

        kpts0 = results["matches"].get("kpts0", [])
        kpts1 = results["matches"].get("kpts1", [])
        mkpts0 = results["matches"]["mkpts0"]
        mkpts1 = results["matches"]["mkpts1"]
        mscores = results["matches"]["mscores"]

        # Get color scheme
        color_inliers, color_outliers, color_lines = COLOR_SCHEMES[color_scheme]

        # Draw all keypoints
        for kp in kpts0:
            cv2.circle(
                composite_image,
                (int(kp[0]), int(kp[1])),
                point_size,
                color_outliers,
                -1,
            )
        for kp in kpts1:
            cv2.circle(
                composite_image,
                (int(kp[0]) + image1_rgb.shape[1] + gap, int(kp[1])),
                point_size,
                color_outliers,
                -1,
            )

        # Draw mutual keypoints
        for i in range(len(mkpts0)):
            cv2.circle(
                composite_image,
                (int(mkpts0[i][0]), int(mkpts0[i][1])),
                point_size,
                color_inliers,
                -1,
            )
            cv2.circle(
                composite_image,
                (
                    int(mkpts1[i][0]) + image1_rgb.shape[1] + gap,
                    int(mkpts1[i][1]),
                ),
                point_size,
                color_inliers,
                -1,
            )

        # Draw matching lines
        if overlay_lines and mscores is not None:
            for i, score in enumerate(mscores):
                kp0 = mkpts0[i]
                kp1_offset = (
                    mkpts1[i][0] + image1_rgb.shape[1] + gap,
                    mkpts1[i][1],
                )
                color = tuple(int(c * score) for c in color_lines)
                cv2.line(
                    composite_image,
                    (int(kp0[0]), int(kp0[1])),
                    (int(kp1_offset[0]), int(kp1_offset[1])),
                    color,
                    line_thickness,
                )

        # Apply alpha blending
        overlay = composite_image.copy()
        cv2.addWeighted(overlay, alpha, composite_image, 1 - alpha, 0, composite_image)

        return composite_image

    def ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Ensures the image is in RGB format."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def draw_composite_image(self, image1: np.ndarray, image2: np.ndarray, gap: int = 10) -> np.ndarray:
        """Creates a single composite image from two input images with a small gap between them."""
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2 + gap
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        composite[:h1, :w1] = image1
        composite[:h2, w1 + gap :] = image2
        return composite

    def update_history(
        self,
        extractor: str,
        matcher: str,
        num_matches: int,
        avg_confidence: float,
    ):
        """Updates the history of matches with the latest results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append([timestamp, extractor, matcher, num_matches, avg_confidence])

    def export_history(self) -> Optional[str]:
        """Exports the matching history to a JSON file and returns the filename."""
        if not self.history:
            return None

        filename = f"matching_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.history, f)
        return filename

    def match_images_wrapper(
        self, *args
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[Dict[str, Union[int, float]]],
        str,
        List[List[Union[str, int, float]]],
    ]:
        """Wrapper function for image matching to handle errors and format the output for Gradio interface."""
        try:
            result_image, stats = self.match_images(*args)
            return result_image, stats, "", self.history
        except Exception as e:
            return None, None, str(e), self.history

    def launch(self):
        """Launches the Gradio app."""
        demo = self.build_interface()
        demo.launch()


@suppress_warnings()
def main():
    app = ImageMatchingGradioApp()
    app.launch()


if __name__ == "__main__":
    main()

# TODO: variable image size and max_keypoints are not affecting the output
