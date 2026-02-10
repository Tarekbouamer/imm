- toImageTensor():
  FIXME: width / height are swapped.
  cv_img is HWC, but code assigns w_new, h_new = cv_img.shape[:2]

- pad_image_bottom_right():
  FIXME: for CHW input, mask is created with shape (C, H, W).
  Mask must be spatial only (H, W) to avoid silent downstream errors.

- load_image():
  FIXME: `padding` argument is accepted but ignored.
  Function always returns mask=None and never applies padding.
  This breaks the function’s API contract.
- Model download configuration:
  FIXME: Models without weights fail with "Invalid pretrained configuration. Specify 'file', 'url', or 'drive'."
  Need a way to label models without weights so they can be skipped during download.
  Error: Failed to download nn: Error downloading model weights.
