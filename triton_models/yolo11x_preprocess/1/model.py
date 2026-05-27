"""
YOLO11x Preprocessing - Triton Python Backend

Applies letterbox resize to 640x640 (preserving aspect ratio), BGR->RGB,
normalize to [0, 1], and transpose HWC -> CHW.

Outputs SCALE_INFO = [ratio, pad_w, pad_h, _reserved] per image so the
postprocess step can reverse the transform back to the original frame.
"""

import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


# Model input spec - YOLO11x default
INPUT_H = 640
INPUT_W = 640


def letterbox(img: np.ndarray, new_shape=(INPUT_H, INPUT_W), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints.

    Returns:
        padded:  HxWx3 uint8 image at new_shape
        ratio:   scalar resize ratio applied
        pad_w:   left/right padding (pixels added on the left)
        pad_h:   top/bottom padding (pixels added on the top)
    """
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)

    new_unpad_w = int(round(w0 * r))
    new_unpad_h = int(round(h0 * r))

    dw = new_shape[1] - new_unpad_w
    dh = new_shape[0] - new_unpad_h
    dw /= 2
    dh /= 2

    if (w0, h0) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, left, top


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        out_cfg = pb_utils.get_output_config_by_name(
            self.model_config, "PREPROCESSED_IMAGE"
        )
        self.out_dtype = pb_utils.triton_string_to_numpy(out_cfg["data_type"])

        scale_cfg = pb_utils.get_output_config_by_name(self.model_config, "SCALE_INFO")
        self.scale_dtype = pb_utils.triton_string_to_numpy(scale_cfg["data_type"])

    def execute(self, requests):
        responses = []

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
            batch = in_tensor.as_numpy()  # shape: (B, H, W, 3) uint8

            # Triton may give us a non-batched single image if max_batch_size=0;
            # we configured max_batch_size=8 so the leading dim is always batch.
            batch_size = batch.shape[0]

            preprocessed = np.empty((batch_size, 3, INPUT_H, INPUT_W), dtype=self.out_dtype)
            scale_info = np.empty((batch_size, 4), dtype=self.scale_dtype)

            for i in range(batch_size):
                img = batch[i]  # HxWx3, uint8, assumed BGR (DeepStream/OpenCV convention)

                padded, ratio, pad_w, pad_h = letterbox(img, (INPUT_H, INPUT_W))

                # BGR -> RGB
                padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

                # HWC uint8 -> CHW float32 in [0, 1]
                chw = padded.astype(np.float32, copy=False).transpose(2, 0, 1) / 255.0

                preprocessed[i] = chw
                scale_info[i] = np.array([ratio, pad_w, pad_h, 0.0], dtype=self.scale_dtype)

            out_img = pb_utils.Tensor("PREPROCESSED_IMAGE", preprocessed)
            out_scale = pb_utils.Tensor("SCALE_INFO", scale_info)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_img, out_scale])
            )

        return responses

    def finalize(self):
        pass
