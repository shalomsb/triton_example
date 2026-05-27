"""
YOLO11x Postprocessing - Triton Python Backend

Input:  RAW_OUTPUT  [B, 84, 8400]  -> 4 bbox (cx, cy, w, h in 640x640 space) + 80 class scores
        SCALE_INFO  [B, 4]         -> [ratio, pad_w, pad_h, _]

Output: DETECTIONS      [B, N, 6]  -> [x1, y1, x2, y2, score, class_id] in ORIGINAL image coords
        NUM_DETECTIONS  [B, 1]     -> valid detections per image (others zero-padded)

Notes:
- YOLO11 (Ultralytics) head has no objectness; final confidence = max class probability.
- Coordinates are in input-resolution (640x640) pixel space and must be reversed
  through the letterbox transform: x_orig = (x_640 - pad_w) / ratio.
"""

import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # Parse parameters
        params = self.model_config.get("parameters", {})
        self.conf_threshold = float(params.get("CONF_THRESHOLD", {}).get("string_value", "0.25"))
        self.iou_threshold = float(params.get("IOU_THRESHOLD", {}).get("string_value", "0.45"))
        self.max_detections = int(params.get("MAX_DETECTIONS", {}).get("string_value", "300"))

        det_cfg = pb_utils.get_output_config_by_name(self.model_config, "DETECTIONS")
        self.det_dtype = pb_utils.triton_string_to_numpy(det_cfg["data_type"])

        num_cfg = pb_utils.get_output_config_by_name(self.model_config, "NUM_DETECTIONS")
        self.num_dtype = pb_utils.triton_string_to_numpy(num_cfg["data_type"])

    def _postprocess_single(self, raw: np.ndarray, scale: np.ndarray):
        """Process one image's raw output.

        Args:
            raw:   [84, 8400] float32
            scale: [4] -> [ratio, pad_w, pad_h, _]

        Returns:
            dets:  [N, 6] -> [x1, y1, x2, y2, score, class_id], N <= max_detections
        """
        # Transpose to [8400, 84]
        pred = raw.T  # [8400, 84]

        boxes_cxcywh = pred[:, :4]            # [8400, 4]
        class_scores = pred[:, 4:]            # [8400, 80]

        # Class-wise max as confidence (no separate objectness in YOLOv8/11)
        class_ids = np.argmax(class_scores, axis=1)            # [8400]
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]  # [8400]

        # Confidence filter
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return np.zeros((0, 6), dtype=self.det_dtype)

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # cxcywh -> xyxy (still in 640x640 letterboxed space)
        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Reverse letterbox: subtract padding, divide by ratio
        ratio, pad_w, pad_h, _ = scale.tolist()
        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h
        if ratio > 0:
            boxes_xyxy /= ratio

        # Class-aware NMS using OpenCV (offset boxes by class to make it class-aware)
        # cv2.dnn.NMSBoxes wants xywh, so convert.
        x = boxes_xyxy[:, 0]
        y = boxes_xyxy[:, 1]
        bw = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        bh = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        # Class offset trick - large enough that boxes from different classes never overlap
        max_coord = float(boxes_xyxy.max(initial=0.0)) + 1.0
        offsets = class_ids.astype(np.float32) * max_coord
        boxes_for_nms = np.stack([x + offsets, y + offsets, bw, bh], axis=1).tolist()

        idxs = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )

        if len(idxs) == 0:
            return np.zeros((0, 6), dtype=self.det_dtype)

        # OpenCV NMSBoxes returns shape (N,) or (N, 1) depending on version
        idxs = np.array(idxs).reshape(-1)

        # Sort by score (desc) and truncate
        ordered = idxs[np.argsort(-scores[idxs])][: self.max_detections]

        dets = np.zeros((len(ordered), 6), dtype=self.det_dtype)
        dets[:, 0:4] = boxes_xyxy[ordered]
        dets[:, 4] = scores[ordered]
        dets[:, 5] = class_ids[ordered].astype(self.det_dtype)
        return dets

    def execute(self, requests):
        responses = []

        for request in requests:
            raw_t = pb_utils.get_input_tensor_by_name(request, "RAW_OUTPUT").as_numpy()
            scale_t = pb_utils.get_input_tensor_by_name(request, "SCALE_INFO").as_numpy()

            batch_size = raw_t.shape[0]

            # First pass: produce per-image detection arrays (variable N per image)
            per_image_dets = []
            for i in range(batch_size):
                dets = self._postprocess_single(raw_t[i], scale_t[i])
                per_image_dets.append(dets)

            # Pad to a common N for stacking. Triton allows variable second dim (-1)
            # but a single response tensor needs a fixed shape across the batch axis;
            # we pad to the max within this batch.
            max_n = max((d.shape[0] for d in per_image_dets), default=0)
            max_n = max(max_n, 1)  # avoid zero-sized dim

            out_dets = np.zeros((batch_size, max_n, 6), dtype=self.det_dtype)
            out_num = np.zeros((batch_size, 1), dtype=self.num_dtype)

            for i, dets in enumerate(per_image_dets):
                n = dets.shape[0]
                if n > 0:
                    out_dets[i, :n] = dets
                out_num[i, 0] = n

            out_dets_t = pb_utils.Tensor("DETECTIONS", out_dets)
            out_num_t = pb_utils.Tensor("NUM_DETECTIONS", out_num)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_dets_t, out_num_t])
            )

        return responses

    def finalize(self):
        pass
