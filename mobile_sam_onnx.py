"""MobileSAM ONNX inference (encoder + single-mask decoder) for CPU RunPod."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SAM_ENCODER_PATH = Path(
    __import__("os").environ.get(
        "MOBILESAM_ENCODER_PATH", "/root/.mobilesam/mobile_sam_image_encoder.onnx"
    )
)
SAM_DECODER_PATH = Path(
    __import__("os").environ.get(
        "MOBILESAM_DECODER_PATH", "/root/.mobilesam/sam_mask_decoder_single.onnx"
    )
)

SAM_INPUT_SIZE = 1024
SAM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _session(model_path: Path) -> ort.InferenceSession | None:
    if not model_path.exists() or model_path.stat().st_size == 0:
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (__import__("os").cpu_count() or 4) // 2)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), opts, providers=["CPUExecutionProvider"]
    )


class MobileSamOnnx:
    def __init__(self) -> None:
        self.encoder = _session(SAM_ENCODER_PATH)
        self.decoder = _session(SAM_DECODER_PATH)
        self.ready = self.encoder is not None and self.decoder is not None
        if self.ready:
            self._enc_in = self.encoder.get_inputs()[0].name

    def _preprocess(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float]:
        h, w = rgb.shape[:2]
        scale = SAM_INPUT_SIZE / float(max(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad = np.zeros((SAM_INPUT_SIZE, SAM_INPUT_SIZE, 3), dtype=np.float32)
        pad[:new_h, :new_w] = resized.astype(np.float32)
        pad = (pad - SAM_MEAN) / SAM_STD
        chw = np.ascontiguousarray(pad.transpose(2, 0, 1)[None, ...])
        return chw, (new_h, new_w), scale

    @staticmethod
    def _transform_point(x: float, y: float, scale: float) -> np.ndarray:
        return np.array([[[x * scale, y * scale]]], dtype=np.float32)

    def predict_mask(
        self, rgb: np.ndarray, point_xy: tuple[int, int], *, positive: int = 1
    ) -> np.ndarray | None:
        """Boolean mask (H, W) aligned with input rgb."""
        if not self.ready:
            return None
        h, w = rgb.shape[:2]
        emb_in, _res_hw, scale = self._preprocess(rgb)
        embeddings = self.encoder.run(None, {self._enc_in: emb_in})[0]

        px, py = point_xy
        point_coords = self._transform_point(float(px), float(py), scale)
        point_labels = np.array([[positive]], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)
        orig_im_size = np.array([h, w], dtype=np.float32)

        decoder_args = {
            "image_embeddings": embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size,
        }
        outputs = self.decoder.run(None, decoder_args)
        masks = outputs[0]
        # (1, num_masks, H, W) or (1, 1, H, W)
        mask = masks[0, 0] > 0.0
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        return mask
