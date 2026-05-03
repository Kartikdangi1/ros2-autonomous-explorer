"""ONNX model loading and inference for the RL local planner."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OnnxPolicy:
    """Loads an ONNX model and runs deterministic inference."""

    def __init__(self, model_path: str, use_gpu: bool = True):
        self._session = None
        self._model_path = model_path
        try:
            import onnxruntime as ort
            # Prefer GPU providers; ONNX Runtime falls back gracefully if unavailable.
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if use_gpu
                else ['CPUExecutionProvider']
            )
            self._session = ort.InferenceSession(model_path, providers=providers)
            active = self._session.get_providers()
            logger.info('ONNX model loaded: %s  (providers: %s)', model_path, active)
        except Exception as e:
            logger.error('Failed to load ONNX model %s: %s', model_path, e)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Run inference and return action array of shape (3,).

        Returns zeros if the model failed to load.
        """
        if self._session is None:
            return np.zeros(3, dtype=np.float32)

        # Add batch dimension if needed — compare each input against its own
        # expected rank, not against inputs()[0] (which is the costmap, rank 4).
        input_ranks = {inp.name: len(inp.shape) for inp in self._session.get_inputs()}
        feeds = {}
        for key in ('costmap', 'scan', 'goal_vector', 'velocity'):
            arr = obs[key].astype(np.float32)
            if arr.ndim < input_ranks.get(key, arr.ndim + 1):
                arr = arr[np.newaxis, ...]
            feeds[key] = arr

        result = self._session.run(None, feeds)
        action = result[0][0]  # remove batch dim
        return np.clip(action, -1.0, 1.0).astype(np.float32)
