"""ONNX model loading and inference for the RL local planner."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OnnxPolicy:
    """Loads an ONNX model and runs deterministic inference."""

    def __init__(self, model_path: str):
        self._session = None
        self._model_path = model_path
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider'],
            )
            logger.info('ONNX model loaded: %s', model_path)
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

        # Add batch dimension if needed
        feeds = {}
        for key in ('costmap', 'scan', 'goal_vector', 'velocity'):
            arr = obs[key]
            if arr.ndim == len(self._session.get_inputs()[0].shape) - 1:
                arr = arr[np.newaxis, ...]
            feeds[key] = arr.astype(np.float32)

        result = self._session.run(None, feeds)
        action = result[0][0]  # remove batch dim
        return np.clip(action, -1.0, 1.0).astype(np.float32)
