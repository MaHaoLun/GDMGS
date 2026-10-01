"""Owned state for a single frozen inference run."""

from .finalized_scene import FinalizedScene


class InferenceSession:
    def __init__(self, model, checkpoint_path, iteration):
        self.model = model
        self.scene = FinalizedScene.from_model(model, checkpoint_path, iteration)
        self.closed = False

    def validate(self):
        if self.closed:
            raise RuntimeError("Inference session is closed.")
        self.scene.assert_current(self.model)

    def close(self):
        self.closed = True

    def __enter__(self):
        self.validate()
        return self

    def __exit__(self, *exc):
        self.close()
