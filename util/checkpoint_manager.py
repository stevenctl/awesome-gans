# Checkpoint all our models
from . import Model
import tensorflow as tf


class CheckpointManager:

    def __init__(self, model: Model, path: str, max_to_keep: int = 5):
        self._checkpoint = tf.train.Checkpoint(**model.model_vars)
        self._manager = tf.train.CheckpointManager(self._checkpoint, path, max_to_keep)

    def save(self) -> str:
        self._manager.save()

    def list(self) -> list:
        return self._manager.checkpoints

    def load(self, path) -> bool:
        if not path in self._manager.checkpoints:
            return False
        self._checkpoint.restore(path)
        return True

    def load_latest(self) -> bool:
        if not self._manager.latest_checkpoint:
            return False
        self._checkpoint.restore(self._manager.latest_checkpoint)
        return True
