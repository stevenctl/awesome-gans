# Checkpoint all our models
import tensorflow as tf
from model import \
    Gx, Gy, Gx_optimizer, Gy_optimizer, \
    Dx, Dy, Dx_optimizer, Dy_optimizer

checkpoint = tf.train.Checkpoint(
    Gx=Gx, Gy=Gy, Dx=Dx, Dy=Dy,
    Gx_optimizer=Gx_optimizer, Gy_optimizer=Gy_optimizer,
    Dx_optimizer=Dx_optimizer, Dy_optimizer=Dy_optimizer,
)

checkpoint_path = './checkpoints/train'
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
