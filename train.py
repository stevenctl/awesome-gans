import time

import tensorflow as tf
from model import generate_images, \
    g_loss, d_loss, loss_func, identity_loss, cycle_loss, \
    Gx, Gy, Gx_optimizer, Gy_optimizer, \
    Dx, Dy, Dx_optimizer, Dy_optimizer

# Checkpoint all our models
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

EPOCHS = 40


@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        """FORWARD PASS - Generate images and discriminate"""
        # Generate fake y and try to recover the original x
        fake_y = Gy(real_x, training=True)
        cycle_x = Gx(fake_y, training=True)

        # Generate fake X and try to recover the original y
        fake_x = Gy(real_y, training=True)
        cycle_y = Gx(fake_x, training=True)

        # Try to generate an identical image from an image
        identity_x = Gx(real_x, training=True)
        identity_y = Gy(real_y, training=True)

        disc_real_x = Dx(real_x, training=True)
        disc_fake_x = Dx(fake_x, training=True)

        disc_real_y = Dy(real_y, training=True)
        disc_fake_y = Dy(fake_y, training=True)

        """CALCULATE LOSSES"""
        # how well generators tricked discriminators
        Gy_loss = g_loss(disc_fake_y)
        Gx_loss = g_loss(disc_fake_x)

        # how well images were able to be recovered by their own category generator
        total_cycle_loss = cycle_loss(real_x, cycle_x) + cycle_loss(real_y, cycle_y)

        total_Gx_loss = Gx_loss + total_cycle_loss + identity_loss(disc_real_x, disc_fake_x)
        total_Gy_loss = Gy_loss + total_cycle_loss + identity_loss(disc_real_y, disc_fake_y)

        Dx_loss = d_loss(disc_real_x, disc_fake_x)
        Dy_loss = d_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    Gx_grad = tape.gradient(total_Gx_loss, Gx.trainable_variables)
    Gy_grad = tape.gradient(total_Gy_loss, Gy.trainable_variables)
    Dx_grad = tape.gradient(Dx_loss, Dx.trainable_variables)
    Dy_grad = tape.gradient(Dy_loss, Dy.trainable_variables)

    Gx_optimizer.apply_gradients(zip(Gx_grad, Gx.trainable_variables))
    Gy_optimizer.apply_gradients(zip(Gy_grad, Gy.trainable_variables))

    Dx_optimizer.apply_gradients(zip(Dx_grad, Dx.trainable_variables))
    Dy_optimizer.apply_gradients(zip(Dy_grad, Dy.trainable_variables))


for epoch in range(EPOCHS):
    from data import train_x, test_x, train_y, test_y

    start = time.time()

    n = 0
    for x, y in tf.data.Dataset.zip((train_x, train_y)):
        train_step(x, y)
        if n % 10 == 0:
            print(".",)
        n += 1

    out = ""
    if (epoch + 1) % 5 == 0:
        out += checkpoint_manager.save() + " "

    print(out + str(time.time() - start))
