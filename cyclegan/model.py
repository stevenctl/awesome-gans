import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from util import Model

NORM_TYPE = "instancenorm"
OUTPUT_CHANNELS = 3

# LAMBDA scales the magnitude of identity and cycle losses
LAMBDA = 10
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class CycleGAN(Model):
    def __init__(self):
        build_discriminator = lambda: pix2pix.discriminator(norm_type=NORM_TYPE, target=False)
        build_optimizer = lambda: tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        build_generator = lambda: pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type=NORM_TYPE)

        # Discriminators (Tries to identify if the input is a real example of its class)
        self.Dx = build_discriminator()
        self.Dy = build_discriminator()
        # Generators (Converts from the opposite class into the new class)
        self.Gx = build_generator()
        self.Gy = build_generator()
        # Optimizers to perform stochastic graident descent
        self.Dx_optimizer = build_optimizer()
        self.Dy_optimizer = build_optimizer()
        self.Gx_optimizer = build_optimizer()
        self.Gy_optimizer = build_optimizer()

    def generate_x(self, image):
        return self.Gx(image)

    def generate_y(self, image):
        return self.Gy(image)

    def get_vars(self):
        return dict(
            Dx=self.Dx, Dy=self.Dy, Gx=self.Gx, Gy=self.Gy,
            Dx_optimizer=self.Dx_optimizer, Dy_optimizer=self.Dy_optimizer,
            Gx_optimizer=self.Gx_optimizer, Gy_optimizer=self.Gy_optimizer,
        )

    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            """FORWARD PASS - Generate images and discriminate"""
            # Generate fake y and try to recover the original x
            fake_y = self.Gy(real_x, training=True)
            cycle_x = self.Gx(fake_y, training=True)

            # Generate fake X and try to recover the original y
            fake_x = self.Gy(real_y, training=True)
            cycle_y = self.Gx(fake_x, training=True)

            # Try to generate an identical image from an image
            identity_x = self.Gx(real_x, training=True)
            identity_y = self.Gy(real_y, training=True)

            disc_real_x = self.Dx(real_x, training=True)
            disc_fake_x = self.Dx(fake_x, training=True)

            disc_real_y = self.Dy(real_y, training=True)
            disc_fake_y = self.Dy(fake_y, training=True)

            """CALCULATE LOSSES"""
            # how well generators tricked discriminators
            Gy_loss = g_loss(disc_fake_y)
            Gx_loss = g_loss(disc_fake_x)

            # how well images were able to be recovered by their own category generator
            total_cycle_loss = cycle_loss(real_x, cycle_x) + cycle_loss(real_y, cycle_y)

            total_Gx_loss = Gx_loss + total_cycle_loss + identity_loss(real_x, identity_x)
            total_Gy_loss = Gy_loss + total_cycle_loss + identity_loss(real_y, identity_y)

            Dx_loss = d_loss(disc_real_x, disc_fake_x)
            Dy_loss = d_loss(disc_real_y, disc_fake_y)

            # Calculate the gradients for generator and discriminator
        Gx_grad = tape.gradient(total_Gx_loss, self.Gx.trainable_variables)
        Gy_grad = tape.gradient(total_Gy_loss, self.Gy.trainable_variables)
        Dx_grad = tape.gradient(Dx_loss, self.Dx.trainable_variables)
        Dy_grad = tape.gradient(Dy_loss, self.Dy.trainable_variables)

        self.Gx_optimizer.apply_gradients(zip(Gx_grad, self.Gx.trainable_variables))
        self.Gy_optimizer.apply_gradients(zip(Gy_grad, self.Gy.trainable_variables))

        self.Dx_optimizer.apply_gradients(zip(Dx_grad, self.Dx.trainable_variables))
        self.Dy_optimizer.apply_gradients(zip(Dy_grad, self.Dy.trainable_variables))


def d_loss(real, fake):
    real_loss = loss_func(tf.ones_like(real), real)
    fake_loss = loss_func(tf.zeros_like(fake), fake)

    return 0.5 * (real_loss + fake_loss)


# Generator loss is 1 - disc "realness". More realistic means lower loss
def g_loss(disc_fake_val):
    return loss_func(tf.ones_like(disc_fake_val), disc_fake_val)


# Identity Loss = Gx(X) - X
# If Gx is given an example from X it should preserve that image
def identity_loss(real_image, gen_same_image):
    loss = tf.reduce_mean(tf.abs(real_image - gen_same_image))
    return LAMBDA * 0.5 * loss


# Cycle loss is Y - Gy(Gx(Y))
# This means if we convert a Y to an X, we should be able to get the Y back
def cycle_loss(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))
