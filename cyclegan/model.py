import tensorflow as tf
from util import Model
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

NORM_TYPE = 'instancenorm'
OUTPUT_CHANNELS = 3

# LAMBDA scales the magnitude of identity and cycle losses
LAMBDA = 10
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class CycleGAN(Model):

    def build_model(self):
        build_discriminator = lambda: pix2pix.discriminator(norm_type=NORM_TYPE, target=False)
        build_optimizer = lambda: tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        build_generator = lambda: pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type=NORM_TYPE)
        return {
            # Discriminators (Tries to identify if the input is a real example of its class)
            'Dx ': build_discriminator(), 'Dy ': build_discriminator(),
            # Generators (Converts from the opposite class into the new class)
            'Gx ': build_generator(), 'Gy ': build_generator(),
            # Optimizers to perform stochastic graident descent
            'Dx_optimizer ': build_optimizer(), 'Dy_optimizer': build_optimizer(),
            'Gx_optimizer ': build_optimizer(), 'Gy_optimizer ': build_optimizer(),
        }

    def generate_x(self, image):
        return self.model_vars['Gx'](image)

    def generate_y(self, image):
        return self.model_vars['Gx'](image)

    @tf.function
    def train_step(self, *inputs):
        real_x, real_y = inputs
        Gx, Gy, Dx, Dy = [self.model_vars[n] for n in ['Gx', 'Gy', 'Dx', 'Dy']]
        Gx_optimizer, Gy_optimizer, Dx_optimizer, Dy_optimizer = [
            self.model_vars[n + '_optimizer']
            for n in ['Gx', 'Gy', 'Dx', 'Dy']
        ]
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

            total_Gx_loss = Gx_loss + total_cycle_loss + identity_loss(real_x, identity_x)
            total_Gy_loss = Gy_loss + total_cycle_loss + identity_loss(real_y, identity_y)

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
