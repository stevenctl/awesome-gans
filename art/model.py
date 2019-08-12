import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from util import Model

NORM_TYPE = "instancenorm"
OUTPUT_CHANNELS = 3

# LAMBDA scales the magnitude of identity and cycle losses
LAMBDA = 10
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

input_shape = (1, 256, 256, 3)


class ArtGAN(Model):
    def __init__(self):
        self.generator = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type=NORM_TYPE)
        self.discriminator = pix2pix.discriminator(norm_type=NORM_TYPE, target=False)
        # Optimizers to perform stochastic graident descent
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def generate(self):
        return self.generator(self._noise())

    def _noise(self):
        return tf.random.uniform(
            (1, 256, 256, 3),
            minval=0,
            maxval=None,
            dtype=tf.dtypes.float32,
            seed=None,
            name=None
        )

    def get_vars(self):
        return dict(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
        )

    @tf.function
    def train_step(self, real):
        with tf.GradientTape(persistent=True) as tape:
            """FORWARD PASS - Generate images and discriminate"""
            # Generate fake y and try to recover the original x
            noise = self._noise()
            fake = self.generator(noise, training=True)

            disc_fake = self.discriminator(fake, training=True)
            disc_real = self.discriminator(real, training=True)

            """CALCULATE LOSSES"""
            # how well generators tricked discriminators
            gen_loss = g_loss(disc_fake)
            disc_loss = d_loss(disc_real, disc_fake)

        # Calculate the gradients for generator and discriminator
        gen_grad = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grad = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))


def d_loss(real, fake):
    real_loss = loss_func(tf.ones_like(real), real)
    fake_loss = loss_func(tf.zeros_like(fake), fake)

    return 0.5 * (real_loss + fake_loss)


# Generator loss is 1 - disc "realness". More realistic means lower loss
def g_loss(disc_fake_val):
    return loss_func(tf.ones_like(disc_fake_val), disc_fake_val)
