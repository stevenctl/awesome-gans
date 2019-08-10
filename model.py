import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

NORM_TYPE='instancenorm'
OUTPUT_CHANNELS=3

# Discriminators (Dx tries to identify the genuine X)
Dx = pix2pix.discriminator(norm_type=NORM_TYPE, target=False)
Dy = pix2pix.discriminator(norm_type=NORM_TYPE, target=False)

# Generators (Gx converts things into an X and vice versa)
Gx = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type=NORM_TYPE)
Gy = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type=NORM_TYPE)

# LAMBDA scales the magnitude of identity and cycle losses
LAMBDA = 10
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)


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


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


Gx_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
Gy_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

Dx_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
Dy_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

