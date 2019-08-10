import tensorflow as tf


# normalizing the images to [-1, 1]
def _normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def _random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[256, 256, 3])

    return cropped_image


def _random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = _random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def _preprocess_image(image):
    return tf.image.resize(image, [256, 256])

def preprocess_image_train(image):
    image = _preprocess_image(image)
    image = _random_jitter(image)
    image = _normalize(image)
    return image


def preprocess_image_test(image,):
    image = _preprocess_image(image)
    image = _normalize(image)
    return image
