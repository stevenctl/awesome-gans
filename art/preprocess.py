import tensorflow as tf
from util.preprocessors import random_crop, random_jitter, normalize


def _resize_for_model(image):
    return tf.image.resize(image, [256, 256])


def preprocess_image_train(image):
    image = _resize_for_model(image)
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image,):
    image = _resize_for_model(image)
    image = normalize(image)
    return image
