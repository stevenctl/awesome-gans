import random
import pathlib

from preprocess import preprocess_image_test, preprocess_image_train
import tensorflow as tf
from util import compose

X_CLASS = 'gloss_atomic_teal'
Y_CLASS = 'gloss_plum_explosion'

data_root = pathlib.Path('./dataset')

x_image_paths = list(data_root.glob('%s/*' % X_CLASS))
y_image_paths = list(data_root.glob('%s/*' % Y_CLASS))
[random.shuffle(ds) for ds in [x_image_paths, y_image_paths]]

print("%d %s images" % (len(x_image_paths), X_CLASS))
print("%d %s images" % (len(y_image_paths), Y_CLASS))


def load_dataset(paths):
    files = [str(p) for p in paths]
    path_ds = tf.data.Dataset.from_tensor_slices(files)
    image_ds = path_ds.map(tf.io.read_file).map(tf.image.decode_jpeg)
    return image_ds

# Use 10% of images from the smallest category
test_portion = int(min(len(x_image_paths), len(y_image_paths)) * 0.1)

# Load images from filesystem
train_x = load_dataset(x_image_paths[test_portion:])

test_x = load_dataset(x_image_paths[:test_portion])

train_y = load_dataset(y_image_paths[test_portion:])
test_y = load_dataset(y_image_paths[:test_portion])

AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 10

train_x = train_x.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

train_y = train_y.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_x = test_x.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_y = test_y.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)
