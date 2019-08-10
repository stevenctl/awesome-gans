import pathlib
import random

import tensorflow as tf


def _paths_in_folder(path: str) -> list:
    return [str(p) for p in list(pathlib.Path(path).glob("*.jpg"))]


def _paths_to_jpeg_ds(paths: list) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(paths) \
        .map(tf.io.read_file) \
        .map(tf.image.decode_jpeg)


def load_jpeg_folder(path: str) -> tf.data.Dataset:
    paths = _paths_in_folder(path)
    return _paths_to_jpeg_ds(paths)


def load_jpeg_folder_split(
        path: str,
        test_portion: float = .2,
        shuffle: bool = True
) -> (tf.data.Dataset, tf.data.Dataset):
    paths = _paths_in_folder(path)
    if shuffle:
        random.shuffle(paths)

    ntest = int(test_portion * len(paths))

    train = _paths_to_jpeg_ds(paths[ntest:])
    test = _paths_to_jpeg_ds(paths[:ntest])

    return train, test
