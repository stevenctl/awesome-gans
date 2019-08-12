import pathlib
import random
from PIL import Image
import tensorflow as tf


def _paths_in_folder(path: str) -> list:
    return [str(p) for p in list(pathlib.Path(path).glob("*.jpg"))]


def _paths_to_jpeg_ds(paths: list) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(paths) \
        .map(tf.io.read_file) \
        .map(tf.image.decode_jpeg)


def crop_cells(path, cell_size=512, grayscale=True) -> list:
    # Load and convert to grayscale
    img = Image.open(path)
    if grayscale:
        img = img.convert('LA').convert('RGB')

    # Trim top and bottom so the width and height are divisible by CELL_SIZE
    h_crop = (img.width % cell_size) / 2
    v_crop = (img.height % cell_size) / 2
    img = img.crop((h_crop, v_crop, img.width - h_crop, img.height - v_crop))

    cells = []
    for x in range(0, img.width, cell_size):
        for y in range(0, img.height, cell_size):
            cells.append(img.crop((x, y, x + cell_size, y + cell_size)))

    return cells


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
