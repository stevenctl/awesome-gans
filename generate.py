from model import translate_image
from checkpoint import Gx, Gy
from data import train_x, test_x, train_y, test_y

batches = [
    (Gx, train_y),
    (Gx, test_y),
    (Gy, train_x),
    (Gy, test_x),
]

for G, dataset in batches:
    for img in dataset:
        translate_image(G, img)
