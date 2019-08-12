import time
import tensorflow as tf
from util import CheckpointManager, load_jpeg_folder_split
from art import ArtGAN, preprocess as art_preprocess
from matplotlib import pyplot as plt

# Initialize our model
model = ArtGAN()

# Try to resume a previous session
checkpoint_manager = CheckpointManager(model, 'checkpoints/artgan/train')
if checkpoint_manager.load_latest():
    print("Successfully loaded checkpoint!")

# Load in the dataset; 10% will be used as "test" data
train_input, test_input = load_jpeg_folder_split('./dataset/trent', test_portion=0.1)


# Preprocess the images and shuffle them
def preprocess_ds(ds: tf.data.Dataset, preprocessor, buffer_size=10) -> tf.data.Dataset:
    return ds \
        .map(preprocessor, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .cache().shuffle(buffer_size).batch(1)


train_input = preprocess_ds(train_input, art_preprocess.preprocess_image_train)
test_input = preprocess_ds(test_input, art_preprocess.preprocess_image_test)

# Setup a progress checker so we can watch the results as we train
sample_x = next(iter(train_input))


def check_progress():
    plt.figure(figsize=(12, 12))
    display_list = [r[0] for r in [model.generate(), model.generate()]]
    title = ['Input Image', 'Generated Image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def train():
    EPOCHS = 40
    for epoch in range(EPOCHS):

        start = time.time()

        n = 0
        for input in train_input:
            model.train_step(input)
            if n % 10 == 0:
                print(".", end="")
            n += 1

        out = ""
        if (epoch + 1) % 5 == 0:
            print(checkpoint_manager.save(), end=" ")
            check_progress()

        print(out + str(time.time() - start))
