import time
import tensorflow as tf
from matplotlib import pyplot as plt
from util import CheckpointManager, load_jpeg_folder_split
from cyclegan import CycleGAN, preprocess as cg_preprocess

# Initialize our model
model = CycleGAN()

# Try to resume a previous session
checkpoint_manager = CheckpointManager(model, 'checkpoints/cyclegan/train')
if checkpoint_manager.load_latest():
    print("Successfully loaded checkpoint!")

# Load in two sets of images, 10% of each will be used as "test" data
train_x, test_x = load_jpeg_folder_split('./dataset/gloss_liquid_copper', test_portion=0.1)
train_y, test_y = load_jpeg_folder_split('./dataset/satin_battleship_gray', test_portion=0.1)


# Preprocess the images and shuffle them
def preprocess_ds(ds: tf.data.Dataset, preprocessor, buffer_size=10) -> tf.data.Dataset:
    return ds \
        .map(preprocessor, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .cache().shuffle(buffer_size).batch(1)


train_x = preprocess_ds(train_x, cg_preprocess.preprocess_image_train)
train_y = preprocess_ds(train_y, cg_preprocess.preprocess_image_train)
test_x = preprocess_ds(test_x, cg_preprocess.preprocess_image_test)
test_y = preprocess_ds(test_y, cg_preprocess.preprocess_image_test)

# Setup a progress checker so we can watch the results as we train
sample_x = next(iter(train_x))
sample_y = next(iter(train_y))


def check_progress():
    for test_input, generator in [(sample_x, model.generate_y), (sample_x, model.generate_x)]:
        test_generation = generator(test_input)
        plt.figure(figsize=(12, 12))
        display_list = [test_input[0], test_generation[0]]
        title = ['Input Image', 'Generated Image']
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()


EPOCHS = 40
for epoch in range(EPOCHS):

    start = time.time()

    n = 0
    for x, y in tf.data.Dataset.zip((train_x, train_y)):
        model.train_step(x, y)
        if n % 10 == 0:
            print(".")
        n += 1

    out = ""
    if (epoch + 1) % 5 == 0:
        print(checkpoint_manager.save() + " ")
        check_progress()

    print(out + str(time.time() - start))
