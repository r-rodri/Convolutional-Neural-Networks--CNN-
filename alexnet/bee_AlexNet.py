# -*- coding: utf-8 -*-
"""
CNN AlexNet
"""

import os
import time
import warnings
import logging

import matplotlib.pyplot as plt
import numpy as np
# Bibliotecas
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import tensorflow_datasets as tfds

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def get_label2(label):
    _label = ''
    for l, v in label.items():
        if v == 1.0:
            _label = _label + ' ' + l

    if _label == '':
        _label = 'sem classe'

    return _label.strip()


def dataset_to_numpy(_ds):
    """
    Convert tensorflow dataset to numpy arrays
    """
    images = []
    labels = []

    # Iterate over a dataset
    for i, (image, label) in enumerate(tfds.as_numpy(_ds)):
        images.append(image)
        labels.append(get_label2(label))

    for i, img in enumerate(images):
        if i < 3:
            print(img.shape, labels[i])

    return np.array(images), labels


if __name__ == "__main__":
    epochs_list = [5, 10, 15]  # Number of epochs for each run

    for epochs in epochs_list:

        start_time = time.time()

        # conjunto de dados
        ds, ds_info = tfds.load('bee_dataset/bee_dataset_150', split='train', as_supervised=True, with_info=True)
        assert isinstance(ds, tf.data.Dataset)
        # fig = tfds.show_examples(ds, ds_info)
        # fig.show()

        images, labels = dataset_to_numpy(ds)
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        x_train, x_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.3, random_state=42)

        print("X_train shape : ", x_train.shape)
        print("y_train shape : ", y_train.shape)
        print("X_test shape : ", x_test.shape)
        print("y_test shape : ", y_test.shape)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # plt.figure(figsize=(20, 20))
        # for i, (image, label) in enumerate(train_ds.take(5)):
        #     ax = plt.subplot(5, 5, i + 1)
        #     plt.imshow(image)
        #     plt.title(CLASS_NAMES[label.numpy()[0]])
        #     plt.axis('off')
        # plt.show()

        # Data pipeline
        train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
        test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
        print("Training data size:", train_ds_size)
        print("Test data size:", test_ds_size)

        train_ds = (train_ds
                    .map(process_images)
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))
        test_ds = (test_ds
                   .map(process_images)
                   .shuffle(buffer_size=train_ds_size)
                   .batch(batch_size=32, drop_remainder=True))

        # Definicao do modelo
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='leaky_relu', input_shape=(227, 227, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='leaky_relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='leaky_relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='leaky_relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='leaky_relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='leaky_relu'),
            keras.layers.Dropout(0.7),
            keras.layers.Dense(4096, activation='leaky_relu'),
            keras.layers.Dropout(0.7),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Modelo
        img_file = '../modelo/model_AlexNet.png'
        tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)

        # Tensorboard
        root_logdir = os.path.join(os.curdir, "logs\\fit\\")
        run_logdir = get_run_logdir()
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

        # Treino e resultados
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        print(model.summary())
        model.fit(train_ds,
                  epochs=epochs,
                  validation_data=test_ds,
                  validation_freq=1,
                  callbacks=[tensorboard_cb])

        # Guardar os parâmetros do modelo
        model.save("alexnet_PlusDropout.h5")

        # Avaliação
        end_time = time.time()
        total_time = end_time - start_time
        score = model.evaluate(test_ds)

        print(f'Epochs: {epochs}')
        print('Test Loss:', score[0])
        print('Test accuracy:', score[1])
        print(f'Total time: {total_time} seconds')
