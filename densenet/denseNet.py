import logging
import os
import timeit

import pandas as pd

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Activation, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.python.keras.models import Model
import tensorflow_datasets as tfds
import sys
import matplotlib.pyplot as plt
from sklearn import metrics


class Results:
    def __init__(self):
        super().__init__()
        self.resultados_df = pd.DataFrame(data=[], columns=[
            'teste', 'input_shape', 'num_classes', 'num_blocks', 'num_layers', 'growth_rate', 'compression_factor', 'epochs', 'batch_size', 'accuracy', 'loss', 'tempo'
        ])

    def add(self, test, input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size, accuracy, loss, tempo):
        new_df = pd.Series(
            [test, input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size, accuracy, loss, tempo],
            index=['teste', 'input_shape', 'num_classes', 'num_blocks', 'num_layers', 'growth_rate', 'compression_factor', 'epochs', 'batch_size', 'accuracy', 'loss', 'tempo'])
        self.resultados_df = pd.concat([self.resultados_df, new_df.to_frame().T], ignore_index=True)

    def get(self):
        return self.resultados_df


def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        # Batch normalization
        y = tf.keras.layers.experimental.SyncBatchNormalization()(x)
        y = Activation('relu')(y)
        # Convolutional layer
        y = Conv2D(growth_rate, kernel_size=(3, 3), padding='same')(y)
        # Concatenate with the input
        x = Concatenate()([x, y])
    return x


def transition_block(x, compression_factor):
    # Batch normalization
    x = tf.keras.layers.experimental.SyncBatchNormalization()(x)
    x = Activation('relu')(x)
    # Convolutional layer
    num_filters = int(x.shape[-1] * compression_factor)
    x = Conv2D(num_filters, kernel_size=(1, 1))(x)
    # Average pooling
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x


def densenet(input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor):
    # Input layer
    inputs = Input(shape=input_shape)
    # Initial Convolutional layer
    x = Conv2D(2 * growth_rate, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.experimental.SyncBatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Dense blocks and transition blocks
    for i in range(num_blocks - 1):
        # Dense block
        x = dense_block(x, num_layers, growth_rate)
        # Transition block
        x = transition_block(x, compression_factor)

    # Final dense block
    x = dense_block(x, num_layers, growth_rate)
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model


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


def dense_net(input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size, test):
    # Create DenseNet model
    model = densenet(input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor)

    # Modelo
    model.summary()
    if save_model:
        img_file = 'model_denseNet.png'
        tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)

    # training
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, Cat_test_y))
    if save_model:
        try:
            model.save('denseNet.h5')
        except TypeError as e:
            print('save model failed')

    # set the matplotlib backend so figures can be saved in the background
    # plot the training loss and accuracy
    print("Generating plots...")
    sys.stdout.flush()
    # matplotlib.use("Agg")
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Bee Image Classification - " + test)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.tight_layout()
    # plt.savefig("../img/denseNet_" + test + ".png")
    plt.show()

    return model


def run(inputs):
    ini = timeit.default_timer()
    model = dense_net(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8])
    fim = timeit.default_timer()

    score = model.evaluate(X_test, Cat_test_y)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    label_pred = model.predict(X_test)

    pred = []
    for i in range(len(label_pred)):
        pred.append(np.argmax(label_pred[i]))

    Y_test = np.argmax(Cat_test_y, axis=1)  # Convert one-hot to index

    print(metrics.classification_report(Y_test, pred))

    label_pred = model.predict(X_test)

    pred = []
    for i in range(len(label_pred)):
        pred.append(np.argmax(label_pred[i]))

    Y_test = np.argmax(Cat_test_y, axis=1)  # Convert one-hot to index

    print(metrics.accuracy_score(Y_test, pred))

    # adiciona uma linha aos resultados
    results.add(inputs[8], input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size, score[1], score[0], fim - ini)


save_model = True
if __name__ == "__main__":
    # iniciar o objeto dos resultados
    results = Results()
    # conjunto de dados
    ds, ds_info = tfds.load('bee_dataset/bee_dataset_150', split='train', as_supervised=True, with_info=True)
    assert isinstance(ds, tf.data.Dataset)
    fig = tfds.show_examples(ds, ds_info)
    fig.tight_layout()
    # fig.savefig('../img/bee_example.png')

    images, labels = dataset_to_numpy(ds)

    res = np.array(labels)
    unique, counts = np.unique(res, return_counts=True)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(np.unique(labels_encoded), counts)
    plt.ylabel("Frequência")
    plt.xlabel("Classes")
    plt.title("Distribuição das classes geradas")
    plt.xticks(np.unique(labels_encoded))
    plt.tight_layout()
    # plt.savefig('../img/distribuicao.png')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.3, random_state=42)

    Cat_test_y = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    print("X_train shape : ", X_train.shape)
    print("y_train shape : ", y_train.shape)
    print("X_test shape : ", X_test.shape)
    print("y_test shape : ", y_test.shape)
    print("Unique labels : ", np.unique(labels_encoded))

    # Define input shape, number of classes, and hyperparameters
    input_shape = (150, 75, 3)
    num_classes = 9

    num_blocks = [3, 4, 5]  # 3
    num_layers = [5, 6]  # 4
    growth_rate = [8, 16]  # 32
    compression_factor = [0.3, 0.7]  # 0.5

    batch_size = [8, 16]  # 32
    epochs = [5, 15]  # 10

    for i in num_blocks:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("num_blocks test")
        # input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size
        aux = [input_shape, num_classes, i, 4, 32, 0.5, 10, 32, "num_blocks_" + str(i)]
        run(aux)

    for i in num_layers:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("num_layers test")
        # input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size
        aux = [input_shape, num_classes, 3, i, 32, 0.5, 10, 32, "num_layers_" + str(i)]
        run(aux)

    for i in growth_rate:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("growth_rate test")
        # input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size
        aux = [input_shape, num_classes, 3, 4, i, 0.5, 10, 32, "growth_rate_" + str(i)]
        run(aux)

    for i in compression_factor:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("compression_factor test")
        # input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size
        aux = [input_shape, num_classes, 3, 4, 32, i, 10, 32, "compression_factor_" + str(i)]
        run(aux)

    for i in batch_size:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("batch_size test")
        # input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size
        aux = [input_shape, num_classes, 3, 4, 32, 0.5, 10, i, "batch_size_" + str(i)]
        run(aux)

    for i in epochs:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("epochs test")
        # input_shape, num_classes, num_blocks, num_layers, growth_rate, compression_factor, epochs, batch_size
        aux = [input_shape, num_classes, 3, 4, 32, 0.5, i, 32, "epochs_" + str(i)]
        run(aux)

    df = results.get()
    print(df)
