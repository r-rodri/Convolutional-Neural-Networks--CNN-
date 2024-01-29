# -*- coding: utf-8 -*-
"""
CNN LeNet
"""
import os
import time
import warnings
import logging

# Bibliotecas
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")


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

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

if __name__ == "__main__":
    
    # Conjunto de dados
    ds, ds_info = tfds.load('bee_dataset/bee_dataset_150', split='train', as_supervised=True, with_info=True)
    assert isinstance(ds, tf.data.Dataset)
    
    images, labels = dataset_to_numpy(ds)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.3, random_state=42)
    
    print('\n')
    print("X_train shape : ", x_train.shape)
    print("y_train shape : ", y_train.shape)
    print("X_test shape : ", x_test.shape)
    print("y_test shape : ", y_test.shape)
    print('\n')

    # One Hot Encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    results = []
    batch_size=[32, 128, 512]
    epochs=[1, 5, 10, 15]
    activation = ['relu','sigmoid', 'tanh']
    for activ in activation:
        # Construção do modelo
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), activation=activ, input_shape=(150, 75, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation=activ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation=activ))
        model.add(Flatten())
        model.add(Dense(120, activation=activ))
        model.add(Dense(84, activation=activ))
        model.add(Dense(10, activation='softmax'))

        # Tensorboard
        root_logdir = os.path.join(os.curdir, "logs\\fit\\")
        run_logdir = get_run_logdir()
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        
        for bat in batch_size:
            for epo in epochs:
                start_time = time.time()
                print('\n')
                print(f'--- Função de activação = {activ} ---')
                print(f'--- Batch Size = {bat} ---')
                print(f'--- Epochs = {epo} ---')
                model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
                model.fit(x_train, y_train, batch_size=bat, epochs=epo, verbose=1, validation_data=(x_test, y_test))
            
                # Avaliação
                end_time = time.time()
                total_time = (end_time - start_time)
                score = model.evaluate(x_test, y_test)
                print('\n')
                print(model.summary())
                print('\nTest Loss:', score[0])
                print('Test accuracy:', score[1])
                print('O modelo demorou: ', total_time,'seg')
                
                # Guardar os parâmetros do modelo
                model.save(f"LeNet_model_{activ}.h5")
            
                # Fazer uma previsão usando modelo construído  
                print('\n--- Usando o modelo criado para fazer uma previsão ---')
                test_image = tf.keras.utils.load_img ('../img/teste_0IpYwrd4CgrsGFU4U1TaCWaIgmCQdLjw.jpeg', target_size = (150, 75))
                test_image = tf.keras.utils.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = model.predict(test_image)
            
                class_names = le.classes_  # Nomes das classes criadas pelo LabelEncoder
            
                max_index = np.argmax(result) # Ir buscar o index que contem o valor maximo
                max_value = round(result[0][max_index]*100,2)  # Ir buscar o valor do index anterior
                # max_value = round(max_value,3)
                class_name = class_names[max_index]
                
                print("A classe predita é:",class_name)
                print("Probabilidade:", max_value,'%')
                print('\n')
                
                # Adicionar resultados ao dicionario
                result_dict = {
                     'CNN': 'LeNet',
                    'Activation': activ,
                    'Batch Size': bat,
                    'Epochs': epo,
                    'Test Loss': score[0],
                    'Test Accuracy': score[1],
                    'Tempo Total [seg]': total_time,
                    'Classe da Previsão': class_name,
                    'Probabilidade da Previsão [%]': max_value}
                
                # Adicionar o dicionario a tabela resultados
                results.append(result_dict)
        
# Criar imagem do Modelo
img_file = '../modelo/model_LeNet.png'
tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)
        
# Criar e guardar tabela
df_results = pd.DataFrame(results)
df_results.to_csv('Tabela de Resultado_LeNet.csv', index=False)

'''
Gráficos de Resultados 

'''
fig, ax1 = plt.subplots()
sns.barplot(data = df_results, x = 'Batch Size', y = 'Test Loss', hue = 'Epochs', errorbar=None).set(title = 'LeNet')
ax1.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Epochs', prop={'weight': 'bold'})
plt.savefig('Lenet_BatchLossEpochs.png', bbox_inches='tight')

fig, ax2 = plt.subplots()
sns.barplot(data = df_results, x = 'Batch Size', y = 'Test Accuracy', hue = 'Epochs', errorbar=None).set(title = 'LeNet')
ax2.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Epochs', prop={'weight': 'bold'})
plt.savefig('Lenet_BatchAccuracyEpochs.png', bbox_inches='tight')

fig, ax3 = plt.subplots()
sns.barplot(data = df_results, x = 'Epochs', y = 'Test Loss', hue = 'Batch Size', errorbar=None).set(title = 'LeNet')
ax3.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Batch Size', prop={'weight': 'bold'})
plt.savefig('Lenet_EpochsLossBatch.png', bbox_inches='tight')

fig, ax4 = plt.subplots()
sns.barplot(data = df_results, x = 'Epochs', y = 'Test Accuracy', hue = 'Batch Size', errorbar=None).set(title = 'LeNet')
ax4.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Batch Size', prop={'weight': 'bold'})
plt.savefig('Lenet_EpochsAccuracyBatch.png', bbox_inches='tight')

fig, ax5 = plt.subplots()
sns.barplot(data = df_results, x = 'Epochs', y = 'Tempo Total [seg]', hue = 'Batch Size', errorbar=None).set(title = 'LeNet')
ax5.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Batch Size', prop={'weight': 'bold'})
plt.savefig('Lenet_EpochsTimeBatch.png', bbox_inches='tight')

fig, ax6 = plt.subplots()
sns.barplot(data = df_results, x = 'Epochs', y = 'Tempo Total [seg]', hue = 'Activation', errorbar=None).set(title = 'LeNet')
ax6.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Activation', prop={'weight': 'bold'})
plt.savefig('Lenet_EpochsTimeActivation.png', bbox_inches='tight')

fig, ax7 = plt.subplots()
sns.barplot(data = df_results, x = 'Epochs', y = 'Test Loss', hue = 'Activation', errorbar=None).set(title = 'LeNet')
ax7.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Activation', prop={'weight': 'bold'})
plt.savefig('Lenet_EpochsLossActivation.png', bbox_inches='tight')

fig, ax8 = plt.subplots()
sns.barplot(data = df_results, x = 'Epochs', y = 'Test Accuracy', hue = 'Activation', errorbar=None).set(title = 'LeNet')
ax8.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Activation', prop={'weight': 'bold'})
plt.savefig('Lenet_EpochsAccuracyActivation.png', bbox_inches='tight')

fig, ax9 = plt.subplots()
sns.barplot(data = df_results, x = 'Activation', y = 'Probabilidade da Previsão [%]', hue = 'Epochs', errorbar=None).set(title = 'LeNet')
ax9.legend(bbox_to_anchor=(1.05, 0.70), loc='upper left', borderaxespad=0).set_title('Epochs', prop={'weight': 'bold'})
plt.savefig('Lenet_Probabilidade.png', bbox_inches='tight')


