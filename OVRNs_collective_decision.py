import numpy as np
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Dropout, Activation, MaxPool2D, concatenate, Flatten, BatchNormalization
from tensorflow.python.keras.models import Model
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def get_scores(logit_mat):
    #collective decision score computation
    Nseen = logit_mat.shape[1]
    score_mat = np.zeros((logit_mat.shape[0], Nseen))
    for k in range(Nseen):
        score_mat[:, k] = logit_mat[:, k] - (logit_mat.sum(1) - logit_mat[:, k])/(Nseen-1)
    return score_mat

#Define model

Nseen = 10

ovrs_net = Sequential()
ovrs_net.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3),
                   kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(MaxPool2D(strides=2))
ovrs_net.add(Dropout(0.2))

ovrs_net.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(MaxPool2D(strides=2))
ovrs_net.add(Dropout(0.3))

ovrs_net.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer="glorot_uniform"))
ovrs_net.add(BatchNormalization())
ovrs_net.add(Activation('relu'))
ovrs_net.add(MaxPool2D(strides=2))
ovrs_net.add(Dropout(0.4))

ovrs_net.add(Flatten())
ovrs_net.add(Dense(512, activation='relu', kernel_initializer="glorot_uniform"))

X = Input(shape=(32, 32, 3))
In = ovrs_net(X)
outputs = list()
for i in range(Nseen):
    c_features = Dense(128, activation='relu', kernel_initializer="glorot_uniform")(In)
    output = Dense(1, kernel_initializer="glorot_uniform")(c_features)
    outputs.append(output)

output = concatenate(outputs)
output = Activation('sigmoid')(output)

ovrs_net = Model(X, output)
opt = Adam(lr=0.0002, beta_1=0.5)
ovrs_net.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

#data setting

data_train, data_test = tf.keras.datasets.cifar10.load_data()

(images_train, labels_train) = data_train
(images_test, labels_test) = data_test
labels_train = labels_train.ravel()
labels_test = labels_test.ravel()

data_train, data_test = tf.keras.datasets.cifar100.load_data()
(images_cifar100, _) = data_test


trainX = images_train/ 255.
trainY = labels_train
testX = images_test/ 255.
testX = np.vstack((testX, images_cifar100/255.))


testY = np.append(labels_test, [Nseen] * len(images_cifar100))
trainY = np_utils.to_categorical(trainY, Nseen)
testY = np_utils.to_categorical(testY, Nseen+1)
trainX = trainX.reshape(trainX.shape[0], 32, 32, 3)
testX = testX.reshape(testX.shape[0], 32, 32, 3)

#Training

epoch = 150
batch_size = 128
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,zoom_range=0.05, shear_range=0.15, horizontal_flip=True)
datagen.fit(trainX)
it = datagen.flow(trainX, trainY, batch_size=batch_size)
ovrs_net.fit_generator(it, steps_per_epoch=trainX.shape[0] // batch_size, epochs=epoch)

ovrs_score_model = Model(ovrs_net.input, ovrs_net.get_layer(index=-2).output) #model for producing logit
logit = ovrs_score_model.predict(testX)
cds_score = get_scores(logit)


