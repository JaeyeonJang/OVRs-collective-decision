import numpy as np
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Dropout, Activation, MaxPool2D, concatenate, Flatten, BatchNormalization
from tensorflow.python.keras.models import Model
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.io import loadmat
from cifar100 import is_animal
import random
import pickle
from PIL import Image
import os


def get_scores(logit_mat):
    #collective decision score computation
    Nseen = logit_mat.shape[1]
    score_mat = np.zeros((logit_mat.shape[0], Nseen))
    for k in range(Nseen):
        score_mat[:, k] = logit_mat[:, k] - (logit_mat.sum(1) - logit_mat[:, k])/(Nseen-1)
    return score_mat

def normalization(array):
    return (array - array.min()) / (array.max() - array.min())

#set epsilon (proportion of known samples to be considered known when setting testing threshold)
epsilon = 0.95

#set unknown data name
unknown_name = 'ImageNet-Crop'

#network definition
batch_size = 128
Nseen = 10
epoch = 150
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
ovrs_net.compile(loss="binary_crossentropy", optimizer=opt)



#data setting

# test_names = ['ImageNet-Crop', 'ImageNet-Resize', 'LSUN-Crop', 'LSUN-Resize']
unknown_set = dict()
if unknown_name == 'ImageNet-Crop':
    path = 'Imagenet_crop'
    images = os.listdir(path)
    unknown_set[unknown_name] = []
    for img in images:
        img_path = os.path.join(path, img)
        foo = Image.open(img_path)
        foo = foo.resize((32, 32), Image.ANTIALIAS)
        img_array = np.array(foo)
        img_array = img_array/255.
        if img_array.shape[-1] != 3:
            temp_array = np.zeros((32,32,3))
            for i in range(3):
                temp_array[:, :, i] = img_array
            img_array = temp_array
        unknown_set[unknown_name].append(img_array)
    unknown_set[unknown_name] = np.array(unknown_set[unknown_name])
elif unknown_name == 'ImageNet-Resize':
    path = 'Imagenet_resize'
    images = os.listdir(path)
    unknown_set[unknown_name] = []
    for img in images:
        img_path = os.path.join(path, img)
        foo = Image.open(img_path)
        foo = foo.resize((32, 32), Image.ANTIALIAS)
        img_array = np.array(foo)
        img_array = img_array/255.
        if img_array.shape[-1] != 3:
            temp_array = np.zeros((32,32,3))
            for i in range(3):
                temp_array[:, :, i] = img_array
            img_array = temp_array
        unknown_set[unknown_name].append(img_array)
    unknown_set[unknown_name] = np.array(unknown_set[unknown_name])
elif unknown_name == 'LSUN-Crop':
    path = 'LSUN_crop'
    images = os.listdir(path)
    unknown_set[unknown_name] = []
    for img in images:
        img_path = os.path.join(path, img)
        foo = Image.open(img_path)
        foo = foo.resize((32, 32), Image.ANTIALIAS)
        img_array = np.array(foo)
        img_array = img_array/255.
        if img_array.shape[-1] != 3:
            temp_array = np.zeros((32,32,3))
            for i in range(3):
                temp_array[:, :, i] = img_array
            img_array = temp_array
        unknown_set[unknown_name].append(img_array)
    unknown_set[unknown_name] = np.array(unknown_set[unknown_name])
elif unknown_name == 'LSUN-Resize':
    path = 'LSUN_resize'
    images = os.listdir(path)
    unknown_set[unknown_name] = []
    for img in images:
        img_path = os.path.join(path, img)
        foo = Image.open(img_path)
        foo = foo.resize((32, 32), Image.ANTIALIAS)
        img_array = np.array(foo)
        img_array = img_array/255.
        if img_array.shape[-1] != 3:
            temp_array = np.zeros((32,32,3))
            for i in range(3):
                temp_array[:, :, i] = img_array
            img_array = temp_array
        unknown_set[unknown_name].append(img_array)
    unknown_set[unknown_name] = np.array(unknown_set[unknown_name])

data_train, data_test = tf.keras.datasets.cifar10.load_data()

# Parse images and labels
(images_train, labels_train) = data_train
(images_test, labels_test) = data_test
labels_train = labels_train.ravel()
labels_test = labels_test.ravel()

trainX = images_train / 255.
trainY = labels_train
trainY = np_utils.to_categorical(trainY, Nseen)
temp_testX, temp_testY = images_test / 255., labels_test
unknown = unknown_set[unknown_name]
testX = np.vstack((temp_testX, unknown))
testY = np.append(temp_testY, [Nseen] * 10000)

#Training
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,zoom_range=0.05, shear_range=0.15, horizontal_flip=True)
datagen.fit(trainX)
it = datagen.flow(trainX, trainY, batch_size=batch_size)
ovrs_net.fit_generator(it, steps_per_epoch=trainX.shape[0] // batch_size, epochs=epoch)

#model setting and producing logit
ovrs_score_model = Model(ovrs_net.input, ovrs_net.get_layer(index=-2).output) #model for producing logit
train_logit = ovrs_score_model.predict(trainX)
logit = ovrs_score_model.predict(testX)

tr_cds_score = get_scores(train_logit)
th = np.zeros(Nseen)
for i in range(Nseen):
    th[i] = np.sort(tr_cds_score[trainY.argmax(1) == i, i])[int(sum(trainY.argmax(1) == i) * (1-epsilon))]


#prediction for closed-set classification
pred = logit.argmax(1)
acc = accuracy_score(testY[testY<Nseen], pred[testY<Nseen])

#calculating collective decision score and getting macro f-measure
cds_score = get_scores(logit)
over_th = cds_score > th #over threshold?
pred = np.argmax(cds_score * over_th, 1)
pred[over_th.sum(1) == 0] = Nseen
f = f1_score(testY, pred, average='macro')

print(unknown_name + ": macro f_measure" + str(f))