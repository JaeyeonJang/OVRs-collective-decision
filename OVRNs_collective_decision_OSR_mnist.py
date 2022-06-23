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
unknown_name = 'omniglot'

#network definition
batch_size = 128
epoch = 30
Nseen = 10
ovrs_net = Sequential()
ovrs_net.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1),
                   kernel_initializer="glorot_uniform"))
ovrs_net.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="glorot_uniform"))
ovrs_net.add(MaxPool2D(strides=2))

ovrs_net.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="glorot_uniform"))
ovrs_net.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="glorot_uniform"))
ovrs_net.add(MaxPool2D(strides=2))
ovrs_net.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="glorot_uniform"))
ovrs_net.add(Flatten())
ovrs_net.add(Dense(500, activation='relu', kernel_initializer="glorot_uniform"))
X = Input(shape=(28, 28, 1,))
In = ovrs_net(X)
outputs = list()
for i in range(Nseen):
    features = Dense(64, activation='relu', kernel_initializer="glorot_uniform")(In)
    # pool = GlobalAveragePooling2D()(features)
    output = Dense(1, kernel_initializer="glorot_uniform")(features)
    outputs.append(output)

output = concatenate(outputs)
output = Activation('sigmoid')(output)
ovrs_net = Model(X, output)
opt = Adam(lr=0.0002, beta_1=0.5)
ovrs_net.compile(loss="binary_crossentropy", optimizer=opt)

#data setting
data_train, data_test = tf.keras.datasets.mnist.load_data()
# Parse images and labels
(images_train, labels_train) = data_train
(images_test, labels_test) = data_test
seq = np.random.choice(np.arange(10), 10, replace=False)
temp_testY = np.zeros(len(labels_test))
temp_trainY = np.zeros(len(labels_train))
for i in range(1, 10):
    temp_testY[labels_test == seq[i]] = i
    temp_trainY[labels_train == seq[i]] = i
trainX = images_train[temp_trainY < Nseen] / 255.
trainY = temp_trainY[temp_trainY < Nseen]
testX, testY = images_test / 255., temp_testY
trainY = np_utils.to_categorical(trainY, Nseen)
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)

temp_testX, temp_testY = images_test / 255., labels_test
temp_testX = temp_testX.reshape(temp_testX.shape[0], 28, 28, 1)

if unknown_name == 'omniglot':
    unknown = pickle.load(open("omniglot_normalized.pkl", 'rb'))
    indices = np.arange(unknown.shape[0])
    random.shuffle(indices)
    testX = np.vstack((temp_testX, unknown[indices[:10000]]))
    testY = np.append(temp_testY, [Nseen] * 10000)

elif unknown_name == 'noise':
    unknown = np.random.uniform(0, 1, (10000, 28, 28, 1))
    testX = np.vstack((temp_testX, unknown))
    testY = np.append(temp_testY, [Nseen] * 10000)

elif unknown_name == 'mnist-noise':
    unknown = temp_testX + 1 * np.random.uniform(0, 1, (10000, 28, 28, 1))
    unknown = np.clip(unknown, 0, 1)
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