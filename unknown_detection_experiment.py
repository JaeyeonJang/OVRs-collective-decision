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

def generate_UD(UX, UY, indices, class_num):
    sampled_class = random.sample(list(indices), class_num)
    u_testX = np.empty((0,32,32,3))

    for i in range(class_num):
        c = sampled_class[i]
        c_X = UX[UY==c]/255.
        u_testX = np.vstack((u_testX,c_X))

    return u_testX, np.array([4]*len(u_testX))

#please set the name of data and batch size
data_name = 'svhn'
batch_size = 128

#Define model
if data_name != 'mnist':
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

    if data_name in ['cifar+10','cifar+50']:
        Nseen = 10
    elif data_name in ['cifar-10','svhn']:
        Nseen = 6
    elif data_name == 'tiny-imagenet':
        Nseen = 20
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

else:
    epoch = 30
    Nseen = 6
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
if data_name == 'mnist':
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

elif data_name == 'svhn':
    train_raw = loadmat('train_32x32.mat')
    test_raw = loadmat('test_32x32.mat')

    images_train = np.array(train_raw['X']) / 255.
    images_test = np.array(test_raw['X']) / 255.

    images_train = np.moveaxis(images_train, -1, 0)
    images_test = np.moveaxis(images_test, -1, 0)

    labels_train = train_raw['y'][:, 0]
    labels_test = test_raw['y'][:, 0]

    labels_train[labels_train == 10] = 0
    labels_test[labels_test == 10] = 0

    seq = np.random.choice(np.arange(10), 10, replace=False)
    temp_testY = np.zeros(len(labels_test))
    temp_trainY = np.zeros(len(labels_train))
    for i in range(1, 10):
        temp_testY[labels_test == seq[i]] = i
        temp_trainY[labels_train == seq[i]] = i
    trainX = images_train[temp_trainY < Nseen]
    testX, testY = images_test, temp_testY
    trainY = temp_trainY[temp_trainY < Nseen]
    trainY = np_utils.to_categorical(trainY, Nseen)

elif data_name == 'cifar-10':
    data_train, data_test = tf.keras.datasets.cifar10.load_data()
    (images_train, labels_train) = data_train
    (images_test, labels_test) = data_test
    labels_train = labels_train.ravel()
    labels_test = labels_test.ravel()

    seq = np.random.choice(np.arange(10), 10, replace=False)
    temp_testY = np.zeros(len(labels_test))
    temp_trainY = np.zeros(len(labels_train))
    for i in range(1,10):
        temp_testY[labels_test==seq[i]] = i
        temp_trainY[labels_train==seq[i]] = i
    trainX = images_train[temp_trainY<Nseen]/255.
    trainY = temp_trainY[temp_trainY<Nseen]
    testX, testY = images_test/255., temp_testY
    trainY = np_utils.to_categorical(trainY, Nseen)

elif data_name in ['cifar+10', 'cifar+50']:
    data_train, data_test = tf.keras.datasets.cifar10.load_data()

    # Parse images and labels
    (images_train, labels_train) = data_train
    (images_test, labels_test) = data_test
    labels_train = labels_train.ravel()
    labels_test = labels_test.ravel()

    known_animal_class = np.array([0, 1, 8, 9])

    trainX = np.empty((0, 32, 32, 3))
    temp_testX = np.empty((0, 32, 32, 3))
    trainY = []
    temp_testY = []

    for i in range(4):
        c = known_animal_class[i]
        c_trainX = images_train[labels_train == c] / 255.
        c_testX = images_test[labels_test == c] / 255.
        trainX = np.vstack((trainX, c_trainX))
        temp_testX = np.vstack((temp_testX, c_testX))
        trainY.extend([i] * len(c_trainX))
        temp_testY.extend([i] * len(c_testX))

    trainY = np_utils.to_categorical(np.array(trainY), 4)

    (_, _), (U_X, U_labels) = tf.keras.datasets.cifar100.load_data()
    U_labels = U_labels.ravel()
    if data_name == 'cifar+10':
        unknown_d, unknown_c = generate_UD(U_X, U_labels.ravel(), np.where(is_animal)[0], 10)
    elif data_name == 'cifar+50':
        unknown_d, unknown_c = generate_UD(U_X, U_labels.ravel(), np.arange(100),50)
    testX = np.vstack((temp_testX, unknown_d))
    testY = np.append(temp_testY, unknown_c)

elif data_name == 'tiny-imagenet':
    data_path = 'timg'
    images_train = pickle.load(open("%s/images_train.pkl" % data_path, 'rb'))
    labels_train = pickle.load(open("%s/labels_train.pkl" % data_path, 'rb'))
    images_test = pickle.load(open("%s/images_test.pkl" % data_path, 'rb'))
    labels_test = pickle.load(open("%s/labels_test.pkl" % data_path, 'rb'))

    seq = np.random.choice(np.arange(200), 200, replace=False)
    temp_testY = np.zeros(len(labels_test))
    temp_trainY = np.zeros(len(labels_train))
    for i in range(1, 200):
        temp_testY[labels_test == seq[i]] = i
        temp_trainY[labels_train == seq[i]] = i
    trainX = images_train[temp_trainY < Nseen]
    testX, testY = images_test, temp_testY
    trainY = temp_trainY[temp_trainY < Nseen]
    trainY = np_utils.to_categorical(trainY, Nseen)

#Training
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,zoom_range=0.05, shear_range=0.15, horizontal_flip=True)
datagen.fit(trainX)
it = datagen.flow(trainX, trainY, batch_size=batch_size)
ovrs_net.fit_generator(it, steps_per_epoch=trainX.shape[0] // batch_size, epochs=epoch)

#model setting and producing logit
ovrs_score_model = Model(ovrs_net.input, ovrs_net.get_layer(index=-2).output) #model for producing logit
logit = ovrs_score_model.predict(testX)

#prediction for closed-set classification
pred = logit.argmax(1)
acc = accuracy_score(testY[testY<Nseen], pred[testY<Nseen])

#calculating collective decision score and getting AUROC score
cds_score = get_scores(logit)
confidence_score = cds_score.max(1)
AUC_class = np.zeros(len(testY))
AUC_class[testY < Nseen] = 1
roc_score = roc_auc_score(AUC_class, normalization(confidence_score))

print(data_name + ": roc_score" + str(roc_score)+ ", ACC" + str(acc))
