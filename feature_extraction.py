import numpy as np
import pandas as pd
import glob
import random
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, LeakyReLU
from keras.optimizers import RMSprop, Adam, Adadelta
import urllib.request
import io
import numpy as np
import random
import scipy.misc
from keras import optimizers
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
import pickle
import shutil
import os
import traceback
from scipy.ndimage import gaussian_gradient_magnitude
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from scipy import misc
import lightgbm
import catboost
import keras
from keras.models import Sequential
from keras.layers import Dense


max_images = 1000
sample_per_image = 10000
files_loc = 'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_input_image_and_masks():
    folders = glob.glob(files_loc + 'stage1_train/*/')
    random.shuffle(folders)

    for folder in folders:
        try:
            image_location = glob.glob(folder + 'images/*')[0]
            mask_locations = glob.glob(folder + 'masks/*')
            image = Image.open(image_location).convert('LA')
            np_image = np.array(image.getdata())[:,0]
            np_image = np_image.reshape(image.size[0], image.size[1])
            np_gradient_image = gaussian_gradient_magnitude(np_image, sigma=.4)
            # np_image = imageio.imread(image_location)

            masks = []
            for i in mask_locations:
                mask_image = Image.open(i)
                np_mask = np.array(mask_image.getdata())
                np_mask = np_mask.reshape(image.size[0], image.size[1])
                masks.append(np_mask)
                # masks.append(imageio.imread(i))
        except OSError:
            continue

        yield np_image, np_gradient_image, masks


def get_image_array(image, size, x, y, masks):
    output_dict = dict()
    output_dict['image'] = image[int(x-size/2):int(x+size/2), int(y-size/2):int(y+size/2)]
    output_dict['image'] = np.expand_dims(output_dict['image'], axis = 2)
    output = 0
    for i in masks:
        if i[x][y] > 0:
            output = 1
    output_dict['output'] = output
    if output_dict['image'].shape != (size, size, 1):
        return None
    else:
        return output_dict


def get_image_arrays(image, size, masks):
    inputs = []
    result_dicts = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            inputs.append((i,j))
    if len(inputs) > sample_per_image:
        inputs = random.sample(inputs, sample_per_image)
    for i in inputs:
        result_dicts.append(
            get_image_array(image, size, i[0], i[1], masks))
    result_dicts = [i for i in result_dicts if i]
    return pd.DataFrame.from_dict(result_dicts)


def train_nn_classifier_keras(x_train, x_test, y_train, y_test, name = None, retrain = True):

    model = Sequential()
    print(x_train.shape, x_test.shape)
    model.add(Dense(2000, input_dim=x_train.shape[1], activation='selu'))
    model.add(Dense(2000, activation='selu'))
    model.add(Dense(2000, activation='selu'))
    model.add(Dense(2000, activation='selu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=1000)
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def train_nn_classifier(x_train, x_test, y_train, y_test, max_iter = 10,
                        nn_shape = (2000, 2000,), activation = 'tanh', name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')

    clf = MLPClassifier(hidden_layer_sizes=nn_shape, activation=activation, max_iter=max_iter)
    clf.fit(x_train, y_train)
    # with open(name + '.plk', 'wb') as model_file:
    #     pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}


def train_et_classifier(x_train, x_test, y_train, y_test, n_estimators = 500, max_features = 'auto', name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')
    clf = ExtraTreesClassifier(n_estimators=n_estimators, max_features = max_features, n_jobs=-2)
    clf.fit(x_train, y_train)
    with open(files_loc+name + '.plk', 'wb') as model_file:
        pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}


def train_lightgbm_classifier(x_train, x_test, y_train, y_test, name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')
    clf = lightgbm.LGBMClassifier()
    clf.fit(x_train, y_train)
    with open(files_loc+name + '.plk', 'wb') as model_file:
        pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}

def train_catboost_classifier(x_train, x_test, y_train, y_test, name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')
    clf = catboost.CatBoostClassifier(verbose=False)
    clf.fit(x_train, y_train)
    with open(files_loc+name + '.plk', 'wb') as model_file:
        pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}


def get_cnn():


    model = Sequential()

    model.add(Conv2D(16, (2, 2), input_shape=(32, 32, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(2))

    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.5, decay=1e-5, momentum=0.0, nesterov=False)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
    return model


def get_dataframes(square_size):
    gen = generate_input_image_and_masks()
    dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            image, image_gm, masks = next(gen)
            #dfs.append(create_image_features(image, image_gm, masks, square_size))
            dfs.append(get_image_arrays(image, square_size, masks))
        except StopIteration:
            traceback.print_exc()
            break

    print('images read')
    df = pd.concat(dfs, ignore_index=True)

    positive_matches = df[df['output'] > 0]
    negative_matches = df[df['output'] == 0]

    print(positive_matches.shape, negative_matches.shape)
    negative_matches = negative_matches.sample(n=positive_matches.shape[0])
    df = pd.concat([positive_matches, negative_matches], ignore_index=True)
    df = df.sample(frac=1)
    return df


def get_model_inputs(df, x_labels=['image'], test_size = 0.01):
    print('testing inputs: {0}'.format(x_labels))
    x, y = [], []
    for _, i in df.iterrows():
        x.append(np.hstack([i[x_label] for x_label in x_labels]))
        #x.append(np.hstack([i['image'], i['general_image_stats']]))
        y.append(i['output'])



    x = np.array(x)
    y = np.vstack(y)

    x = np.nan_to_num(x)

    print('arrays processed')

    print(x.shape, y.shape)

    x_train = x[0:int((1-test_size)*x.shape[0])]
    x_test = x[int((1-test_size)*x.shape[0]):]
    y_train = y[0:int((1-test_size)*x.shape[0])]
    y_test = y[int((1-test_size)*x.shape[0]):]
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


    print('inputs preprocessed')

    return x_train, x_test, y_train, y_test


def image_clustering(image_path, output_path):

    scipy_image = misc.imread(image_path)
    mask = scipy_image.astype(bool)
    graph = image.img_to_graph(scipy_image, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())
    labels = spectral_clustering(graph)
    label_im = -np.ones(mask.shape)
    label_im[mask] = labels
    print()

    # plt.imshow(scipy_image)
    # plt.show()


def test_nn():

    n = 14

    #activations = ['relu', 'tanh']
    n_estimators = [100, 250, 500, 1000]
    max_features = ['sqrt', 'log2', None]


    df = get_dataframes(n)
    x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image', 'general_image_stats'])
    for s in n_estimators:
        for m in max_features:
            print(s, m)
            #train_models(x_train, x_test, y_train, y_test)
            train_et_classifier(x_train, x_test, y_train, y_test, name='NN1', n_estimators=s, max_features=m)



def train_models(x_train, x_test, y_train, y_test):
    train_nn_classifier_keras(x_train, x_test, y_train, y_test)
    train_et_classifier(x_train, x_test, y_train, y_test, name='ET1')
    train_lightgbm_classifier(x_train, x_test, y_train, y_test, name='LGBM1')
    #train_nn_classifier(x_train, x_test, y_train, y_test, name='NN1')
    train_catboost_classifier(x_train, x_test, y_train, y_test, name='CAT1')


def get_objects( ):
    pass


def main():
    df = get_dataframes(32)
    x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image'])
    #train_models(x_train, x_test, y_train, y_test)

    model = get_cnn()

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)


    # with open(files_loc + 'RF1' + '.plk', 'rb') as model_file:
    #     clf = pickle.load(model_file)

    # folders = glob.glob(files_loc + 'stage1_train/*/')
    # random.shuffle(folders)
    #
    # for folder in folders:
    #     image_location = glob.glob(folder + 'images/*')[0]
    #     ensure_dir(folder + 'output/')
    #     #predict_picture(image_location,folder + 'output/', clf, 128)
    #     image_clustering(image_location, folder + 'output/')



if __name__ == '__main__':
    main()



