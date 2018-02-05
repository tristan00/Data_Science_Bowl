import numpy as np
import pandas as pd
import glob
import random
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
import pickle
import shutil
import os
import traceback
#from dnn import DNN
from scipy.misc import imsave
from scipy.ndimage import gaussian_gradient_magnitude
from sklearn.feature_extraction import image
from scipy import ndimage
from sklearn.cluster import spectral_clustering
from scipy import misc
import matplotlib.pyplot as plt
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


    # while True:
    #
    #     folder = random.choice(folders)
    #     print(folder)
    #     image_location = glob.glob(folder + 'images/*')[0]
    #     mask_locations = glob.glob(folder + 'masks/*')
    #     image = Image.open(image_location)
    #     np_image = np.array(image.getdata())
    #     np_image = np_image.reshape(image.size[0], image.size[1], 4)
    #     print(np_image.shape)
    #     #np_image = imageio.imread(image_location)
    #
    #     masks = []
    #     for i in mask_locations:
    #         mask_image = Image.open(i)
    #         np_mask = np.array(mask_image.getdata())
    #         np_mask = np_mask.reshape(image.size[0], image.size[1])
    #         masks.append(np_mask)
    #         #masks.append(imageio.imread(i))
    #
    #     yield np_image, masks


def get_features_of_point(x, y, image, image_gradient, gradients, masks, square_size, general_image_stats):
    x_list = []

    image_list = []
    gm_list = []
    gd_list = []

    for i in range(x - square_size//2, 1 + x + square_size//2):
        for j in range(y - square_size // 2, 1 + y + square_size // 2):
            if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
                image_list.append(np.full([1,], np.nan))
                gm_list.append(np.full([1,], np.nan))
                gd_list.append(np.full([1,], np.nan))

            else:
                image_list.append(image[i][j])
                gm_list.append(image_gradient[i][j])
                gd_list.append(math.atan2(gradients[1][i][j], gradients[0][i][j]))
                #gd_list.append(np.arctan2((0,0), (gradients[0][i][j], gradients[1][i][j])))
    try:
        gd_list = np.hstack(gd_list)
        gm_list = np.hstack(gm_list)
        gradient_list = np.hstack([gd_list, gm_list])
    except ValueError:
        traceback.print_exc()
        gradient_list = np.array([])
    image_list = np.hstack(image_list)

    y = 0
    for i in masks:
        if i[x][y] > 0:
            y = 1

    return {'output': y, 'image': image_list,
            'gradients': gradient_list,
            'general_image_stats':general_image_stats}



def create_image_features(image, image_gm, masks, square_size):
    result_dicts = []

    inputs = []

    gradients = np.gradient(image)

    mean = np.mean(image[:,:])
    std = np.std(image[:, :])
    median =np.median(image[:, :])
    np_histograms = np.hstack(np.array([np.histogram(image[:, :])[0],
                              np.histogram(image[:, :])[1]]))

    general_image_stats = np.hstack([np.array([mean, std, median]), np_histograms])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            inputs.append((i,j))
    if len(inputs) > sample_per_image:
        inputs = random.sample(inputs, sample_per_image)

    for i in inputs:
        result_dicts.append(get_features_of_point(i[0], i[1], image, image_gm, gradients, masks, square_size, general_image_stats))

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



def get_dataframes(square_size):
    gen = generate_input_image_and_masks()
    dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            image, image_gm, masks = next(gen)
            dfs.append(create_image_features(image, image_gm, masks, square_size))
        except StopIteration:
            traceback.print_exc()
            break

    print('images read')
    df = pd.concat(dfs, ignore_index=True)

    positive_matches = df[df['output'] > 0]
    negative_matches = df[df['output'] == 0]
    negative_matches = negative_matches.sample(n=positive_matches.shape[0])
    df = pd.concat([positive_matches, negative_matches], ignore_index=True)
    df = df.sample(frac=1)
    return df


def get_model_inputs(df, x_labels=['image']):
    print('testing inputs: {0}'.format(x_labels))
    x, y = [], []
    for _, i in df.iterrows():
        x.append(np.hstack([i[x_label] for x_label in x_labels]))
        #x.append(np.hstack([i['image'], i['general_image_stats']]))
        y.append(i['output'])

    x = np.vstack(x)
    y = np.vstack(y)

    x = np.nan_to_num(x)
    y = np.ravel(y)

    print('arrays processed')

    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
    min_max_preprocessor = MinMaxScaler([-1,1])
    x_train = min_max_preprocessor.fit_transform(x_train)
    x_test = min_max_preprocessor.transform(x_test)
    with open(files_loc+'scaler' + '.plk', 'wb') as model_file:
        pickle.dump(min_max_preprocessor, model_file)

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


def main():
    df = get_dataframes(16)
    x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image', 'gradients', 'general_image_stats'])
    train_models(x_train, x_test, y_train, y_test)


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



