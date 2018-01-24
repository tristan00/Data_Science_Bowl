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
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
import pickle
import shutil
import os
import traceback
#from dnn import DNN
from scipy.misc import imsave
from sklearn.feature_extraction import image
from scipy import ndimage
from sklearn.cluster import spectral_clustering
from scipy import misc
import matplotlib.pyplot as plt
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
            image = Image.open(image_location)
            np_image = np.array(image.getdata())
            np_image = np_image.reshape(image.size[0], image.size[1], 4)
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

        yield np_image, masks


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


def get_features_of_point(x, y, image, image_gradient, masks, square_size, general_image_stats):
    x_list = []

    image_list = []
    gradient_list = []

    for i in range(x - square_size//2, 1 + x + square_size//2):
        for j in range(y - square_size // 2, 1 + y + square_size // 2):
            if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
                image_list.append(np.full([4,], np.nan))
                for g in image_gradient:
                    gradient_list.append(np.full([4, ], np.nan))
            else:
                image_list.append(image[i][j])
                for g in image_gradient:
                    gradient_list.append(g[i][j])
    try:
        gradient_list = np.hstack(gradient_list)
    except ValueError:
        gradient_list = np.array([])
    image_list = np.hstack(image_list)

    #print(image_list.shape, gradient_list.shape, general_image_stats.shape)


    y = 0
    for i in masks:
        if i[x][y] > 0:
            y = 1

    return {'output': y, 'image': image_list, 'gradients': gradient_list, 'general_image_stats':general_image_stats}



def create_image_features(image, masks, square_size):
    result_dicts = []

    inputs = []
    image_gradient = np.gradient(image)
    mean_array = np.array([np.mean(image[:,:,0]),
                           np.mean(image[:, :, 1]),
                           np.mean(image[:, :, 2]),
                           np.mean(image[:, :, 3])])
    std_array = np.array([np.std(image[:, :, 0]),
                           np.std(image[:, :, 1]),
                           np.std(image[:, :, 2]),
                           np.std(image[:, :, 3])])
    median_array = np.array([np.median(image[:, :, 0]),
                           np.median(image[:, :, 1]),
                           np.median(image[:, :, 2]),
                           np.median(image[:, :, 3])])
    np_histograms = np.hstack(np.array([np.histogram(image[:, :, 0])[0],
                              np.histogram(image[:, :, 0])[1],
                              np.histogram(image[:, :, 1])[0],
                              np.histogram(image[:, :, 1])[1],
                              np.histogram(image[:, :, 2])[0],
                              np.histogram(image[:, :, 2])[1],
                              np.histogram(image[:, :, 3])[0],
                              np.histogram(image[:, :, 3])[1]]))

    general_image_stats = np.hstack([mean_array, std_array, median_array, np_histograms])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            inputs.append((i,j))
    if len(inputs) > sample_per_image:
        inputs = random.sample(inputs, sample_per_image)

    for i in inputs:
        result_dicts.append(get_features_of_point(i[0], i[1], image, image_gradient, masks, square_size, general_image_stats))

    return pd.DataFrame.from_dict(result_dicts)


def train_nn_classifier(x_train, x_test, y_train, y_test, max_iter = 200,
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


def train_adaboost_classifier(x_train, x_test, y_train, y_test, n_estimators = 500, name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(x_train, y_train)
    # with open(name + '.plk', 'wb') as model_file:
    #     pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}



def train_rf_classifier(x_train, x_test, y_train, y_test, n_estimators = 100,
                        max_depth = None, max_features = None, name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=-1)
    clf.fit(x_train, y_train)
    with open(files_loc+name + '.plk', 'wb') as model_file:
        pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}

def train_et_classifier(x_train, x_test, y_train, y_test, n_estimators = 500, name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')
    clf = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-2)
    clf.fit(x_train, y_train)
    with open(files_loc+name + '.plk', 'wb') as model_file:
        pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}


def train_gb_classifier(x_train, x_test, y_train, y_test, n_estimators = 500, name = None, retrain = True):
    if not retrain:
        try:
            with open(name + '.plk') as model_file:
                clf = pickle.load(model_file)
            return {'model':clf}
        except IOError:
            print('model not found, retraining')
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
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
        try:
            image, masks = next(gen)
            dfs.append(create_image_features(image, masks, square_size))
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


def train_models(x_train, x_test, y_train, y_test):
    #train_rf_classifier(x_train, x_test, y_train, y_test, name = 'RF1')
    #train_adaboost_classifier(x_train, x_test, y_train, y_test, name='ADA1')
    train_nn_classifier(x_train, x_test, y_train, y_test, name='NN1')
    #train_et_classifier(x_train, x_test, y_train, y_test, name='ET1')
    #train_gb_classifier(x_train, x_test, y_train, y_test, name='GB1')


def predict_picture(image_location, output_folder, clf, cuttoff):

    input_image = Image.open(image_location)
    input_image.save(output_folder + 'input.png')

    np_image = np.array(input_image.getdata())
    np_image = np_image.reshape(input_image.size[0], input_image.size[1], 4)
    input_image = np_image

    mean_array = np.array([np.mean(input_image[:,:,0]),
                           np.mean(input_image[:, :, 1]),
                           np.mean(input_image[:, :, 2]),
                           np.mean(input_image[:, :, 3])])
    std_array = np.array([np.std(input_image[:, :, 0]),
                           np.std(input_image[:, :, 1]),
                           np.std(input_image[:, :, 2]),
                           np.std(input_image[:, :, 3])])
    median_array = np.array([np.median(input_image[:, :, 0]),
                           np.median(input_image[:, :, 1]),
                           np.median(input_image[:, :, 2]),
                           np.median(input_image[:, :, 3])])
    np_histograms = np.hstack(np.array([np.histogram(input_image[:, :, 0])[0],
                              np.histogram(input_image[:, :, 0])[1],
                              np.histogram(input_image[:, :, 1])[0],
                              np.histogram(input_image[:, :, 1])[1],
                              np.histogram(input_image[:, :, 2])[0],
                              np.histogram(input_image[:, :, 2])[1],
                              np.histogram(input_image[:, :, 3])[0],
                              np.histogram(input_image[:, :, 3])[1]]))

    general_image_stats = np.hstack([mean_array, std_array, median_array, np_histograms])

    output = []
    for i in range(input_image.shape[0]):
        temp = []
        for j in range(input_image.shape[1]):
            temp.append(0)
        output.append(temp)
    output_image = np.array(output)

    with open(files_loc + 'scaler' + '.plk', 'rb') as model_file:
        scaler = pickle.load(model_file)


    output_vector = []
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            features = get_features_of_point(i, j, input_image, [], [], 12, general_image_stats)
            features =np.hstack([features['image'], features['general_image_stats']])
            features = np.nan_to_num(features)
            output_vector.append(features)
    output_vector = np.vstack(output_vector)
    output_vector = scaler.transform(output_vector)

    output_vector = clf.predict(output_vector)
    output_vector = output_vector.reshape(np_image.shape[0], np_image.shape[1])*255
    #output_image = Image.fromarray(output_vector)
    #output_image.save(output_folder + 'output.png')
    graph = image.img_to_graph(output_vector)

    labels = spectral_clustering(graph)
    print(labels)

    imsave(output_folder + 'output.png', output_vector)
    print(output_folder)
    # for i in rmultiply arrayange(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         features = get_features_of_point(i, j, image, [], [], 12, general_image_stats)
    #         features =np.hstack([features['image'], features['general_image_stats']])
    #         features = np.reshape(features, (1, -1))
    #         features = np.nan_to_num(features)
    #         features = scaler.transform(features)
    #         output_image[i,j] = 255 if clf.predict_proba(features)[0][1] > cuttoff else 0
    #     print(i)
    # print(1)
    # output_image = Image.fromarray(output_image)
    # output_image.save(output_folder + 'output.png')

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


def main():
    df = get_dataframes(16)
    x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image', 'general_image_stats'])
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


def test_nn():

    n = 8

    shapes = [(2000, ), (2000, 2000,), (3000,), (3000, 3000,), (2000,2000,2000,)]
    activations = ['relu', 'tanh']
    max_epochs = [1, 5]

    df = get_dataframes(n)
    x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image', 'general_image_stats', 'gradients'])
    for s in shapes:
        for a in activations:
            for m in max_epochs:


                print(s, a, m)
                #train_models(x_train, x_test, y_train, y_test)
                train_nn_classifier(x_train, x_test, y_train, y_test, name='NN1', nn_shape=s, activation=a, max_iter=m)


if __name__ == '__main__':
    test_nn()



