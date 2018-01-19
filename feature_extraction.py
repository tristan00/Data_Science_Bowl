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
import pickle
import traceback


files_loc = 'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/'

def generate_input_image_and_masks():
    folders = glob.glob(files_loc + 'stage1_train/*/')
    random.shuffle(folders)
    folders = folders[:100]

    for folder in folders:
        image_location = glob.glob(folder + 'images/*')[0]
        mask_locations = glob.glob(folder + 'masks/*')
        image = Image.open(image_location)
        np_image = np.array(image.getdata())
        np_image = np_image.reshape(image.size[0], image.size[1], 4)
        print(np_image.shape)
        # np_image = imageio.imread(image_location)

        masks = []
        for i in mask_locations:
            mask_image = Image.open(i)
            np_mask = np.array(mask_image.getdata())
            np_mask = np_mask.reshape(image.size[0], image.size[1])
            masks.append(np_mask)
            # masks.append(imageio.imread(i))

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

    for i in range(x - square_size//2, 1 + x + square_size//2):
        for j in range(y - square_size // 2, 1 + y + square_size // 2):
            if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
                x_list.append(np.full([4,], np.nan))
                for g in image_gradient:
                    x_list.append(np.full([4, ], np.nan))
            else:
                x_list.append(image[i][j])
                for g in image_gradient:
                    x_list.append(g[i][j])

    x_list = np.hstack(x_list + [general_image_stats])

    y = 0
    for i in masks:
        if i[x][y] > 0:
            y = 1

    return {'output':y, 'input':x_list}



def create_image_features(image, masks, square_size, max_feature_count=20000):
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
    if len(inputs) > max_feature_count:
        inputs = random.sample(inputs, max_feature_count)

    for i in inputs:
        result_dicts.append(get_features_of_point(i[0], i[1], image, image_gradient, masks, square_size, general_image_stats))

    return pd.DataFrame.from_dict(result_dicts)




def train_nn_classifier(x_train, x_test, y_train, y_test, max_iter = 1000,
                        nn_shape = (1000, 1000,), activation = 'tanh', name = None, retrain = True):
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



def train_rf_classifier(x_train, x_test, y_train, y_test, n_estimators = 500,
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
    # with open(name + '.plk', 'wb') as model_file:
    #     pickle.dump(clf, model_file)
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
    clf = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
    clf.fit(x_train, y_train)
    # with open(name + '.plk', 'wb') as model_file:
    #     pickle.dump(clf, model_file)
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
    # with open(name + '.plk', 'wb') as model_file:
    #     pickle.dump(clf, model_file)
    print('trained:', name, clf.score(x_test, y_test))

    test_res = clf.predict_proba(x_train)
    train_res = clf.predict_proba(x_test)
    return {'test_predictions':test_res,
            'train_predictions':train_res,
            'clf':clf}


def get_model_inputs():

    gen = generate_input_image_and_masks()

    dfs = []

    while True:
        try:
            image, masks = next(gen)
            dfs.append(create_image_features(image, masks, 10))
        except OSError:
            traceback.print_exc()
        except:
            traceback.print_exc()
            break
    df = pd.concat(dfs, ignore_index=True)
    try:
        pass
        #df.to_pickle(files_loc + 'temp_input.plk')
    except:
        traceback.print_exc()

    positive_matches = df[df['output'] > 0]
    negative_matches = df[df['output'] == 0]
    negative_matches = negative_matches.sample(n=positive_matches.shape[0])
    df = pd.concat([positive_matches, negative_matches], ignore_index=True)
    df = df.sample(frac=1)

    x, y = [], []
    for _, i in df.iterrows():
        x.append(i['input'])
        y.append(i['output'])

    x = np.vstack(x)
    y = np.vstack(y)

    x = np.nan_to_num(x)
    y = np.ravel(y)

    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
    min_max_preprocessor = MinMaxScaler([-1,1])
    x_train = min_max_preprocessor.fit_transform(x_train)
    x_test = min_max_preprocessor.transform(x_test)

    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test):
    train_rf_classifier(x_train, x_test, y_train, y_test, name = 'RF1')
    train_adaboost_classifier(x_train, x_test, y_train, y_test, name='ADA1')
    #train_nn_classifier(x_train, x_test, y_train, y_test, name='NN1')
    train_et_classifier(x_train, x_test, y_train, y_test, name='ET1')
    train_gb_classifier(x_train, x_test, y_train, y_test, name='GB1')



def main():
    x_train, x_test, y_train, y_test = get_model_inputs()
    train_models(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()



