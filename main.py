import pandas as pd
import glob
from PIL import Image
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, LeakyReLU, Convolution2D
from keras.models import load_model
import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, SpectralClustering
import random
import scipy.misc
from keras import optimizers
import shutil
import os
import traceback
from scipy.ndimage import gaussian_gradient_magnitude
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from scipy import misc
import keras
from keras.models import Sequential
from keras.layers import Dense
import h5py
import pickle

max_images = 1000
sample_per_image = 50000
files_loc = 'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/'

k_means_image_size = (128, 128)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_input_image_and_masks():
    folders = glob.glob(files_loc + 'stage1_train/*/')
    random.shuffle(folders)

    for folder in folders:
        try:
            image_location = glob.glob(folder + 'images/*')[0]
            mask_locations = glob.glob(folder + 'masks/*')
            start_image = Image.open(image_location).convert('LA')
            np_image = np.array(start_image.getdata())[:,0]
            np_image = np_image.reshape(start_image.size[1], start_image.size[0])

            resized_np_image = scipy.misc.imresize(np_image, k_means_image_size)
            # resized_np_image = np.array(start_image.getdata())[:,0]
            # resized_np_image = resized_np_image.reshape(resized_image.size[0], resized_image.size[1])

            np_gradient_image = gaussian_gradient_magnitude(np_image, sigma=.4)
            # np_image = imageio.imread(image_location)

            masks = []
            for i in mask_locations:
                mask_image = Image.open(i)
                np_mask = np.array(mask_image.getdata())
                np_mask = np_mask.reshape(start_image.size[1], start_image.size[0])
                masks.append(np_mask)
                # masks.append(imageio.imread(i))
        except OSError:
            continue

        yield np_image, np_gradient_image, masks, resized_np_image


def get_cnn():
    model = Sequential()
    #model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1)))
    model.add(ZeroPadding2D((1,1),input_shape=(32, 32, 2)))
    model.add(Conv2D(16, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(16, (3, 3), activation='elu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='elu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.0, nesterov=False)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
    return model


def get_image_array_for_edge_detection(image, np_gradient_image, size, x, y, masks):
    output_dict = dict()
    sub_image = image[int(x):int(x+size), int(y):int(y+size)]
    sub_image = np.expand_dims(sub_image, axis = 2)
    sub_image_gradient = np.expand_dims(np_gradient_image[int(x):int(x+size), int(y):int(y+size)], axis = 2)

    output_dict['image'] = np.dstack([sub_image, sub_image_gradient])
    output = 0
    for i in masks:
        if i[x,y] > 0:
            temp_square = i[x-1:x+2, y-1: y+2]
            if min(temp_square.shape) > 0 and np.mean(temp_square) < np.max(temp_square):
                output = 1
            elif min(temp_square.shape) == 0:
                output = 1

    output_dict['output'] = output
    if output_dict['image'].shape != (size, size, 2):
        print(x,y,output_dict['image'].shape)
        return None
    else:
        return output_dict


def get_image_arrays_for_edge_detection_training(input_image, size, masks):
    inputs = []
    result_dicts = []

    adj_image = np.pad(input_image, size//2, mode='constant')
    np_gradient_image = gaussian_gradient_magnitude(adj_image, sigma=.4)

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            inputs.append((i,j))
    if len(inputs) > sample_per_image:
        inputs = random.sample(inputs, sample_per_image)
    for i in inputs:
        result_dicts.append(
            get_image_array_for_edge_detection(adj_image,  np_gradient_image, size, i[0], i[1], masks))
    result_dicts = [i for i in result_dicts if i]

    df = pd.DataFrame.from_dict(result_dicts)

    try:
        positive_matches = df[df['output'] > 0]
        negative_matches = df[df['output'] == 0]
        negative_matches = negative_matches.sample(n=positive_matches.shape[0])
        df = pd.concat([positive_matches, negative_matches], ignore_index=True)
    except:
        #TODO: handle case with more positives than negatives
        pass
    df = df.sample(frac=1)

    return df


def get_dataframes_for_edge_detection_training(square_size):
    gen = generate_input_image_and_masks()
    dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            image, image_gm, masks, _ = next(gen)
            #dfs.append(create_image_features(image, image_gm, masks, square_size))
            dfs.append(get_image_arrays_for_edge_detection_training(image, square_size, masks))
        except StopIteration:
            traceback.print_exc()
            break

    print('images read')
    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1)
    return df


def get_model_inputs(df, x_labels=['image'], test_size = 0.01):
    print('testing inputs: {0}'.format(x_labels))
    x, y = [], []

    #TODO: vectorize
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

def get_edge_detection_model():
    try:
        model = load_model(files_loc + 'cnn3.h5')
        print(model)
    except:
        traceback.print_exc()
        df = get_dataframes_for_edge_detection_training(32)
        x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image'])
        #train_models(x_train, x_test, y_train, y_test)

        model = get_cnn()

        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
        model.save(files_loc + 'cnn3.h5')
    return model

def extract_solutions_from_edges():
    pass


def get_predictions(model1, sub_image_size):
    folders = glob.glob(files_loc + 'stage1_test/*/')
    random.shuffle(folders)
    output_dicts = []

    for folder in folders:
        try:
            image_location = glob.glob(folder + 'images/*')[0]
            print(image_location)
            #mask_locations = glob.glob(folder + 'masks/*')

            image_id = os.path.basename(image_location).split('.')[0]

            pil_image = Image.open(image_location).convert('LA')
            np_image = np.array(pil_image.getdata())[:, 0]
            np_image = np_image.reshape(pil_image.size[1], pil_image.size[0])
            #np_image = np_image[0:100, 0:150]
            adj_image = np.pad(np_image, sub_image_size // 2, mode='constant')

            sub_image_array = []

            for i in range(np_image.shape[0]):
                for j in range(np_image.shape[1]):
                    sub_image = adj_image[int(i):int(i + sub_image_size), int(j):int(j + sub_image_size)]
                    sub_image_array.append(np.expand_dims(sub_image, axis=2))
                    #sub_image = np.expand_dims(sub_image, axis=0)

            sub_image_array = np.array(sub_image_array)
            print(sub_image_array.shape)

            predictions = model1.predict(sub_image_array)

            confidence_tresholds = [.5]
            cluster_dataset = []
            for c in confidence_tresholds:
                preprocessed_image = np_image.copy()
                for count, prediction in enumerate(predictions):
                    x = count%np_image.shape[1]
                    y = count//np_image.shape[1]
                    if prediction[1] > c:
                        preprocessed_image[y,x] = 255
                        cluster_dataset.append(np.array([x, y]))
                    else:
                        preprocessed_image[y, x] = 0
                scipy.misc.imsave('preprocessed_file_{0}.jpg'.format(str(int(c*100))), preprocessed_image)
                scipy.misc.imsave('starting_file_{0}.jpg'.format(str(int(c*100))), np_image)



        except:
            traceback.print_exc()

    df = pd.DataFrame.from_dict(output_dicts)
    df.to_csv(files_loc + 'output.csv', index = False)


def main():
    model = get_edge_detection_model()


if __name__ == '__main__':
    main()
