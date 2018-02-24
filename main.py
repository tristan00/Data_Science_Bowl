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
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras import losses
import scipy.misc
from keras import optimizers
import shutil
import multiprocessing
import functools
import operator
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
import cv2
import tensorflow as tf
import pickle
from keras.layers.core import Dropout, Lambda
from keras.models import Model, load_model
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from skimage.transform import resize

max_images = 1000
sample_per_image_location_model= 5000
sample_per_image_edge_model= 60000
files_loc = 'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/'

full_image_read_size = (256, 256)
k_means_image_size = (128, 128)
min_nuclei_size = 10


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

            resized_np_image = resize(np_image, full_image_read_size)

            # resized_np_image = np.array(start_image.getdata())[:,0]
            # resized_np_image = resized_np_image.reshape(resized_image.size[0], resized_image.size[1])

            np_gradient_image = gaussian_gradient_magnitude(np_image, sigma=.4)
            # np_image = imageio.imread(image_location)

            masks = []
            resized_masks = []
            for i in mask_locations:
                mask_image = Image.open(i)
                np_mask = np.array(mask_image.getdata())
                np_mask = np_mask.reshape(start_image.size[1], start_image.size[0])
                masks.append(np_mask)
                resized_np_mask = resize(np_mask, full_image_read_size)
                resized_masks.append(resized_np_mask)
        except OSError:
            continue

        yield np_image, np_gradient_image, masks, resized_np_image, resized_masks


def get_cnn():
    model = Sequential()
    #model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1)))
    model.add(ZeroPadding2D((1,1),input_shape=(64, 64, 2)))
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

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.001, decay=1e-8, momentum=0.0, nesterov=False)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
    return model




def get_image_array_for_location_detection(image, np_gradient_image, size, x, y, masks):
    output_dict = dict()
    sub_image = image[int(x):int(x+size), int(y):int(y+size)]
    sub_image = np.expand_dims(sub_image, axis = 2)
    sub_image_gradient = np.expand_dims(np_gradient_image[int(x):int(x+size), int(y):int(y+size)], axis = 2)
    output_dict['image'] = np.dstack([sub_image, sub_image_gradient])

    output = 0
    for i in masks:
        if i[x,y] > 0:
            output = 1

    output_dict['output'] = output
    if output_dict['image'].shape != (size, size, 2):
        print(x,y,output_dict['image'].shape)
        return None
    else:
        return output_dict


def get_image_arrays_for_location_detection_training(input_image, size, masks):
    inputs = []
    result_dicts = []

    adj_image = np.pad(input_image, size//2, mode='constant')
    np_gradient_image = gaussian_gradient_magnitude(adj_image, sigma=.4)

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            inputs.append((i,j))
    if len(inputs) > sample_per_image_location_model:
        inputs = random.sample(inputs, sample_per_image_location_model)
    for i in inputs:
        result_dicts.append(
            get_image_array_for_location_detection(adj_image,  np_gradient_image, size, i[0], i[1], masks))
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


def get_dataframes_for_location_training(square_size):
    gen = generate_input_image_and_masks()
    dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            image, image_gm, masks, _ = next(gen)
            #dfs.append(create_image_features(image, image_gm, masks, square_size))
            dfs.append(get_image_arrays_for_location_detection_training(image, square_size, masks))
        except StopIteration:
            traceback.print_exc()
            break

    print('images read')
    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1)
    return df


def get_location_detection_model():
    try:
        model = load_model(files_loc + 'cnn_location2.h5')
        print(model)
    except:
        traceback.print_exc()
        df = get_dataframes_for_location_training(64)
        x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image'])
        #train_models(x_train, x_test, y_train, y_test)

        model = get_cnn()

        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2)
        model.save(files_loc + 'cnn_location2.h5')
    return model


def get_image_array_for_edge_detection(image, np_gradient_image, size, x, y, masks):
    output_dict = dict()
    sub_image = image[int(x):int(x+size), int(y):int(y+size)]
    sub_image = np.expand_dims(sub_image, axis = 2)
    sub_image_gradient = np.expand_dims(np_gradient_image[int(x):int(x+size), int(y):int(y+size)], axis = 2)
    output_dict['image'] = np.dstack([sub_image, sub_image_gradient])

    output = 0
    for i in masks:
        if i[x,y] == 0:
            temp_square = i[x-1:x+2, y-1: y+2]
            if min(temp_square.shape) == 3 and (temp_square[0,1] > 0 or temp_square[1,0] > 0 or temp_square[2,1] > 0 or temp_square[1,2] > 0):
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
    if len(inputs) > sample_per_image_edge_model:
        inputs = random.sample(inputs, sample_per_image_edge_model)
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
        traceback.print_exc()
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


def get_edge_detection_model(image, np_gradient_image, size, x, y, masks):
    try:
        model = load_model(files_loc + 'cnn_edge.h5')
        print(model)
    except:
        traceback.print_exc()
        df = get_dataframes_for_edge_detection_training(32)
        x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image'])
        #train_models(x_train, x_test, y_train, y_test)

        model = get_cnn()

        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2)
        model.save(files_loc + 'cnn_edge.h5')
    return model


def get_image_arrays_for_full_location_training(input_image, masks):
    mask_sum = functools.reduce(operator.add, masks)
    vectorized = np.vectorize(lambda t: 1 if t>0 else 0)
    mask_sum = vectorized(mask_sum)
    output_dict = []

    output_dict.append({'image':np.expand_dims(input_image,axis=2), 'output':np.expand_dims(mask_sum,axis=2)})
    output_dict.append({'image': np.expand_dims(np.transpose(input_image),axis=2), 'output': np.expand_dims(np.transpose(mask_sum),axis=2)})
    output_dict.append({'image': np.expand_dims(np.rot90(input_image, 1),axis=2), 'output': np.expand_dims(np.rot90(mask_sum, 1),axis=2)})
    output_dict.append({'image': np.expand_dims(np.rot90(input_image, 2),axis=2), 'output': np.expand_dims(np.rot90(mask_sum, 2),axis=2)})
    output_dict.append({'image': np.expand_dims(np.rot90(input_image, 3),axis=2), 'output': np.expand_dims(np.rot90(mask_sum, 3),axis=2)})
    output_dict.append({'image': np.expand_dims(np.rot90(np.transpose(input_image), 1),axis=2), 'output': np.expand_dims(np.rot90(np.transpose(mask_sum), 1),axis=2)})
    output_dict.append({'image': np.expand_dims(np.rot90(np.transpose(input_image), 2),axis=2), 'output': np.expand_dims(np.rot90(np.transpose(mask_sum), 2),axis=2)})
    output_dict.append({'image': np.expand_dims(np.rot90(np.transpose(input_image), 3),axis=2), 'output': np.expand_dims(np.rot90(np.transpose(mask_sum), 3),axis=2)})



    df = pd.DataFrame.from_dict(output_dict)
    return df


def get_dataframes_for_full_location_training():
    gen = generate_input_image_and_masks()
    dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            _, image_gm, _, images, masks = next(gen)
            #dfs.append(create_image_features(image, image_gm, masks, square_size))
            dfs.append(get_image_arrays_for_full_location_training(images, masks))
        except StopIteration:
            traceback.print_exc()
            break

    print('images read')
    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1)
    return df


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


#taken from https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
def get_cnn2():
    inputs = Input((256, 256, 1))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    return model

def full_location_training_model():
    try:
        model = load_model(files_loc + 'cnn_full_loc.h5')
        print(model)
    except:
        traceback.print_exc()
        df = get_dataframes_for_full_location_training()
        x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image'])
        #train_models(x_train, x_test, y_train, y_test)

        model = get_cnn2()

        # y_train = keras.utils.to_categorical(y_train, 2)
        # y_test = keras.utils.to_categorical(y_test, 2)


        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25)
        model.save(files_loc + 'cnn_full_loc.h5')
    return model


def get_model_inputs(df, x_labels=['image'], test_size = 0.1):
    print('testing inputs: {0}'.format(x_labels))
    x, y = [], []

    #TODO: vectorize
    for _, i in df.iterrows():
        x.append(np.hstack([i[x_label] for x_label in x_labels]))
        #x.append(np.hstack([i['image'], i['general_image_stats']]))
        y.append(i['output'])

    x = np.array(x)
    y = np.array(y)

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


def get_outputs_from_flat_array(a):
    on_label = False

    image_list = []
    temp_image_list = []
    for count, i in enumerate(a):
        if i != 0 and not on_label:
            temp_image_list = []
            temp_image_list.append(count)
            on_label = True
        elif i != 0 and on_label:
            temp_image_list.append(count)
        elif i == 0 and on_label:
            on_label = False
            image_list.append(temp_image_list)

    res_str = ''
    for i in image_list:
        res_str += str(int(i[0]))
        res_str += ' '
        res_str += str(len(i))
        res_str += ' '
    res_str = res_str[:-1]



    return res_str



def to_output_format(label_dict, np_image, image_name):
    output_dicts = []
    for count, i in enumerate(label_dict):
        image_copy = np_image.copy()
        image_copy.fill(0)
        for j in i:
            image_copy[j[1], j[0]] = 1
        image_copy = np.transpose(image_copy)
        flat_image = image_copy.flatten()

        output_dict = dict()
        output_dict['ImageId'] = image_name
        output_dict['EncodedPixels'] = get_outputs_from_flat_array(flat_image)
        output_dicts.append(output_dict)
        print('output created:', count, image_name)
    return output_dicts



def get_nuclei_from_predictions(locations, image_id):
    location_set = set(locations)
    prediction_n_locations = location_set
    nuclei_predictions = []

    while len(prediction_n_locations) > 0:
        starting_location = prediction_n_locations.pop()
        prediction_n_locations.add(starting_location)

        temp_neucli_locations = [starting_location]



        while True:
            location_added = False

            for n_loc in temp_neucli_locations:
                if (n_loc[0] + 1, n_loc[1]) in prediction_n_locations and (n_loc[0] + 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.append((n_loc[0] + 1, n_loc[1]))
                    location_added = True
                    break
                if (n_loc[0] - 1, n_loc[1]) in prediction_n_locations and (n_loc[0] - 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.append((n_loc[0] - 1, n_loc[1]))
                    location_added = True
                    break
                if (n_loc[0], n_loc[1] + 1) in prediction_n_locations and (n_loc[0], n_loc[1] + 1) not in temp_neucli_locations:
                    temp_neucli_locations.append((n_loc[0], n_loc[1] + 1))
                    location_added = True
                    break
                if (n_loc[0], n_loc[1] - 1) in prediction_n_locations and (n_loc[0], n_loc[1] - 1) not in temp_neucli_locations:
                    temp_neucli_locations.append((n_loc[0], n_loc[1] - 1))
                    location_added = True
                    break
            if not location_added:
                break
        for i in temp_neucli_locations:
            prediction_n_locations.remove(i)
        nuclei_predictions.append(temp_neucli_locations)
        print('clusters found:', len(nuclei_predictions), ' pixels_left:', len(prediction_n_locations),image_id)
    return nuclei_predictions


def get_outputs(input_dict):
    locations, np_image, image_id = input_dict['clusters'], input_dict['np_image'], input_dict['image_id']
    prediction_image_to_location_list
    clusters = get_nuclei_from_predictions(locations, image_id)
    return to_output_format(clusters, np_image, image_id)


def prediction_image_to_location_list(prediction_image):
    output = []
    for i in prediction_image.shape[0]:
        for j in prediction_image[1]:
            if prediction_image[i,j] > 0:
                output.append((i,j))
    return output


def get_predictions(location_model, edge_model, sub_image_size):
    folders = glob.glob(files_loc + 'stage1_test/*/')
    random.shuffle(folders)
    output_dicts = []

    predicted_pixel_list = []

    for folder in folders:
        try:
            image_location = glob.glob(folder + 'images/*')[0]
            print(image_location)
            #mask_locations = glob.glob(folder + 'masks/*')

            image_id = os.path.basename(image_location).split('.')[0]

            pil_image = Image.open(image_location).convert('LA')
            np_image = np.array(pil_image.getdata())[:, 0]
            np_image = np_image.reshape(pil_image.size[1], pil_image.size[0])
            adj_image = np.pad(np_image, sub_image_size // 2, mode='constant')
            np_gradient_image = gaussian_gradient_magnitude(adj_image, sigma=.4)

            sub_image_array = []

            for i in range(np_image.shape[0]):
                for j in range(np_image.shape[1]):
                    output_dict = dict()
                    sub_image = adj_image[int(i):int(i + sub_image_size), int(j):int(j + sub_image_size)]
                    sub_image = np.expand_dims(sub_image, axis=2)
                    sub_image_gradient = np.expand_dims(np_gradient_image[int(i):int(i + sub_image_size), int(j):int(j + sub_image_size)],
                                                        axis=2)
                    sub_image = np.dstack([sub_image, sub_image_gradient])
                    sub_image_array.append(sub_image)
                    #sub_image = np.expand_dims(sub_image, axis=0)

            sub_image_array = np.array(sub_image_array)
            print(sub_image_array.shape)


            location_predictions = location_model.predict(sub_image_array)
            #edge_predictions = edge_model.predict(sub_image_array)

            confidence_tresholds = [.7]
            cluster_dataset = []
            for c in confidence_tresholds:
                location_image, edge_image = np_image.copy(), np_image.copy()
                for count, location_prediction in enumerate(location_predictions):
                    x = count%np_image.shape[1]
                    y = count//np_image.shape[1]

                    if location_prediction[1] > c:
                        location_image[y,x] = 255
                        cluster_dataset.append((x, y))
                    else:
                        location_image[y, x] = 0

                    # if edge_prediction[1] > c:
                    #     edge_image[y,x] = 255
                    #     cluster_dataset.append(np.array([x, y]))
                    # else:
                    #     edge_image[y, x] = 0

                # scipy.misc.imsave('location_image_{0}.jpg'.format(str(int(c * 100))), location_image)
                # scipy.misc.imsave('starting_file_{0}.jpg'.format(str(int(c*100))), np_image)
                # with open('temp_cluster.plk', 'wb') as infile:
                #     pickle.dump(cluster_dataset, infile)
                # with open('image_id.plk', 'wb') as infile:
                #     pickle.dump(image_id, infile)
                # with open('np_image.plk', 'wb') as infile:
                #     pickle.dump(np_image, infile)

                # print('predicting clusters')

                predicted_pixel_list.append({'clusters':cluster_dataset, 'np_image':np_image, 'image_id':image_id})
                # clusters = get_nuclei_from_predictions(cluster_dataset)
                # output_dicts.extend()


        except:
            traceback.print_exc()
    try:
        with open('locations.plk', 'wb') as infile:
            pickle.dump(predicted_pixel_list, infile)
    except:
        traceback.print_exc()
    p = multiprocessing.Pool(processes=15)
    output_dicts =functools.reduce(operator.concat, p.map_async(get_outputs, predicted_pixel_list).get())

    df = pd.DataFrame.from_dict(output_dicts)
    df.to_csv('output.csv', index = False)


def main():
    #location_model = get_location_detection_model()
    #edge_model = get_edge_detection_model()
    full_location_training_model()

    #get_predictions(location_model, None, 64)


if __name__ == '__main__':
    main()


    # with open('temp_cluster.plk', 'rb') as infile:
    #     cluster_dataset = pickle.load(infile)
    # with open('image_id.plk', 'rb') as infile:
    #     image_id = pickle.load(infile)
    # with open('np_image.plk', 'rb') as infile:
    #     np_image = pickle.load(infile)
    # clusters = get_nuclei_from_predictions(cluster_dataset)

    # output_dicts = to_output_format(clusters, np_image, image_id)
    # df = pd.DataFrame.from_dict(output_dicts)
    # df.to_csv('output.csv', index=False)

