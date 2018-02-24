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

full_image_read_size = (128, 128)
min_nuclei_size = 10


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
def get_cnn():
    inputs = Input((128, 128, 2))
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

            masks = []
            for i in mask_locations:
                mask_image = Image.open(i)
                np_mask = np.array(mask_image.getdata())
                np_mask = np_mask.reshape(start_image.size[1], start_image.size[0])
                masks.append(np_mask)
        except OSError:
            continue

        yield np_image, masks


def get_subimages(input_image, gradient, input_mask, max_subimages, transpose = False, rotation = 0):
    if transpose:
        input_image = np.transpose(input_image)
        input_mask = np.transpose(input_mask)
        gradient = np.transpose(gradient)
    input_image = np.rot90(input_image, rotation)
    input_mask = np.rot90(input_mask, rotation)
    input_gradient = np.rot90(gradient, rotation)

    output = []
    for i in range(max_subimages):
        for j in range(max_subimages):
            x1 = i*full_image_read_size[0]
            x2 = (1+i)*full_image_read_size[0]
            y1 = j*full_image_read_size[0]
            y2 = (1+j)*full_image_read_size[0]


            output.append({'input':np.dstack((np.expand_dims(np.transpose(input_image[x1:x2,y1:y2]), axis=2),
                                             np.expand_dims(np.transpose(input_gradient[x1:x2,y1:y2]), axis=2))),
                           'output':np.expand_dims(np.transpose(input_mask[x1:x2,y1:y2]), axis=2)})
    return output


def get_image_arrays_for_full_location_training(input_image, masks):
    max_subimages = (min(input_image.shape))//min(full_image_read_size)
    for i in range(max_subimages):
        pass

    gradient = gaussian_gradient_magnitude(input_image, sigma=.4)

    mask_sum = functools.reduce(operator.add, masks)
    vectorized = np.vectorize(lambda t: 1 if t>0 else 0)
    mask_sum = vectorized(mask_sum)

    output = []
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=0))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=1))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=2))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=3))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=0))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=1))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=2))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=3))

    return pd.DataFrame(output)


def get_image_arrays_for_full_edge_training(input_image, masks):
    max_subimages = (min(input_image.shape))//min(full_image_read_size)

    gradient = gaussian_gradient_magnitude(input_image, sigma=.4)

    masks = [gaussian_gradient_magnitude(i, sigma=.4) for i in masks]
    mask_sum = functools.reduce(operator.add, masks)
    vectorized = np.vectorize(lambda t: 1 if t>0 else 0)
    mask_sum = vectorized(mask_sum)

    output = []
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=0))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=1))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=2))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=False, rotation=3))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=0))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=1))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=2))
    output.extend(get_subimages(input_image, gradient, mask_sum, max_subimages, transpose=True, rotation=3))

    return pd.DataFrame(output)


def get_dataframes_for_training_location():
    gen = generate_input_image_and_masks()
    location_dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            images, masks = next(gen)
            location_dfs.append(get_image_arrays_for_full_location_training(images, masks))
        except StopIteration:
            traceback.print_exc()
            break


    print('images read')
    location_df = pd.concat(location_dfs, ignore_index=True)
    location_df = location_df.sample(frac=1)

    return location_df


def get_dataframes_for_training_edge():
    gen = generate_input_image_and_masks()
    edge_dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            images, masks = next(gen)
            edge_dfs.append(get_image_arrays_for_full_edge_training(images, masks))
        except StopIteration:
            traceback.print_exc()
            break


    print('images read')
    edge_df = pd.concat(edge_dfs, ignore_index=True)
    edge_df = edge_df.sample(frac=1)

    return edge_df


def get_model_inputs(df, x_labels, test_size = 0.1):
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


def get_models():
    try:
        loc_model = load_model(files_loc + 'cnn_full_loc.h5', custom_objects={'mean_iou': mean_iou})
        edge_model = load_model(files_loc + 'cnn_full_edge.h5', custom_objects={'mean_iou': mean_iou})
    except:
        traceback.print_exc()
        df_loc = get_dataframes_for_training_location()

        x_train, x_test, y_train, y_test = get_model_inputs(df_loc, x_labels=['input'])
        loc_model = get_cnn()
        loc_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
        loc_model.save(files_loc + 'cnn_full_loc.h5')
        del df_loc

        df_edge = get_dataframes_for_training_edge()
        x_train, x_test, y_train, y_test = get_model_inputs(df_edge, x_labels=['input'])
        edge_model = get_cnn()
        edge_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
        edge_model.save(files_loc + 'cnn_full_edge.h5')
    return loc_model, edge_model



def combine_predictions():
    pass


def predict_subimages(input_image, gradient, input_mask, max_subimages, transpose, rotation, loc_model, edge_model):
    if transpose:
        input_image = np.transpose(input_image)
        input_mask = np.transpose(input_mask)
        gradient = np.transpose(gradient)
    input_image = np.rot90(input_image, rotation)
    input_mask = np.rot90(input_mask, rotation)
    input_gradient = np.rot90(gradient, rotation)




def predict_image(loc_model, edge_model, np_image):
    image_gradient = gaussian_gradient_magnitude(np_image, sigma=.4)




def main():
    loc_model, edge_model = get_models()


if __name__ == '__main__':
    main()