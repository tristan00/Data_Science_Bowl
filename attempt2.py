import pandas as pd
import glob
from PIL import Image
import numpy as np
import random
import functools
import operator
import traceback
from scipy.ndimage import gaussian_gradient_magnitude
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
import os
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D

confidence_threshold = .5
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


            output.append({'input':np.dstack((np.expand_dims(input_image[x1:x2,y1:y2], axis=2),
                                             np.expand_dims(input_gradient[x1:x2,y1:y2], axis=2))),
                           'output':np.expand_dims(input_mask[x1:x2,y1:y2], axis=2)})
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

    # masks = [gaussian_gradient_magnitude(i, sigma=.4) for i in masks]
    # mask_sum = functools.reduce(operator.add, masks)

    mask_sum = input_image.copy()
    mask_sum[:] = 0

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            for m in masks:
                if m[i, j] == 0:
                    small_mask = m[i - 1:i + 2, j - 1: j + 2]
                    if min(small_mask.shape) > 0 and np.max(small_mask) > 0:
                        mask_sum[i,j] = 1



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


def get_model_inputs(df, x_labels, test_size = 0.05):
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


def get_loc_model():
    try:
        loc_model = load_model(files_loc + 'cnn_full_loc.h5', custom_objects={'mean_iou': mean_iou})
    except:
        traceback.print_exc()
        df_loc = get_dataframes_for_training_location()

        x_train, x_test, y_train, y_test = get_model_inputs(df_loc, x_labels=['input'])
        loc_model = get_cnn()
        loc_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
        loc_model.save(files_loc + 'cnn_full_loc.h5')
    return loc_model


def get_edge_model():
    return None
    try:
        edge_model = load_model(files_loc + 'cnn_full_edge.h5', custom_objects={'mean_iou': mean_iou})
    except:
        traceback.print_exc()

        df_edge = get_dataframes_for_training_edge()
        x_train, x_test, y_train, y_test = get_model_inputs(df_edge, x_labels=['input'])
        edge_model = get_cnn()
        edge_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2)
        edge_model.save(files_loc + 'cnn_full_edge.h5')
    return edge_model


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
            image_copy[j[0], j[1]] = 1
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

        temp_neucli_locations = set([starting_location])
        failed_locations = set()

        while True:
            location_added = False

            search_set = temp_neucli_locations - failed_locations

            for n_loc in search_set:
                if (n_loc[0] + 1, n_loc[1]) in prediction_n_locations and (n_loc[0] + 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0] + 1, n_loc[1]))
                    location_added = True
                if (n_loc[0] - 1, n_loc[1]) in prediction_n_locations and (n_loc[0] - 1, n_loc[1]) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0] - 1, n_loc[1]))
                    location_added = True
                if (n_loc[0], n_loc[1] + 1) in prediction_n_locations and (n_loc[0], n_loc[1] + 1) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0], n_loc[1] + 1))
                    location_added = True
                if (n_loc[0], n_loc[1] - 1) in prediction_n_locations and (n_loc[0], n_loc[1] - 1) not in temp_neucli_locations:
                    temp_neucli_locations.add((n_loc[0], n_loc[1] - 1))
                    location_added = True
                failed_locations.add(n_loc)
            if not location_added:
                break
        for i in temp_neucli_locations:
            prediction_n_locations.remove(i)
        nuclei_predictions.append(temp_neucli_locations)
        print('clusters found:', len(nuclei_predictions), ' pixels_left:', len(prediction_n_locations),image_id)
    return nuclei_predictions


def prediction_image_to_location_list(prediction_image):
    output = []
    for i in range(prediction_image.shape[0]):
        for j in range(prediction_image.shape[1]):
            if prediction_image[i,j] > 0:
                output.append((i,j))
    return output


def get_outputs(input_dict):
    output, np_image, image_id = input_dict['output'], input_dict['np_image'], input_dict['image_id']
    locations = prediction_image_to_location_list(output)
    clusters = get_nuclei_from_predictions(locations, image_id)
    return to_output_format(clusters, np_image, image_id)


def predict_subimages(input_image, gradient, max_subimages, transpose, rotation, loc_model, edge_model):
    if transpose:
        input_image = np.transpose(input_image)
        gradient = np.transpose(gradient)
    input_image = np.rot90(input_image, rotation)
    input_gradient = np.rot90(gradient, rotation)

    output = []
    for i in range(max_subimages):
        predictions = []
        for j in range(max_subimages):
            x1 = i * full_image_read_size[0]
            x2 = (1 + i) * full_image_read_size[0]
            y1 = j * full_image_read_size[0]
            y2 = (1 + j) * full_image_read_size[0]
            temp_image = input_image[x1:x2,y1:y2]
            temp_gradient = input_gradient[x1:x2,y1:y2]
            model_input = np.dstack([np.expand_dims(temp_image, axis=2), np.expand_dims(temp_gradient, axis=2)])
            model_input = np.expand_dims(model_input, axis = 0)
            prediction = loc_model.predict(model_input)
            prediction = np.squeeze(prediction)
            predictions.append(prediction)
        output.append(np.concatenate(predictions, 1))
    output = np.concatenate(output, 0)
    pad = np.zeros(input_image.shape)
    pad[:] = np.nan
    pad[:output.shape[0],:output.shape[1]] = output
    output = pad

    rotated_output = np.rot90(output, 4-rotation)

    return rotated_output


def predict_image(loc_model, edge_model, np_image, image_id):
    image_gradient = gaussian_gradient_magnitude(np_image, sigma=.4)
    max_subimages = (min(np_image.shape)) // min(full_image_read_size)

    results = []
    results.append(predict_subimages(np_image, image_gradient, max_subimages, False, 0, loc_model, edge_model))
    results.append(predict_subimages(np_image, image_gradient, max_subimages, False, 1, loc_model, edge_model))
    results.append(predict_subimages(np_image, image_gradient, max_subimages, False, 2, loc_model, edge_model))
    results.append(predict_subimages(np_image, image_gradient, max_subimages, False, 3, loc_model, edge_model))
    result_array = np.dstack(results)

    mean = np.nanmean(result_array, 2)
    print(mean.shape)

    mean_bool = mean > confidence_threshold
    prediction_locations = mean_bool.astype(int)
    input_dict = {'output':prediction_locations, 'image_id':image_id, 'np_image':np_image}

    output_dicts = []
    output_dicts.extend(get_outputs(input_dict))
    return output_dicts




def run_predictions(loc_model, edge_model):
    folders = glob.glob(files_loc + 'stage1_train/*/')
    random.shuffle(folders)

    output_dicts = []

    for folder in folders:
        image_location = glob.glob(folder + 'images/*')[0]
        start_image = Image.open(image_location).convert('LA')
        image_id = os.path.basename(image_location).split('.')[0]
        np_image = np.array(start_image.getdata())[:, 0]
        np_image = np_image.reshape(start_image.size[1], start_image.size[0])
        output_dicts.extend(predict_image(loc_model, edge_model, np_image, image_id))

    df = pd.DataFrame.from_dict(output_dicts)
    df.to_csv('output.csv', index = False)



def main():
    edge_model = get_edge_model()
    loc_model = get_loc_model()
    run_predictions(loc_model, edge_model)


if __name__ == '__main__':
    main()