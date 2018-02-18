import pandas as pd
import glob
from PIL import Image
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, LeakyReLU
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


max_images = 1000
sample_per_image = 12500
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
    output_dict['image'] = image[int(x):int(x+size), int(y):int(y+size)]
    output_dict['image'] = np.expand_dims(output_dict['image'], axis = 2)
    output = 0
    for i in masks:
        if i[x][y] > 0:
            output = 1
    output_dict['output'] = output
    if output_dict['image'].shape != (size, size, 1):
        print(x,y,output_dict['image'].shape)
        return None
    else:
        return output_dict


def get_image_arrays(image, size, masks):
    inputs = []
    result_dicts = []

    adj_image = np.pad(image, size//2, mode='constant')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            inputs.append((i,j))
    if len(inputs) > sample_per_image:
        inputs = random.sample(inputs, sample_per_image)
    for i in inputs:
        result_dicts.append(
            get_image_array(adj_image, size, i[0], i[1], masks))
    result_dicts = [i for i in result_dicts if i]
    return pd.DataFrame.from_dict(result_dicts)


def preprocess_images_for_kmeans(input_image, masks):
    resized_image = np.resize(input_image, (256, 256))
    transposed_image = np.transpose(resized_image)
    resized_image2 = np.rot90(resized_image, 1)
    resized_image3 = np.rot90(resized_image, 2)
    resized_image4 = np.rot90(resized_image, 3)
    transposed_image2 = np.rot90(resized_image, 1)
    transposed_image3 = np.rot90(resized_image, 2)
    transposed_image4 = np.rot90(resized_image, 3)

    num_of_results = len(masks)

    results = []
    results.append({'input': resized_image, 'output':num_of_results})
    results.append({'input': transposed_image, 'output':num_of_results})
    results.append({'input': resized_image2, 'output':num_of_results})
    results.append({'input': resized_image3, 'output':num_of_results})
    results.append({'input': resized_image4, 'output':num_of_results})
    results.append({'input': transposed_image2, 'output':num_of_results})
    results.append({'input': transposed_image3, 'output':num_of_results})
    results.append({'input': transposed_image4, 'output':num_of_results})

    return pd.DataFrame.from_dict(results)




def get_inputs_for_k_means_cnn(test_size = 0.1):
    gen = generate_input_image_and_masks()
    dfs = []

    for count, _ in enumerate(range(max_images)):
        print(count)
        try:
            image, image_gm, masks = next(gen)
            dfs.append(preprocess_images_for_kmeans(image, masks))
        except StopIteration:
            traceback.print_exc()
            break

    df = pd.concat(dfs)
    df = df.sample(frac=1)

    x, y = [], []
    for _, i in df.iterrows():
        x.append(i['input'])
        #x.append(np.hstack([i['image'], i['general_image_stats']]))
        y.append(i['output'])

    x = np.array(x)
    y = np.vstack(y)

    x_train = x[0:int((1-test_size)*x.shape[0])]
    x_test = x[int((1-test_size)*x.shape[0]):]
    y_train = y[0:int((1-test_size)*x.shape[0])]
    y_test = y[int((1-test_size)*x.shape[0]):]
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print('inputs preprocessed')

    return x_train, x_test, y_train, y_test



def get_cnn():


    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.5, decay=1e-5, momentum=0.0, nesterov=False)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
    return model


def get_cnn_for_k():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(256, 256, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    sgd = optimizers.SGD(lr=0.5, decay=1e-5, momentum=0.0, nesterov=False)
    model.compile(loss=keras.losses.mean_squared_error, optimizer=sgd, metrics=['mae'])
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


def get_max_second_order_derivative(x, y):
    dy = np.diff(y, 1)
    dx = np.diff(x, 1)
    yfirst = dy / dx
    xfirst = 0.5 * (x[:-1] + x[1:])

    dyfirst = np.diff(yfirst, 1)
    dxfirst = np.diff(xfirst, 1)
    ysecond = dyfirst / dxfirst

    max = np.argmax(ysecond)
    xsecond = 0.5 * (xfirst[:-1] + xfirst[1:])


    return xsecond[max]


def get_outputs(model, sub_image_size):
    folders = glob.glob(files_loc + 'stage1_test/*/')
    random.shuffle(folders)
    output_dicts = []

    for folder in folders:
        try:
            image_location = glob.glob(folder + 'images/*')[0]
            print(image_location)
            #mask_locations = glob.glob(folder + 'masks/*')

            image_id = image_location.split('/')[-1].split('.')[0]

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

            predictions = model.predict(sub_image_array)

            confidence_tresholds = [.9]
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
                preprocessed_image_t = np.transpose(preprocessed_image)
                scipy.misc.imsave('preprocessed_file_{0}.jpg'.format(str(int(c*100))), preprocessed_image)
                scipy.misc.imsave('preprocessed_file_t_{0}.jpg'.format(str(int(c*100))), preprocessed_image_t)
                scipy.misc.imsave('starting_file_{0}.jpg'.format(str(int(c*100))), np_image)

            #kmeans with elbow method
            nc = range(1, 50)
            kmeans = [KMeans(n_clusters=i) for i in nc]
            score = [kmeans[i].fit(cluster_dataset).score(cluster_dataset) for i in range(len(kmeans))]
            x_values = np.array([float(i) for i in nc])
            score_np = np.array(score)

            optimal_k = get_max_second_order_derivative(x_values, score_np)
            print(optimal_k, image_location)

            cluster_model = KMeans(n_clusters=int(optimal_k))
            predictions = cluster_model.fit_predict(cluster_dataset)

            cluster_locations = dict()
            for i, j in zip(cluster_dataset, predictions):
                cluster_locations.setdefault(j, [])
                cluster_locations[j].append(i)

            output_dicts.extend(to_output_format(cluster_locations, np_image, image_id))

        except:
            traceback.print_exc()

    df = pd.DataFrame.from_dict(output_dicts)
    df.to_csv(files_loc + 'output.csv', index = False)


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



    return res_str




def to_output_format(label_dict, np_image, image_name):

    output_dicts = []


    for i in label_dict.keys():
        image_copy = np_image.copy()
        image_copy.fill(0)
        for j in label_dict[i]:
            image_copy[j[1], j[0]] = 1

        flat_image = image_copy.flatten()

        output_dict = dict()
        output_dict['ImageId'] = image_name
        output_dict['EncodedPixels'] = get_outputs_from_flat_array(flat_image)
        output_dicts.append(output_dict)
    return output_dicts


def main():
    try:
        model = load_model(files_loc + 'cnn1.h5')
        print(model)
    except:
        traceback.print_exc()
        df = get_dataframes(32)
        x_train, x_test, y_train, y_test = get_model_inputs(df, x_labels=['image'])
        #train_models(x_train, x_test, y_train, y_test)

        model = get_cnn()

        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
        model.save(files_loc + 'cnn1.h5')

    get_outputs(model, 32)



if __name__ == '__main__':
    #main()
    x_train, x_test, y_train, y_test = get_inputs_for_k_means_cnn()
    model = get_cnn_for_k()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)



