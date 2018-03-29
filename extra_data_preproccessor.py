# extra data provided: https://www.kaggle.com/voglinio/external-h-e-data-with-mask-annotations/notebook
# cleaning up overlapping nuclei then putting it in same format as input data


import pandas as pd
import glob
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
import os
import scipy.misc

extra_data_input_path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/extra_data/'
extra_data_output_path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/extra_data_processed/'


def create_folder(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def image_to_location_list(prediction_image):
    output = set()
    for i in range(prediction_image.shape[0]):
        for j in range(prediction_image.shape[1]):
            if prediction_image[i,j] > 0:
                output.add((i,j))
    return output


def locations_to_np_array(locations, np_image):
    np_image = np_image.copy()
    np_image[:] = 0
    for l in locations:
        np_image[l[0], l[1]] = 255
    return np_image


def point_to_cluster_classification_model_features(t):
    return [t[0], t[1], t[0] / (t[1] + 1), (t[1])/(t[0] + 1), t[0]+t[1], (t[0]+t[1])**2, (t[0]+t[1])/((t[0]+t[1] + 1)**2)]


def proccess_images(f):
    print(f)
    image_location = glob.glob(f + 'images/*.png')[0]
    image_id = str(os.path.basename(image_location).split('.')[0])
    create_folder(extra_data_output_path + image_id)
    image_dir = create_folder(extra_data_output_path + image_id + '/images/')
    mask_dir = create_folder(extra_data_output_path + image_id + '/masks/')


    im = Image.open(image_location)
    im.save(image_dir + '/' + image_id + '.png')
    print('saved image at', image_dir + '/'+  image_id + '.png')


    mask_locations = glob.glob(f + 'masks/*.png')
    masks = []

    mask_image = Image.open(mask_locations[0])
    mask_sum = np.array(mask_image.getdata())
    mask_sum = mask_sum.reshape(mask_image.size[1], mask_image.size[0])
    mask_sum[:] = 0

    for i in mask_locations:
        mask_image = Image.open(i)
        np_mask = np.array(mask_image.getdata())
        np_mask = np_mask.reshape(mask_image.size[1], mask_image.size[0])
        np_mask = (np_mask > 0).astype(int)
        masks.append(np_mask)
        print('mask shape', np_mask.shape)
        mask_sum = np.add(mask_sum, np_mask)

    #mask_sum = np.add(masks)
    overlap = (mask_sum > 1).astype(int)

    print('overlap', overlap.shape)

    print('calculated mask overlap')

    cleaned_masks = []
    for m in masks:
        one_mask = (m > 0).astype(int)
        sub_mask = np.subtract(one_mask,overlap)
        cleaned_masks.append((sub_mask > 0).astype(int))

    o_pixels = image_to_location_list(overlap)

    clusters = dict()

    for count, m in enumerate(cleaned_masks):
        clusters[count] = image_to_location_list(m)

    x = []
    y = []

    for i in clusters.keys():

        # if len(clusters[i]) == 1:
        #     for j in clusters[i]:
        #         x.append(np.array(point_to_cluster_classification_model_features(j)))
        #         y.append(np.array([int(i)]))
        for j in clusters[i]:
            x.append(np.array(point_to_cluster_classification_model_features(j)))
            y.append(np.array([int(i)]))

    x = np.array(x)
    y = np.array(y)
    y = np.ravel(y)

    pred_x = []
    for i in o_pixels:
        pred_x.append(np.array(point_to_cluster_classification_model_features(i)))
    pred_x = np.array(pred_x)
    print('training model ')
    clf = ExtraTreesClassifier()
    clf.fit(x, y)

    if min(pred_x.shape) > 0:
        predictions = clf.predict(pred_x)
        for i, j in zip(o_pixels, predictions):
            clusters[j].add(i)
    print('saving masks ')
    for i in clusters.keys():
        scipy.misc.imsave(mask_dir + '/' +'{0}.png'.format(i), locations_to_np_array(clusters[i], mask_sum))




def main():
    folders = glob.glob(extra_data_input_path + '*/')

    create_folder(extra_data_output_path)
    for f in folders:
        proccess_images(f)

if __name__ == '__main__':
    main()