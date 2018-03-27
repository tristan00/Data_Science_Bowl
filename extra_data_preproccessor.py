# extra data provided: https://www.kaggle.com/voglinio/external-h-e-data-with-mask-annotations/notebook
# cleaning up overlapping nuclei then putting it in same format as input data


import pandas as pd
import glob
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier

extra_data_input_path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/extra_data/'


def create_folder():
    pass

def image_to_location_list(prediction_image):
    output = []
    for i in range(prediction_image.shape[0]):
        for j in range(prediction_image.shape[1]):
            if prediction_image[i,j] > 0:
                output.append((i,j))
    return output


def locations_to_np_array(locations, np_image):
    np_image = np_image.copy()
    np_image[:] = 0
    for l in locations:
        np_image[l[0], l[1]] = 1
    return np_image


def point_to_cluster_classification_model_features(t):
    return [t[0], t[1], t[0] / (t[1] + 1), (t[1])/(t[0] + 1), t[0]+t[1], (t[0]+t[1])**2, (t[0]+t[1])/((t[0]+t[1] + 1)**2)]


def proccess_images(f):
    mask_locations = glob.glob(f + 'masks/*.png')
    masks = []
    for i in mask_locations:
        mask_image = Image.open(i)
        np_mask = np.array(mask_image.getdata())
        np_mask = np_mask.reshape(mask_image.size[1], mask_image.size[0])
        masks.append(np_mask)
    mask_sum = np.sum(masks)
    overlap = (mask_sum > 1).astype(int)

    cleaned_masks = []
    for m in masks:
        one_mask = (m > 0).astype(int)
        sub_mask = np.subtract(one_mask,overlap)
        cleaned_masks.append((sub_mask > 0).astype(int))

    o_pixels = image_to_location_list(overlap)

    clusters = dict()

    for count, m in cleaned_masks:
        clusters[count] = image_to_location_list(m)

    x = []
    y = []

    for i in clusters.keys():

        if len(clusters[i]) == 1:
            for j in clusters[i]:
                x.append(np.array(point_to_cluster_classification_model_features(j)))
                y.append(np.array([int(i)]))
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

    clf = ExtraTreesClassifier()
    clf.fit(x, y)

    if min(pred_x.shape) > 0:
        predictions = clf.predict(pred_x)
        for i, j in zip(o_pixels, predictions):
            clusters[j].add(i)




def main():
    folders = glob.glob(extra_data_input_path + '*/')

    for f in folders:
        proccess_images(f)

if __name__ == '__main__':
    main()