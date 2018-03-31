# extra data provided: https://www.kaggle.com/voglinio/external-h-e-data-with-mask-annotations/notebook
# cleaning up overlapping nuclei then putting it in same format as input data


import pandas as pd
import glob
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
import os
import scipy.misc

extra_data_input_path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/extra_data2/'
extra_data_output_path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/extra_data2_processed/'


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



def get_nuclei_from_mask(locations):
    location_set = set(locations)
    prediction_n_locations = location_set
    nuclei_predictions = dict()

    counter = 0
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
        prediction_n_locations = prediction_n_locations - temp_neucli_locations

        if len(temp_neucli_locations) > 0:
            nuclei_predictions[counter] = temp_neucli_locations
            counter += 1

    return nuclei_predictions


def proccess_images(f, m):
    print(f)
    image_id = str(os.path.basename(f).split('.')[0])
    create_folder(extra_data_output_path + image_id)
    image_dir = create_folder(extra_data_output_path + image_id + '/images/')
    mask_dir = create_folder(extra_data_output_path + image_id + '/masks/')


    im = Image.open(f)
    im.save(image_dir + '/' + image_id + '.png')
    print('saved image at', image_dir + '/'+  image_id + '.png')

    masks_input_im = Image.open(m).convert('LA')
    masks_input_np = np.array(masks_input_im.getdata())[:,0]
    masks_input_np = masks_input_np.reshape(masks_input_im.size[1], masks_input_im.size[0])

    m_locs = image_to_location_list(masks_input_np)
    clusters = get_nuclei_from_mask(m_locs)

    for k, v in clusters.items():
        scipy.misc.imsave(mask_dir + '/' + '{0}.png'.format(k), locations_to_np_array(v, masks_input_np))




def main():
    all_files = glob.glob(extra_data_input_path + '*.png')
    image_files = glob.glob(extra_data_input_path + '*_original_result.png')


    create_folder(extra_data_output_path)
    for f in image_files:
        print(f)
        mask_file = None
        for m in all_files:
            if '_'.join(f.split('/')[-1].split('_')[0:3]) in m:
                mask_file = m
                break

        proccess_images(f, m)

if __name__ == '__main__':
    main()