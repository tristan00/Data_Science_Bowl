import pandas as pd
import glob
import os


files_loc = 'C:/Users/tdelforge/Documents/Kaggle_datasets/data_science_bowl/'


folders = glob.glob(files_loc + 'stage1_test/*/images/*.png')

df = pd.read_csv('output.csv')

for i in folders:
    print(i)
    print(os.path.basename(i))