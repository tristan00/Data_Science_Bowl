import pandas as pd
import re
import numpy as np


dtypes = { 'ImageId':str,
            'EncodedPixels':str}
df = pd.read_csv('output2.csv', dtype=dtypes)

empty_images = set(df['ImageId'])

df['EncodedPixels'] = df['EncodedPixels'].apply(lambda x: np.nan if len(re.findall(r'1 3 \d+ 3 \d+ 3', str(x))) > 0 and len(str(x).split(' ')) == 6 else x)
df = df.dropna(how='any')


df.to_csv('output.csv', index = False)

for i in empty_images:
    with open('output.csv', 'a') as infile:
        infile.writelin('{0},'.format(i))