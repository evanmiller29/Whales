import math
from collections import Counter

import image_funcs as funcs

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

from tqdm import tqdm

from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, img_to_array)

np.random.seed(42)

# Reading in data and looking at images

train_df = pd.read_csv('data/train.csv')
train_df.head()

rand_rows = train_df.sample(frac=1.)[:20]
imgs = list(rand_rows['Image'])
labels = list(rand_rows['Id'])

funcs.plot_images_for_filenames(imgs, labels)

num_categories = len(train_df['Id'].unique())
print(f'Number of categories: {num_categories}')

size_buckets = Counter(train_df['Id'].value_counts().values)

total = len(train_df['Id'])
print(f'Total images in training set {total}')

w_1287fbc = train_df.loc[train_df['Id'] == 'w_1287fbc', :]
funcs.plot_images_for_filenames(list(w_1287fbc['Image']), None, rows=9)

# Looking at classes that have only one image

one_image_ids = train_df['Id'].value_counts().tail(8).keys()
one_image_filenames = []
labels = []

for i in one_image_ids:
    one_image_filenames.extend(list(train_df[train_df['Id'] == i]['Image']))
    labels.append(i)

funcs.plot_images_for_filenames(one_image_filenames, labels, rows=3)

# Checking the proportion of images that are greyscale already in a sample

# is_grey = [is_grey_scale(f'{INPUT_DIR}/train/{i}') for i in train_df['Image'].sample(frac=0.1)]
# grey_perc = round(sum([i for i in is_grey]) / len([i for i in is_grey]) * 100, 2)
# print(f"% of grey images: {grey_perc}")

# Examples of data augmentation

img = Image.open('data/train/ff38054f.jpg')
img_arr = img_to_array(img)
plt.imshow(img)

# Random rotations

imgs = [
    random_rotation(img_arr, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255 # Was * 255 but then it got confused as was float
    for _ in range(5)]

funcs.plot_images(imgs, None, rows=1)

# Random shift

imgs = [
    random_shift(img_arr, wrg=0.1, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255
    for _ in range(5)]
funcs.plot_images(imgs, None, rows=1)

# Random shear

imgs = [
    random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255
    for _ in range(5)]
funcs.plot_images(imgs, None, rows=1)

# Random zoom

imgs = [
    random_zoom(img_arr, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255
    for _ in range(5)]
funcs.plot_images(imgs, None, rows=1)

# Applying image augmentation pipeline

imgs = [funcs.augmentation_pipeline(img_arr) / 255 for _ in range(5)]
funcs.plot_images(imgs, None, rows=1)
