def random_greyscale(img, p):

    import random
    import numpy as np

    if random.random() < p:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])

    else: return img

def augmentation_pipeline(img_arr):

    import keras.preprocessing.image as im

    img_arr = im.random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = im.random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = im.random_zoom(img_arr, zoom_range=(0.9, 2.0), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_greyscale(img_arr, 0.4)

    return img_arr