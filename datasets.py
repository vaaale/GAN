import glob
import os

import numpy as np
from PIL import Image, ImageChops, ImageOps
from scipy.misc import imread


def make_thumb(f_in, f_out, size=(128, 128), pad=False):
    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    if pad:
        thumb = image.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - image_size[0]) / 2, 0)
        offset_y = max((size[1] - image_size[1]) / 2, 0)

        thumb = ImageChops.offset(thumb, offset_x, offset_y)

    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))

    thumb.save(f_out)


def load_data():
    files = glob.glob(os.path.join("../data", '*.jpg'))
    np.random.shuffle(files)
    data = []

    for i, file in enumerate(files):
        img = imread(file, mode='RGB')
        if len(img.shape) < 3:
            continue
        data.append(img)

    data = np.asarray(data)
    data = data / 255.
    return data, np.ones(shape=(data.shape[0], 1))
