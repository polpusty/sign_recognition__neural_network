import io

import numpy
import scipy.misc
from PIL import Image


def transform_image_to_array(size, image_stream):
    stream = io.BytesIO(image_stream)
    image = Image.open(stream)
    image = image.resize(size, Image.ANTIALIAS)
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return scipy.misc.imread(stream)


def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())
