from PIL import Image


def resize(image, size):
    return image.resize(size, Image.ANTIALIAS)


