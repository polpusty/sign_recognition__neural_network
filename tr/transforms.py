from nn.preprocessing import transform_image_to_array


class BytesToNdarray(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, _bytes):
        return transform_image_to_array(self.size, _bytes)
