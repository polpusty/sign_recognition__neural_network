import io

from aiohttp import ClientSession
import dill
from PIL import Image
import numpy as np
import scipy.misc

from api import ApiClient
from nn.math_functions import softmax
from nn.VGGNet import VGGNet


class NetworkCommand:

    async def create_operation(self, name, step):
        async with ClientSession() as client:
            api = ApiClient(client)
            return await api.create_operation(name, step)

    @staticmethod
    async def update_operation(operation):
        async with ClientSession() as client:
            api = ApiClient(client)
            return await api.update_operation(operation)

    @staticmethod
    async def get_network(network_id):
        async with ClientSession() as client:
            api = ApiClient(client)
            return await api.get_network(network_id)

    async def get_image(self, image_url):
        async with ClientSession() as client:
            api = ApiClient(client)
            image_stream = await api.get_image_file(image_url)
            return image_stream

    def transform_image_bytes_to_array(self, image_bytes):
        stream = io.BytesIO(image_bytes)
        image = Image.open(stream)
        image = image.resize((32, 32), Image.ANTIALIAS)
        result_stream = io.BytesIO()
        image.save(result_stream, "PNG")
        image_array = scipy.misc.imread(result_stream).swapaxes(2, 0)
        return image_array


class NetworkTrain(NetworkCommand):
    @classmethod
    async def create(cls, network_id):
        collections = await cls.get_network_collections(network_id)
        self = NetworkTrain(network_id, len(collections))
        self.network_data = await self.get_network(self.network_id)
        self.operation_data = await self.create_operation(f"trained network {network_id}", 0)
        self.collections = collections
        return self

    def __init__(self, network_id, number_classes):
        self.network_id = network_id
        self.network = VGGNet.get_network(number_classes)
        self.network_data = None
        self.operation_data = None
        self.collections = []

    @staticmethod
    async def get_network_collections(network_id):
        async with ClientSession() as client:
            api = ApiClient(client)
            return await api.get_collections_in_network(network_id)

    async def get_collection_images(self, collection_id):
        async with ClientSession() as client:
            api = ApiClient(client)
            return await api.get_images_in_collection(collection_id)

    def number_classes(self):
        return len(self.collections)

    def get_result_for_classes(self, number_class):
        result = np.zeros((self.number_classes(), 1))
        result[number_class] = 1.0
        return result

    async def get_training_data(self):
        training_data = list()
        for number_class, collection in enumerate(self.collections):
            images_data = await self.get_collection_images(collection['_id'])
            images_data = images_data['_items']
            for index, image_data in enumerate(images_data):
                image = await self.get_image(image_data['image'])
                image_array = self.transform_image_bytes_to_array(image)
                training_data.append((image_array, self.get_result_for_classes(number_class)))
        return training_data

    async def train(self):
        training_data = await self.get_training_data()
        await self.network.train(training_data)
        await self.upload_network_object(self.network, self.network_data)

    @staticmethod
    async def upload_network_object(network, network_data):
        stream = io.BytesIO()
        dill.dump(network, stream)
        async with ClientSession() as client:
            api = ApiClient(client)
            await api.upload_network_trained(network_data, stream.getvalue())


class NetworkPredict(NetworkCommand):
    @classmethod
    async def create(cls, network_id, image_id):
        self = NetworkPredict(network_id, image_id)
        self.network_data = await self.get_network(self.network_id)
        self.network = await self.get_network_trained(self.network_data['trained'])
        self.image_data = await self.get_image_data(self.image_id)
        return self

    def __init__(self, network_id, image_id):
        self.network_id = network_id
        self.image_id = image_id
        self.network_data = None
        self.image_data = None
        self.network = None

    async def get_network_trained(self, network_trained_url):
        async with ClientSession() as client:
            api = ApiClient(client)
            stream = io.BytesIO(await api.get_network_trained_file(network_trained_url))
            return dill.load(stream)

    async def get_image_data(self, image_id):
        async with ClientSession() as client:
            api = ApiClient(client)
            return await api.get_image_data(image_id)

    async def predict(self):
        image = await self.get_image(self.image_data['image'])
        image_array = self.transform_image_bytes_to_array(image)
        return softmax(self.network.forward(image_array))
