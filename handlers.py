<<<<<<< HEAD
import io
import json

import dill
from tornado import ioloop, web

from api import ApiClient
from nn.models import LeNet


class NetworkTrainHandler(web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super(NetworkTrainHandler, self).__init__(*args, **kwargs)
        self.client = ApiClient(self.settings['api'])

    async def post(self):
        network_id = self.get_argument('network_id')
        operation_data = await self.client.add_operation(network_id)
        ioloop.IOLoop.current().spawn_callback(self.train_network, network_id, operation_data)
        self.write(json.dumps(operation_data))

    async def train_network(self, network_id, operation_data):
        network_data = await self.client.get_network_data(network_id)
        training_data = await self.client.get_training_data(network_data['collections'])
        network = LeNet.get_network(list(map(lambda collection: collection['class_code'], network_data['collections'])),
                                    (32, 32), operation_data, self.settings['api'])
        await network.fit(training_data, 5, 15, self.client)
        await self.upload_trained_network(network, network_data)

    async def upload_trained_network(self, network, network_data):
        network_bytes = io.BytesIO()
        dill.dump(network, network_bytes)
        await self.client.upload_network(network_data, network_bytes.getvalue())

    def data_received(self, chunk):
        return chunk


class NetworkPredictHandler(web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super(NetworkPredictHandler, self).__init__(*args, **kwargs)
        self.client = ApiClient(self.settings['api'])

    async def post(self, *args, **kwargs):
        network_id = self.get_argument('network_id')
        image_id = self.get_argument('image_id')
        image = await self.client.get_image(image_id=image_id)
        network = await self.load_network(network_id)
        result = network.predict(image).tolist()
        print(result)
        self.write(json.dumps(result))

    async def load_network(self, network_id):
        network_bytes = io.BytesIO(await self.client.get_network(network_id=network_id))
        return dill.load(network_bytes)

    def data_received(self, chunk):
        return chunk
=======
import io
import json

import dill
from tornado import ioloop, web

from api import ApiClient
from nn.models import LeNet


class NetworkTrainHandler(web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super(NetworkTrainHandler, self).__init__(*args, **kwargs)
        self.client = ApiClient(self.settings['api'])

    async def post(self):
        network_id = self.get_argument('network_id')
        operation_data = await self.client.add_operation(network_id)
        ioloop.IOLoop.current().spawn_callback(self.train_network, network_id, operation_data)
        self.write(json.dumps(operation_data))

    async def train_network(self, network_id, operation_data):
        network_data = await self.client.get_network_data(network_id)
        training_data = await self.client.get_training_data(network_data['collections'])
        network = LeNet.get_network(list(map(lambda collection: collection['class_code'], network_data['collections'])),
                                    (32, 32), operation_data, self.settings['api'])
        await network.fit(training_data, 5, 15, self.client)
        await self.upload_trained_network(network, network_data)

    async def upload_trained_network(self, network, network_data):
        network_bytes = io.BytesIO()
        dill.dump(network, network_bytes)
        await self.client.upload_network(network_data, network_bytes.getvalue())

    def data_received(self, chunk):
        return chunk


class NetworkPredictHandler(web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super(NetworkPredictHandler, self).__init__(*args, **kwargs)
        self.client = ApiClient(self.settings['api'])

    async def post(self, *args, **kwargs):
        network_id = self.get_argument('network_id')
        image_id = self.get_argument('image_id')
        image = await self.client.get_image(image_id=image_id)
        network = await self.load_network(network_id)
        result = network.predict(image).tolist()
        self.write(json.dumps(result))

    async def load_network(self, network_id):
        network_bytes = io.BytesIO(await self.client.get_network(network_id=network_id))
        return dill.load(network_bytes)

    def data_received(self, chunk):
        return chunk
>>>>>>> be1166b6555c110b5013a51645a58c3c880c2925
