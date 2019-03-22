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
        ioloop.IOLoop.current().spawn_callback(self.train_network, network_id)
        self.write(json.dumps(operation_data))

    async def train_network(self, network_id):
        network_data = await self.client.get_network_data(network_id)
        training_data = await self.client.get_training_data(network_data['collections'])
        network = LeNet.get_network(list(map(lambda collection: collection['class_code'], network_data['collections'])),
                                    (32, 32))
        await network.fit(training_data, 15, 15)
        await self.upload_trained_network(network, network_data)

    async def upload_trained_network(self, network, network_data):
        network_bytes = io.BytesIO()
        dill.dump(network, network_bytes)
        await self.client.upload_network(network_data, network_bytes.getvalue())

    def data_received(self, chunk):
        return chunk
