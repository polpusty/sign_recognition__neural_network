import json

from tornado import ioloop, web

from api import ApiClient
from nn.models import VGGNet, LeNet


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
        network = LeNet.get_network(network_data['collections'], (32, 32))
        await network.train(training_data, 15, 6)

    def data_received(self, chunk):
        return chunk
