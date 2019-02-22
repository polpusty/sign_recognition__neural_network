import json

from tornado import httpclient, ioloop, web

from nn.VGGNet import VGGNet


class NetworkTrainHandler(web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super(NetworkTrainHandler, self).__init__(*args, **kwargs)
        self.client = httpclient.AsyncHTTPClient()

    async def post(self):
        network_id = self.get_argument('network_id')
        operation_data = await self.add_operation(network_id)
        ioloop.IOLoop.current().spawn_callback(self.train_network, network_id)
        self.write(json.dumps(operation_data))

    async def train_network(self, network_id):
        network_data = await self.get_network_data(network_id)
        training_data = await self.get_training_data(network_data['collections'])
        network = VGGNet.get_network(network_data['collections'], self.settings['size_input_image'])
        await network.train(training_data, 20, 15)

    async def get_network_data(self, network_id):
        url = f'{self.settings["api"]}/networks/{network_id}/?embedded={{"collections":1,"collections.images":1}}'
        response = await self.client.fetch(url)
        return json.loads(response.body)

    async def get_training_data(self, collections):
        training_data = []
        for collection in collections:
            for image_data in collection['images']:
                image = await self.get_image(image_data['image'])
                training_data.append((image, collection))
        return training_data

    async def get_image(self, image_url):
        url = f'{self.settings["api"]}{image_url}'
        response = await self.client.fetch(url)
        return response.body

    async def add_operation(self, network_id, step=0.01):
        url = f'{self.settings["api"]}/operations'
        headers = {'Content-Type': 'application/json'}
        body = json.dumps({'name': f'train{network_id}',
                           'network': network_id,
                           'step': step})
        response = await self.client.fetch(url, method='POST', body=body, headers=headers)
        return json.loads(response.body)

    def data_received(self, chunk):
        return chunk
