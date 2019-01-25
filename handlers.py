import json

from tornado import web, httpclient

from nn.VGGNet import VGGNet


class NetworkTrainHandler(web.RequestHandler):
    async def post(self):
        network_id = self.get_argument('network_id')
        network_data = await self.get_network_data(network_id)
        operation_data = await self.add_operation(network_id)
        network = self.get_network(len(network_data['collections']))
        self.write(json.dumps(operation_data))

    async def get_network_data(self, network_id):
        client = httpclient.AsyncHTTPClient()
        response = await client.fetch(
            'http://localhost/api/networks/%s/?embedded={"collections":1,"collections.images":1}' % network_id)
        return json.loads(response.body)

    async def add_operation(self, network_id, step=0.01):
        headers = {'Content-Type': 'application/json'}
        client = httpclient.AsyncHTTPClient()
        body = json.dumps({'name': 'train %s' % network_id, 'network': network_id, 'step': step})
        response = await client.fetch("http://localhost/api/operations", method='POST', body=body, headers=headers)
        return json.loads(response.body)

    def get_network(self, number_classes):
        return VGGNet.get_network(number_classes)


class NetworkPredictHandler(web.RequestHandler):
    def post(self):
        network_id = self.get_argument('network_id')
        image_id = self.get_argument('image_id')
        result = None
        self.write(result)
