import json

from tornado import httpclient


class ApiClient:
    def __init__(self, api_url):
        self.api_url = api_url

        self.client = httpclient.AsyncHTTPClient()

    async def get_network_data(self, network_id):
        url = f'{self.api_url}/networks/{network_id}/?embedded={{"collections":1,"collections.images":1}}'
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
        url = f'{self.api_url}{image_url}'
        response = await self.client.fetch(url)
        return response.body

    async def add_operation(self, network_id, step=0.01):
        url = f'{self.api_url}/operations'
        headers = {'Content-Type': 'application/json'}
        body = json.dumps({'name': f'train{network_id}',
                           'network': network_id,
                           'step': step})
        response = await self.client.fetch(url, method='POST', body=body, headers=headers)
        return json.loads(response.body)
