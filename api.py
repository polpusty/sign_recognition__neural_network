<<<<<<< HEAD
import json

import requests
from tornado import httpclient


class ApiClient:
    def __init__(self, api_url):
        self.api_url = api_url

        self.client = httpclient.AsyncHTTPClient()

    async def get_network_data(self, network_id):
        url = f'{self.api_url}/networks/{network_id}/?embedded={{"collections":1,"collections.images":1}}'
        response = await self.client.fetch(url)
        return json.loads(response.body)

    async def get_network(self, network_data=None, network_id=None):
        if network_data is None:
            network_data = await self.get_network_data(network_id)

        url = f'{self.api_url}{network_data["trained"]}'
        response = await self.client.fetch(url)
        return response.body

    async def update_network(self, network, **kwargs):
        if 'trained' in kwargs:
            raise httpclient.HTTPClientError("Use `upload_network` method")
        url = f'{self.api_url}/networks/{network["_id"]}/'
        headers = {'Content-Type': 'application/json',
                   'If-Match': network['_etag']}
        body = json.dumps(kwargs)
        response = await self.client.fetch(url, raise_error=False, method='PATCH', body=body, headers=headers)
        print(response.body)
        return json.loads(response.body)

    async def get_training_data(self, collections):
        training_data = []
        for collection in collections:
            for image_data in collection['images']:
                image = await self.get_image(image_data=image_data)
                training_data.append((image, collection['class_code']))
        return training_data

    async def get_image_data(self, image_id):
        url = f'{self.api_url}/images/{image_id}/'
        response = await self.client.fetch(url)
        return json.loads(response.body)

    async def get_image(self, image_data=None, image_id=None):
        if image_data is None:
            image_data = await self.get_image_data(image_id)

        url = f'{self.api_url}{image_data["image"]}'
        response = await self.client.fetch(url)
        return response.body

    async def add_operation(self, network_id, step=None, status="started"):
        if step is None:
            step = []
        url = f'{self.api_url}/operations'
        headers = {'Content-Type': 'application/json'}
        body = json.dumps({'name': f'train{network_id}',
                           'network': network_id,
                           'status': status,
                           'step': json.dumps(step)})
        response = await self.client.fetch(url, method='POST', body=body, headers=headers)
        return json.loads(response.body)

    async def update_operation(self, operation_data, training_progress, status="pending"):
        url = f'{self.api_url}/operations/{operation_data["_id"]}'
        headers = {'Content-Type': 'application/json',
                   'If-Match': operation_data['_etag']}
        body = json.dumps({'step': json.dumps(training_progress),
                           'status': status})
        response = await self.client.fetch(url, method='PATCH', body=body, headers=headers)
        return json.loads(response.body)

    async def upload_network(self, network_data, network):
        url = f'{self.api_url}/networks/{network_data["_id"]}/'
        files = [('trained', (f'{network_data["_id"]}.bin', network, 'binary'))]
        headers = {'If-Match': network_data['_etag']}

        response = requests.patch(url, files=files, headers=headers)
        return response.text
=======
import json

import requests
from tornado import httpclient


class ApiClient:
    def __init__(self, api_url):
        self.api_url = api_url

        self.client = httpclient.AsyncHTTPClient()

    async def get_network_data(self, network_id):
        url = f'{self.api_url}/networks/{network_id}/?embedded={{"collections":1,"collections.images":1}}'
        response = await self.client.fetch(url)
        return json.loads(response.body)

    async def get_network(self, network_data=None, network_id=None):
        if network_data is None:
            network_data = await self.get_network_data(network_id)

        url = f'{self.api_url}{network_data["trained"]}'
        response = await self.client.fetch(url)
        return response.body

    async def update_network(self, network, **kwargs):
        if 'trained' in kwargs:
            raise httpclient.HTTPClientError("Use `upload_network` method")
        url = f'{self.api_url}/networks/{network["_id"]}/'
        headers = {'Content-Type': 'application/json',
                   'If-Match': network['_etag']}
        body = json.dumps(kwargs)
        response = await self.client.fetch(url, raise_error=False, method='PATCH', body=body, headers=headers)
        print(response.body)
        return json.loads(response.body)

    async def get_training_data(self, collections):
        training_data = []
        for collection in collections:
            for image_data in collection['images']:
                image = await self.get_image(image_data=image_data)
                training_data.append((image, collection['class_code']))
        return training_data

    async def get_image_data(self, image_id):
        url = f'{self.api_url}/images/{image_id}/'
        response = await self.client.fetch(url)
        return json.loads(response.body)

    async def get_image(self, image_data=None, image_id=None):
        if image_data is None:
            image_data = await self.get_image_data(image_id)

        url = f'{self.api_url}{image_data["image"]}'
        response = await self.client.fetch(url)
        return response.body

    async def add_operation(self, network_id, step=None, status="started"):
        if step is None:
            step = []
        url = f'{self.api_url}/operations'
        headers = {'Content-Type': 'application/json'}
        body = json.dumps({'name': f'train{network_id}',
                           'network': network_id,
                           'status': status,
                           'step': json.dumps(step)})
        response = await self.client.fetch(url, method='POST', body=body, headers=headers)
        return json.loads(response.body)

    async def update_operation(self, operation_data, training_progress, status="pending"):
        url = f'{self.api_url}/operations/{operation_data["_id"]}'
        headers = {'Content-Type': 'application/json',
                   'If-Match': operation_data['_etag']}
        body = json.dumps({'step': json.dumps(training_progress),
                           'status': status})
        response = await self.client.fetch(url, method='PATCH', body=body, headers=headers)
        return json.loads(response.body)

    async def upload_network(self, network_data, network):
        url = f'{self.api_url}/networks/{network_data["_id"]}/'
        files = [('trained', (f'{network_data["_id"]}.bin', network, 'binary'))]
        headers = {'If-Match': network_data['_etag']}

        response = requests.patch(url, files=files, headers=headers)
        return response.text
>>>>>>> be1166b6555c110b5013a51645a58c3c880c2925
