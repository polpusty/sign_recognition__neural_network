import json
import mimetypes
import uuid

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
                training_data.append((image, collection['class_code']))
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

    async def upload_network(self, network_data, network):
        url = f'{self.api_url}/networks/{network_data["_id"]}/'
        files = {'trained': (f'{network_data["_id"]}.bin', str(network))}
        boundary = uuid.uuid4().hex
        body = self.encode_form_data(files=files, boundary=boundary)
        headers = {'If-Match': network_data['_etag'],
                   'Content-Type': f'multipart/form-data; boundary={boundary}'}

        response = await self.client.fetch(url, method='PATCH', body=body, headers=headers, raise_error=False)
        return response.body

    def encode_form_data(self, files=None, fields=None, boundary=''):
        if fields is None:
            fields = {}
        if files is None:
            files = {}
        lines = []

        for key, value in fields.items():
            lines.append(f'--{boundary}')
            lines.append(f'Content-Disposition: form-data; name="{key}"')
            lines.append('')
            lines.append(value)

        for key, (filename, value) in files.items():
            lines.append(f'--{boundary}')
            lines.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"')
            lines.append(f'Content-Type: {self.get_content_type(filename)}')
            lines.append('')
            lines.append(value)

        lines.append(f'--{boundary}--')
        lines.append('')

        return '\r\n'.join(lines)

    @staticmethod
    def get_content_type(filename):
        return mimetypes.guess_type(filename)[0] or 'binary'
