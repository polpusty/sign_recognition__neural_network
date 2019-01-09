import json

from aiohttp import FormData

API_URL = 'http://localhost/api'


class ApiClient:
    def __init__(self, client):
        self.client = client

    async def create_operation(self, name, step):
        endpoint = f"operations/"
        data = {
            'name': name,
            'step': step
        }
        async with self.client.post(f"{API_URL}/{endpoint}", data=data) as response:
            return await response.json()

    async def update_operation(self, operation):
        headers = {'If-Match': operation['_etag']}
        endpoint = f"operations/{operation['_id']}/"
        data = {
            'step': operation['step']
        }
        async with self.client.patch(f"{API_URL}/{endpoint}", data=data, headers=headers) as response:
            return await response.json()

    async def get_image_data(self, image_id):
        endpoint = f"images/{image_id}"
        async with self.client.get(f"{API_URL}/{endpoint}") as response:
            return await response.json()

    async def get_image_file(self, image_url):
        async with self.client.get(f"{API_URL}{image_url}/") as response:
            return await response.read()

    async def get_collections_in_network(self, network_id):
        endpoint = f"collections/network/{network_id}"
        async with self.client.get(f"{API_URL}/{endpoint}") as response:
            return json.loads(await response.read())

    async def get_images_in_collection(self, collection_id):
        endpoint = 'images/?where={"collection": "' + collection_id + '"}'
        async with self.client.get(f"{API_URL}/{endpoint}") as response:
            return await response.json()

    async def get_network(self, network_id):
        endpoint = f"networks/{network_id}"
        async with self.client.get(f"{API_URL}/{endpoint}") as response:
            return await response.json()

    async def get_network_trained_file(self, network_trained_url):
        async with self.client.get(f"{API_URL}{network_trained_url}/") as response:
            return await response.read()

    async def upload_network_trained(self, network, network_trained):
        headers = {'If-Match': network['_etag']}
        endpoint = f"networks/{network['_id']}/"
        data = FormData()
        data.add_field('trained',
                       network_trained,
                       filename=f"{network['_id']}.bin",
                       content_type="binary")
        async with self.client.patch(f"{API_URL}/{endpoint}", data=data, headers=headers) as response:
            return await response.read()
