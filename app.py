import json
from aiohttp import web

import commands

routes = web.RouteTableDef()


@routes.get('/')
async def ok(request):
    return web.Response(text="OK")


@routes.get('/network/train/{network_id}')
async def network_train(request):
    network_id = request.match_info['network_id']
    network_train_command = await commands.NetworkTrain.create(network_id)
    request.loop.create_task(network_train_command.train())
    return web.Response(text=json.dumps(network_train_command.operation_data))


@routes.get('/network/predict/{network_id}/{image_id}')
async def network_train(request):
    network_id = request.match_info['network_id']
    image_id = request.match_info['image_id']
    network_predict_command = await commands.NetworkPredict.create(network_id, image_id)
    prediction = await network_predict_command.predict()
    return web.Response(text=json.dumps(prediction.tolist()))


def create_app():
    app = web.Application(debug=True)
    app.add_routes(routes)
    return app
