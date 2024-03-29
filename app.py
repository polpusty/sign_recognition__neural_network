from tornado import web, ioloop

from handlers import NetworkTrainHandler, NetworkPredictHandler


def make_app():
    settings = {
        'api': 'http://api',
        'size_input_image': (32, 32)
    }
    return web.Application([
        (r"/train/", NetworkTrainHandler),
        (r"/predict/", NetworkPredictHandler),
    ], **settings)


if __name__ == '__main__':
    app = make_app()
    app.listen(80)
    ioloop.IOLoop.current().start()
