from tornado import web, ioloop

from handlers import NetworkTrainHandler


def make_app():
    settings = {
        'api': 'http://localhost/api',
        'size_input_image': (32, 32)
    }
    return web.Application([
        (r"/train/", NetworkTrainHandler),
    ], **settings)


if __name__ == '__main__':
    app = make_app()
    app.listen(8080)
    ioloop.IOLoop.current().start()
