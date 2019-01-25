from tornado import web, ioloop

from handlers import NetworkTrainHandler, NetworkPredictHandler


def make_app(**config):
    return web.Application([
        (r"/train/", NetworkTrainHandler),
        (r"/predict/", NetworkPredictHandler)
    ], **config)


if __name__ == '__main__':
    app = make_app(debug=True, api='http://localhost/api')
    app.listen(8080)
    ioloop.IOLoop.current().start()
