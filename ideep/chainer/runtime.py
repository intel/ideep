from mkldnn.api.support import engine
from mkldnn.api.support import stream


class Engine(object):
    __instance = None

    def __new__(cls, *args):
        if cls.__instance is None:
            cls.__instance = cls.inner()

        return cls.__instance

    class inner(engine):
        def __init__(self, other=None):
            super(Engine.inner, self).__init__(engine.cpu, 0)

            if other is None:
                self.id = 0
            else:
                self.id = int(other)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __int__(self):
            return 0

        def use(self):
            pass


def Stream():
    return stream(stream.eager)
