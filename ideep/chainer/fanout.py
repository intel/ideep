import weakref


def fanout_recorder_clear(ref):
    FanoutRecorder.clear()


class FanoutRecorder(object):
    fanout = {}
    head = None

    @classmethod
    def new(recorder, func):
        if not recorder.fanout:
            recorder.head = weakref.ref(func, fanout_recorder_clear)

        if recorder.fanout.get(func.rank) is None:
            recorder.fanout[func.rank] = 0
        else:
            recorder.fanout[func.rank] += 1

        return recorder.fanout[func.rank]

    @classmethod
    def clear(recorder):
        if recorder.fanout:
            recorder.head = None
            recorder.fanout = {}
