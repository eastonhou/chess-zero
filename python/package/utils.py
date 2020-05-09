import time
from collections import defaultdict

class Timer:
    __TIMERS__ = defaultdict(lambda: 0)
    __TOTAL__ = defaultdict(lambda: 0)

    def __init__(self):
        self._time = time.time()

    def check(self, name=None, count=None):
        now = time.time()
        elapsed = now - self._time
        self._time = now
        if name is not None:
            __class__.__TIMERS__[name] += elapsed
            __class__.__TOTAL__[name] += count or 0
        return elapsed

    @staticmethod
    def print():
        for k, v in __class__.__TIMERS__.items():
            c = __class__.__TOTAL__[k]
            if c == 0:
                print(f'{k}: {v:>.2F}')
            else:
                print(f'{k}: {v:>.2F}/{c}={v/c}')

def enumerate_pairs(group):
    if isinstance(group, int):
        group = list(range(group))
    for j in range(len(group)):
        for i in range(j):
            yield group[i], group[j]