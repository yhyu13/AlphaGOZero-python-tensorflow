'''
From MuGo:https://github.com/yhyu13/MuGo
'''

from collections import defaultdict
import functools
import itertools
import operator
import random
import re
import time

import gtp
import utils.go as go

KGS_COLUMNS = 'ABCDEFGHJKLMNOPQRST'
SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def parse_sgf_to_flat(sgf):
    return flatten_coords(parse_sgf_coords(sgf))

def flatten_coords(c):
    return go.N * c[0] + c[1]

def unflatten_coords(f):
    return divmod(f, go.N)

def parse_sgf_coords(s):
    'Interprets coords. aa is top left corner; sa is top right corner'
    if s is None or s == '':
        return None
    return SGF_COLUMNS.index(s[1]), SGF_COLUMNS.index(s[0])

def unparse_sgf_coords(c):
    if c is None:
        return ''
    return SGF_COLUMNS[c[1]] + SGF_COLUMNS[c[0]]

def parse_kgs_coords(s):
    'Interprets coords. A1 is bottom left; A9 is top left.'
    if s == 'pass':
        return None
    s = s.upper()
    col = KGS_COLUMNS.index(s[0])
    row_from_bottom = int(s[1:]) - 1
    return go.N - row_from_bottom - 1, col

def parse_pygtp_coords(vertex):
    'Interprets coords. (1, 1) is bottom left; (1, 9) is top left.'
    if vertex in (gtp.PASS, gtp.RESIGN):
        return None
    return go.N - vertex[1], vertex[0] - 1

def unparse_pygtp_coords(c):
    if c is None:
        return gtp.PASS
    return c[1] + 1, go.N - c[0]

def parse_game_result(result):
    if re.match(r'[bB]\+', result):
        return go.BLACK
    elif re.match(r'[wW]\+', result):
        return go.WHITE
    else:
        return None

def product(numbers):
    return functools.reduce(operator.mul, numbers)

def take_n(n, iterable):
    return list(itertools.islice(iterable, n))

def iter_chunks(chunk_size, iterator):
    while True:
        '''
        try statement portection from unclean data
        '''
        try:
            next_chunk = take_n(chunk_size, iterator)
            # If len(iterable) % chunk_size == 0, don't return an empty chunk.
            if next_chunk:
                yield next_chunk
            else:
                break
        except:
            continue

def shuffler(iterator, pool_size=10**5, refill_threshold=0.9):
    yields_between_refills = round(pool_size * (1 - refill_threshold))
    # initialize pool; this step may or may not exhaust the iterator.
    pool = take_n(pool_size, iterator)
    while True:
        random.shuffle(pool)
        for i in range(yields_between_refills):
            yield pool.pop()
        next_batch = take_n(yields_between_refills, iterator)
        if not next_batch:
            break
        pool.extend(next_batch)
    # finish consuming whatever's left - no need for further randomization.
    yield from pool

class timer(object):
    all_times = defaultdict(float)
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.tick = time.time()
    def __exit__(self, type, value, traceback):
        self.tock = time.time()
        self.all_times[self.label] += self.tock - self.tick
    @classmethod
    def print_times(cls):
        for k, v in cls.all_times.items():
            print("%s: %.3f" % (k, v))

'''
From ThalNet: https://github.com/yhyu13/thalnet
'''
import functools
from time import strftime

import tensorflow as tf

# lazy_property: no need for if $ not None logic
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')


# from https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2:

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def unzip(iterable):
    return zip(*iterable)


def single(list):
    first = list[0]

    assert (len(list) == 1)

    return first
