#!/home/hangyu5/anaconda2/envs/py3dl/bin/python
import argparse
import argh
from time import time
from contextlib import contextmanager
import os
import random
import re
import sys
from collections import namedtuple

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from config import FLAGS, HPS


@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    logger.info(f"{message}: {(tock - tick):.3f} seconds")


'''
params:
    @ train_step: total number of mini-batch updates
    @ usage: learning rate annealling
'''


def schedule_lrn_rate(train_step):
    """train_step equals total number of min_batch updates"""
    f = 1  # rl schedule factor
    lr = 1e-3
    if train_step < 1 * f:
        lr = 1e-3  # 1e-1 blows up, sometimes 1e-2 blows up too.
    elif train_step < 2 * f:
        lr = 1e-4
    elif train_step < 3 * f:
        lr = 1e-4
    elif train_step < 4 * f:
        lr = 1e-4
    elif train_step < 5 * f:
        lr = 1e-5
    else:
        lr = 1e-5
    return lr


'''
params:
    @ usage: Go text protocol to play in Sabaki
'''
# Credit: Brain Lee


def gtp(flags=FLAGS, hps=HPS):
    from utils.gtp_wrapper import make_gtp_instance
    engine = make_gtp_instance(flags=flags, hps=hps)
    if engine is None:
        sys.stderr.write("Unknown strategy")
        sys.exit()
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not engine.disconnect:
        inpt = input()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()


'''
params:
    @ usage: self play with search pipeline
'''


def selfplay(flags=FLAGS, hps=HPS):
    from utils.load_data_sets import DataSet
    from model.SelfPlayWorker import SelfPlayWorker
    from Network import Network

    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))
    #test_dataset = None

    """set the batch size to -1==None"""
    flags.n_batch = -1
    net = Network(flags, hps)
    Worker = SelfPlayWorker(net, flags)

    def train(epoch: int):
        lr = schedule_lrn_rate(epoch)
        Worker.run(lr=lr)

    # TODO: consider tensorflow copy_to_graph
    def get_best_model():
        return Network(flags, hps)

    def evaluate_generations():
        best_model = get_best_model()
        Worker.evaluate_model(best_model)

    def evaluate_testset():
        Worker.evaluate_testset(test_dataset)

    """Self Play Pipeline starts here"""
    for g_epoch in range(flags.global_epoch):
        logger.info(f'Global epoch {g_epoch} start.')

        """Train"""
        train(g_epoch)

        """Evaluate on test dataset"""
        evaluate_testset()

        """Evaluate against best model"""
        evaluate_generations()

        logger.info(f'Global epoch {g_epoch} finish.')


'''
params:
    @ usage: train a supervised learning network
'''


def train(flags=FLAGS, hps=HPS):
    from utils.load_data_sets import DataSet
    from Network import Network

    TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

    net = Network(flags, hps)

    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))

    train_chunk_files = [os.path.join(flags.processed_dir, fname)
                         for fname in os.listdir(flags.processed_dir)
                         if TRAINING_CHUNK_RE.match(fname)]

    def training_datasets():
        random.shuffle(train_chunk_files)
        return (DataSet.read(file) for file in train_chunk_files)

    global_step = 0
    lr = flags.lr

    with open("result.txt", "a") as f:
        for g_epoch in range(flags.global_epoch):

            """Train"""
            lr = schedule_lrn_rate(g_epoch)
            for train_dataset in training_datasets():
                global_step += 1
                # prepare training set
                logger.info(f"Global step {global_step} start")
                train_dataset.shuffle()
                with timer("training"):
                    net.train(train_dataset, lrn_rate=lr)

                """Evaluate"""
                if global_step % 1 == 0:
                    with timer("test set evaluation"):
                        net.test(test_dataset, proportion=0.25,
                                 force_save_model=global_step % 10 == 0)

                logger.info(f'Global step {global_step} finshed.')
            logger.info(f'Global epoch {g_epoch} finshed.')


'''
params:
    @ usage: test a trained network on test dataset
'''


def test(flags=FLAGS, hps=HPS):
    from utils.load_data_sets import DataSet
    from Network import Network
    import tensorflow as tf

    net = Network(flags, hps)

    # print(net.sess.run({var.name:var for var in tf.global_variables() if 'bn' in var.name}))

    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))

    with timer("test set evaluation"):
        net.test(test_dataset, proportion=0.25, force_save_model=False)


if __name__ == '__main__':

    if not os.path.exists('./train_log'):
        os.makedirs('./train_log')

    if not os.path.exists('./test_log'):
        os.makedirs('./test_log')

    if not os.path.exists('./savedmodels'):
        os.makedirs('./savedmodels')

    if not os.path.exists('./result.txt'):
        # hacky way to creat a file
        open("result.txt", "a").close()

    fn = {'train': lambda: train(),
          'gtp': lambda: gtp(),
          'selfplay': lambda: selfplay(),
          'test': lambda: test()}

    if fn.get(FLAGS.MODE, 0) != 0:
        fn[FLAGS.MODE]()
    else:
        logger.info('Please choose a mode among "train", "selfplay", "gtp", and "test".')
