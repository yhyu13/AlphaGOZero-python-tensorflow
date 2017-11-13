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

parser = argparse.ArgumentParser(description='Define parameters.')
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--global_epoch', type=int, default=50)
parser.add_argument('--n_batch', type=int, default=128)
parser.add_argument('--n_img_row', type=int, default=19)
parser.add_argument('--n_img_col', type=int, default=19)
parser.add_argument('--n_img_channels', type=int, default=17)
parser.add_argument('--n_classes', type=int, default=19**2+1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_resid_units', type=int, default=6)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--dataset', dest='processed_dir',default='./processed_data')
parser.add_argument('--model_path',dest='load_model_path',default='./savedmodels')
parser.add_argument('--model_type',dest='model',default='full',\
                    help='choose residual block architecture {original,elu,full}')
parser.add_argument('--optimizer',dest='opt',default='adam')
parser.add_argument('--force_save',dest='force_save_model',action='store_true',default=False,\
                    help='if Ture, then save checkpoint for every evaluation period')
parser.add_argument('--policy',dest='policy',default='mctspolicy',help='choose gtp bot player')
parser.add_argument('--mode',dest='MODE', default='train',help='either gtp or train')
FLAGS = parser.parse_args()


HParams = namedtuple('HParams',
                 'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                 'num_residual_units, use_bottleneck, weight_decay_rate, '
                 'relu_leakiness, optimizer, temperature, global_norm, num_gpu')

HPS = HParams(batch_size=FLAGS.n_batch,
               num_classes=FLAGS.n_classes,
               min_lrn_rate=0.0001,
               lrn_rate=FLAGS.lr,
               num_residual_units=FLAGS.n_resid_units,
               use_bottleneck=False,
               weight_decay_rate=0.0001,
               relu_leakiness=0,
               optimizer=FLAGS.opt,
               temperature=1.0,
               global_norm=100,
               num_gpu=FLAGS.n_gpu)

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
    f = 10 # rl schedule factor
    lr = 1e-2
    if train_step < 1*f:
        lr = 1e-2 #1e-1 blows up, sometimes 1e-2 blows up too.
    elif train_step < 2*f:
        lr = 1e-2
    elif train_step < 3*f:
        lr = 1e-3
    elif train_step < int(3.5*f):
        lr = 1e-4
    elif train_step < 4*f:
        lr = 1e-5
    else:
        lr = 1e-5
    return lr

# Credit: Brain Lee
def gtp(flags=FLAGS,hps=HPS):
    from utils.gtp_wrapper import make_gtp_instance
    engine = make_gtp_instance(strategy=flags.policy,flags=flags,hps=hps)
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

def selfplay(flags=FLAGS,hps=HPS):
    import utils.go as go
    from utils.strategies import simulate_game_mcts,extract_moves
    from Network import Network

    """set the batch size to -1"""
    flags.n_batch = -1
    net = Network(flags,hps)
    N_gamer_per_train = 10
    N_games = 25000
    position = go.Position(to_play=go.BLACK)
    final_position_collections = []
    for g_epoch in range(flags.global_epoch):
        logger.info(f'Global epoch {g_epoch} start.')
        lr = schedule_lrn_rate(g_epoch)
        for i in range(N_games):
            """self play with MCTS search"""
            with timer(f"Self-Play Simulation Game #{i}"):
                final_position = simulate_game_mcts(net,position)
                logger.debug(f'\n{final_position}')
            final_position_collections.append(final_position)

            if (i+1) % N_gamer_per_train == 0:
                winners_training_samples, losers_training_samples = extract_moves(final_position_collections)
                net.train(winners_training_samples, direction=1.,lrn_rate=lr)
                net.train(losers_training_samples, direction=-1.,lrn_rate=lr)
                final_position_collections = []

        logger.info(f'Global epoch {g_epoch} finish.')
    logger.info('Now, I am the Master.')

def train(flags=FLAGS,hps=HPS):
    from utils.load_data_sets import DataSet
    from Network import Network

    TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

    run = Network(flags,hps)

    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))

    train_chunk_files = [os.path.join(flags.processed_dir, fname)
        for fname in os.listdir(flags.processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]

    random.shuffle(train_chunk_files)

    #training_datasets = [DataSet.read(file) for file in train_chunk_files]

    global_step = 0
    lr = flags.lr
    with open("result.txt","a") as f:
        for g_epoch in range(flags.global_epoch):

            lr = schedule_lrn_rate(g_epoch)

            #for train_dataset in training_datasets:
            for file in train_chunk_files:
                global_step += 1
                # prepare training set
                print(f"Using {file}", file=f)
                train_dataset = DataSet.read(file)
                train_dataset.shuffle()
                with timer("training"):
                    # train
                    run.train(train_dataset,lrn_rate=lr)
                if global_step % 1 == 0:
                    # eval
                    with timer("test set evaluation"):
                        run.test(test_dataset,proportion=.1)

                print(f'Global step {global_step} finshed.', file=f)
            print(f'Global epoch {g_epoch} finshed.', file=f)
        print('Now, I am the Master.', file=f)



if __name__ == '__main__':

    fn = {'train': lambda: train(),
          'gtp': lambda: gtp(),
          'selfplay': lambda: selfplay()}

    if fn.get(FLAGS.MODE,0) != 0:
        fn[FLAGS.MODE]()
    else:
        print('Please choose a mode among "train", "selfplay", and "gtp".')
