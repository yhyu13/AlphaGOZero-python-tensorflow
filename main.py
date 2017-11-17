#!/Users/yuhang/anaconda3/envs/py3dl/bin/python
"""
After installing all requirement,
Type 'which python' in your terminal.
And paste the path to your python env
in the bash bang line above.
Then 'chmod u+x main.py', so that main.py would become an excuteable.
"""

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
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--n_img_row', type=int, default=19)
parser.add_argument('--n_img_col', type=int, default=19)
parser.add_argument('--n_img_channels', type=int, default=17)
parser.add_argument('--n_classes', type=int, default=19**2+1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_resid_units', type=int, default=6)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--dataset', dest='processed_dir',default='./processed_data')
parser.add_argument('--model_path',dest='load_model_path',default='./savedmodels')#'./savedmodels'
parser.add_argument('--model_type',dest='model',default='full',\
                    help='choose residual block architecture {original,elu,full}')
parser.add_argument('--optimizer',dest='opt',default='adam')
parser.add_argument('--gtp_policy',dest='gpt_policy',default='mctspolicy',help='choose gtp bot player')#random,mctspolicy
parser.add_argument('--num_playouts',type=int,dest='num_playouts',default=1600,help='The number of MC search per move, the more the better.')
parser.add_argument('--selfplay_games_per_epoch',type=int,dest='selfplay_games_per_epoch',default=25000)
parser.add_argument('--mode',dest='MODE', default='train',help='among selfplay, gtp and train')
FLAGS = parser.parse_args()


HParams = namedtuple('HParams',
                 'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                 'num_residual_units, use_bottleneck, weight_decay_rate, '
                 'relu_leakiness, optimizer, temperature, global_norm, num_gpu, '
                 'name')

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
               num_gpu=FLAGS.n_gpu,
               name='01')

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
    engine = make_gtp_instance(flags=flags,hps=hps)
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
    from utils.load_data_sets import DataSet
    from model.SelfPlayWorker import SelfPlayWorker
    from Network import Network

    #test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))
    test_dataset = None

    """set the batch size to -1==None"""
    flags.n_batch = -1
    net = Network(flags,hps)
    Worker = SelfPlayWorker(net,flags)

    def train(epoch:int):
        lr = schedule_lrn_rate(epoch)
        Worker.run(lr=lr)

    def get_best_model():
        return net

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
        #evaluate_testset()

        """Evaluate against best model"""
        #evaluate_generations()

        logger.info(f'Global epoch {g_epoch} finish.')
    logger.info('Now, I am the Master! 现在，请叫我棋霸！')



def train(flags=FLAGS,hps=HPS):
    from utils.load_data_sets import DataSet
    from Network import Network

    TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

    net = Network(flags,hps)

    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))

    train_chunk_files = [os.path.join(flags.processed_dir, fname)
        for fname in os.listdir(flags.processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]

    def training_datasets():
        random.shuffle(train_chunk_files)
        return (DataSet.read(file) for file in train_chunk_files)

    global_step = 0
    lr = flags.lr
    with open("result.txt","a") as f:
        for g_epoch in range(flags.global_epoch):

            lr = schedule_lrn_rate(g_epoch)

            for train_dataset in training_datasets():
                global_step += 1
                # prepare training set
                logger.info(f"Using {file}", file=f)
                train_dataset.shuffle()
                with timer("training"):
                    # train
                    net.train(train_dataset,lrn_rate=lr)
                if global_step % 1 == 0:
                    # eval
                    with timer("test set evaluation"):
                        net.test(test_dataset,proportion=.1,force_save_model=False)

                logger.info(f'Global step {global_step} finshed.', file=f)
            logger.info(f'Global epoch {g_epoch} finshed.', file=f)
        logger.info('Now, I am the Master! 现在，请叫我棋霸！', file=f)



if __name__ == '__main__':

    fn = {'train': lambda: train(),
          'gtp': lambda: gtp(),
          'selfplay': lambda: selfplay()}

    if fn.get(FLAGS.MODE,0) != 0:
        fn[FLAGS.MODE]()
    else:
        logger.info('Please choose a mode among "train", "selfplay", and "gtp".')
