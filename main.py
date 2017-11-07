# -*- coding: future_fstrings -*-
import argparse
import argh
from time import time
from contextlib import contextmanager
import os
import random
import re
import sys
from collections import namedtuple

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

parser = argparse.ArgumentParser(description='Define parameters.')
parser.add_argument('--n_epoch', type=int, default=5)
parser.add_argument('--global_epoch', type=int, default=20)
parser.add_argument('--n_batch', type=int, default=1)
parser.add_argument('--n_img_row', type=int, default=19)
parser.add_argument('--n_img_col', type=int, default=19)
parser.add_argument('--n_img_channels', type=int, default=17)
parser.add_argument('--n_classes', type=int, default=19**2+1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_factor', type=float, default=.1)
parser.add_argument('--n_resid_units', type=int, default=1)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--dataset', dest='processed_dir',default='./processed_data')
parser.add_argument('--model_path',dest='load_model_path',default='./savedmodels')
parser.add_argument('--model_type',dest='model',default='elu',help='choose activation method')
parser.add_argument('--optimizer',dest='opt',default='mom')
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
    print("%s: %.3f" % (message, (tock - tick)))

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

# Credit: Brain Lee
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

    global_step = 0
    lr = flags.lr
    with open("result.txt","a") as f:
        for g_epoch in range(flags.global_epoch):
            
            for file in train_chunk_files:
                global_step += 1
                # prepare training set
                print(f"Using {file}", file=f)
                train_dataset = DataSet.read(file)
                train_dataset.shuffle()
                with timer("training"):
                    # train 
                    run.train(train_dataset)
                if global_step % 1 == 0:
                    # eval
                    with timer("test set evaluation"):
                        run.test(test_dataset,proportion=.1)
                print(f'Global step {global_step} finshed.', file=f)
            print(f'Global epoch {g_epoch} finshed.', file=f)
        print('Now, I am the Master.', file=f)


if __name__ == '__main__':

    fn = {'train': lambda: train(),
          'gtp': lambda: gtp()}

    if fn.get(FLAGS.MODE,0) != 0:
        fn[FLAGS.MODE]()
    else:
        print('Please choose a mode between "train" and "gtp".')
