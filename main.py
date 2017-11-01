import argparse
import argh
from time import time
from contextlib import contextmanager
import os
import random
import re
import sys
from collections import namedtuple

@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    print("%s: %.3f" % (message, (tock - tick)))

parser = argparse.ArgumentParser(description='Define parameters.')
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--n_batch', type=int, default=128)
parser.add_argument('--n_img_row', type=int, default=19)
parser.add_argument('--n_img_col', type=int, default=19)
parser.add_argument('--n_img_channels', type=int, default=17)
parser.add_argument('--n_classes', type=int, default=19**2)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--n_resid_units', type=int, default=1)
parser.add_argument('--lr_schedule', type=int, default=60)
parser.add_argument('--lr_factor', type=float, default=1.)
parser.add_argument('--dataset', dest='processed_dir',default='./processed_data')
parser.add_argument('--model_path',dest='load_model_path',default='./savedmodels/model--0.0.ckpt')
parser.add_argument('--force_save',dest='force_save_model',action='store_true',default=False)
parser.add_argument('--policy',dest='policy',default='mcts')
parser.add_argument('--mode',dest='MODE')
args = parser.parse_args()


HParams = namedtuple('HParams',
                 'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                 'num_residual_units, use_bottleneck, weight_decay_rate, '
                 'relu_leakiness, optimizer, temperature, global_norm')

hps = HParams(batch_size=args.n_batch,
               num_classes=args.n_classes,
               min_lrn_rate=0.0001,
               lrn_rate=args.lr,
               num_residual_units=args.n_resid_units,
               use_bottleneck=False,
               weight_decay_rate=0.0001,
               relu_leakiness=0.1,
               optimizer='mom',
               temperature=1.0,
               global_norm=100)

# Credit: Brain Lee
def gtp(strategy=args.policy,args=args,load_model_path=args.load_model_path,hps=hps):
    from utils.gtp_wrapper import make_gtp_instance
    engine = make_gtp_instance(strategy,args,load_model_path,hps)
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
def train(args=args,hps=hps):

    from utils.load_data_sets import DataSet
    from Network import Network
    
    TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

    run = Network(args,hps,args.load_model_path)

    test_dataset = DataSet.read(os.path.join(args.processed_dir, "test.chunk.gz"))
    train_chunk_files = [os.path.join(args.processed_dir, fname) 
        for fname in os.listdir(args.processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]
    random.shuffle(train_chunk_files)

    global_step = 0
    for file in train_chunk_files:
        global_step += 1
        print("Using %s" % file)
        train_dataset = DataSet.read(file)
        train_dataset.shuffle()
        with timer("training"):
            run.train(train_dataset)

        if global_step % 1 == 0:
            with timer("test set evaluation"):
                run.test(test_dataset,proportion=.1)
    print('Now, I am the Master.')


if __name__ == '__main__':
    if args.MODE == 'train':
        train()
    elif args.MODE == 'gtp':
        gtp()
    else:
        print('Please choose a mode between "train" and "gtp".')
