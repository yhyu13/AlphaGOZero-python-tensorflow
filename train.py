import argparse
from time import time
import math
import tensorflow as tf
import numpy as np
from contextlib import contextmanager
import os
import random
import re
import sys

from utils.load_data_sets import DataSet
import utils.features as features
import utils.go as go
from model import alphagozero_resnet_model

@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    print("%s: %.3f" % (message, (tock - tick)))

class GOTrainEnv:
    def __init__(self,hps,load_model_path=None):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # For generator
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # Basic info
        self.batch_num = args.n_batch
        self.num_epoch = args.n_epoch
        self.img_row = args.n_img_row
        self.img_col = args.n_img_col
        self.img_channels = args.n_img_channels
        self.nb_classes = args.n_classes

        self.img = tf.placeholder(tf.float32, shape=[None, self.img_row, self.img_col, self.img_channels])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.nb_classes])
        self.results = tf.placeholder(tf.float32,shape=[None,1])

        # whether this example should be positively or negatively reinforced.
        # Set to 1 for positive, -1 for negative.
        self.reinforce_direction = tf.placeholder(tf.float32, shape=[None])

        print('Building Model...')
        self.model = alphagozero_resnet_model.AlphaGoZeroResNet(hps, self.img, self.labels, self.results,'train')
        self.model.build_graph()
        print('Complete...')

        self.merged = self.model.summaries

        if not os.path.exists('./train_log'):
            os.makedirs('./train_log')

        if not os.path.exists('./savedmodels'):
            os.makedirs('./savedmodels')
        
        self.train_writer = tf.summary.FileWriter("./train_log", self.sess.graph)

        self.saver = tf.train.Saver()

        if load_model_path is not None:
            print('Loading Model...')
            self.saver.resotre(sess, load_model_path)
            print('Complete...')
            
        self.sess.run(tf.global_variables_initializer())
        print('Done initializing variables')

    def reinforcement(self, history, direction):
        print('Training model...')

        pass

    def train(self, training_data):
    
        
        print('Training model...')

        self.num_iter = training_data.data_size // self.batch_num

        # Set default learning rate for scheduling
        lr = args.lr
        start_time = time()
        for j in range(self.num_epoch):
            print('Epoch {}'.format(j+1))

            # Decrease learning rate every args.lr_schedule epoch
            # By args.lr_factor
            if (j + 1) % args.lr_schedule == 0:
                lr *= args.lr_factor

            for i in range(self.num_iter):
                batch = training_data.get_batch(self.batch_num)
                batch = [np.asarray(item).astype(np.float32) for item in batch]
                # convert the last feature: player colour to -1 & 1 from 0 & 1
                batch[0][...,16] = (batch[0][...,16]-0.5)*2
                batch[2] = (batch[2]-0.5)*2
                
                feed_dict = {self.img: batch[0],
                             self.labels: batch[1],
                             self.results: batch[2],
                             self.model.lrn_rate: lr}
                _, l, ac, result_ac,summary, lr,temp, global_norm = self.sess.run([self.model.train_op, self.model.cost,self.model.acc, self.model.result_acc , self.merged, self.model.lrn_rate,self.model.temp,self.model.norm], feed_dict=feed_dict)
                self.train_writer.add_summary(summary, i)
                self.sess.run(self.model.increase_global_step)
                
                if i % 100 == 0:
                    print('step', i+1)
                    print('Training loss', l)
                    print('Play move training accuracy', ac)
                    print('Win ratio training accuracy', result_ac)
                    print('Learning rate', lr)
                    print('Temperature', temp)
                    print('Magnitude of global norm', global_norm)
                    print('Total step', self.sess.run(self.model.global_step))
                    

    def test(self,test_data, proportion=0.1):
        
        print('Running evaluation...')

        num_minibatches = test_data.data_size // self.batch_num

        test_loss, test_acc, test_result_acc ,n_batch = 0, 0, 0,0
        for i in range(int(num_minibatches * proportion)):
            batch = test_data.get_batch(self.batch_num)
            batch = [np.asarray(item).astype(np.float32) for item in batch]
            # convert the last feature: player colour to -1 & 1 from 0 & 1
            batch[0][...,16] = (batch[0][...,16]-0.5)*2
            batch[2] = (batch[2]-0.5)*2
            
            feed_dict_eval = {self.img: batch[0], self.labels: batch[1],self.results:batch[2]}

            loss, ac, result_acc = self.sess.run([self.model.cost, self.model.acc,self.model.result_acc], feed_dict=feed_dict_eval)
            test_loss += loss
            test_acc += ac
            test_result_acc += result_acc
            n_batch += 1

        tot_test_loss = test_loss / n_batch
        tot_test_acc = test_acc / n_batch
        test_result_acc = test_result_acc / n_batch

        print('   Test loss: {}'.format(tot_test_loss))
        print('   play move test accuracy: {}'.format(tot_test_acc))
        print('   Win ratio test accuracy: {}'.format(test_result_acc))

        if tot_test_acc > 0.4:
            # if test acc is bigger than 40%, save
            self.saver.save(self.sess,'./savedmodels/model-'+str(round(tot_test_acc,3))+'.ckpt')

            

if __name__=="__main__":

    TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

    parser = argparse.ArgumentParser(description='Define parameters.')

    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--n_batch', type=int, default=128)
    parser.add_argument('--n_img_row', type=int, default=go.N)
    parser.add_argument('--n_img_col', type=int, default=go.N)
    parser.add_argument('--n_img_channels', type=int, default=17)
    parser.add_argument('--n_classes', type=int, default=go.N**2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--n_resid_units', type=int, default=20)
    parser.add_argument('--lr_schedule', type=int, default=60)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--dataset', dest='processed_dir',default='./processed_data')
    parser.add_argument('--model_path',dest='load_model_path',default=None)

    args = parser.parse_args()

    hps = alphagozero_resnet_model.HParams(batch_size=args.n_batch,
                               num_classes=args.n_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=args.lr,
                               num_residual_units=args.n_resid_units,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='adam')
    
    run = GOTrainEnv(hps,args.load_model_path)

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

        if global_step % 10 == 0:
            with timer("test set evaluation"):
                run.test(test_dataset,proportion=.1)
    print('Now, I am the Master.')
