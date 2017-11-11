# -*- coding: future_fstrings -*-
import tensorflow as tf
import numpy as np
import os
import sys

from model.alphagozero_resnet_model import AlphaGoZeroResNet
from model.alphagozero_resnet_elu_model import AlphaGoZeroResNetELU
from model.alphagozero_resnet_full_model import AlphaGoZeroResNetFULL
import utils.features as features

class Network:

    def __init__(self,flags,hps):

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Basic info
        self.batch_num = flags.n_batch
        self.num_epoch = flags.n_epoch
        self.img_row = flags.n_img_row
        self.img_col = flags.n_img_col
        self.img_channels = flags.n_img_channels
        self.nb_classes = flags.n_classes
        self.lr = flags.lr
        self.lr_factor = flags.lr_factor
        self.force_save_model = flags.force_save_model
        self.optimizer_name = hps.optimizer

        '''
           img: ?x19x19x17
           labels: ?x362
           results: ?x1
        '''
        self.imgs = tf.placeholder(tf.float32, shape=[None, self.img_row, self.img_col, self.img_channels])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.nb_classes])
        self.results = tf.placeholder(tf.float32,shape=[None,1])

        # potentially add previous alphaGo mdoels
        # Right now, there are two models,
        # One bing the original AlphaGo Zero relu
        # Two being the elu deep residul net with AlphaGo Zero architecture
        models = {'elu': lambda: AlphaGoZeroResNetELU(hps, self.imgs, self.labels, self.results,'train'),
                  'full': lambda: AlphaGoZeroResNetFULL(hps, self.imgs, self.labels, self.results,'train'),
                  'original': lambda: AlphaGoZeroResNet(hps, self.imgs, self.labels, self.results,'train')}
        print('Building Model...')
        self.model = models[flags.model]()
        self.model.build_graph()
        print(f'Building Model Complete...\nTotal parameters: {self.model.total_parameters()}')

        self.summary = self.model.summaries

        if not os.path.exists('./train_log'):
            os.makedirs('./train_log')

        if not os.path.exists('./savedmodels'):
            os.makedirs('./savedmodels')

        if not os.path.exists('./result.txt'):
            # hacky way to creat a file
            open("result.txt", "a").close()

        self.train_writer = tf.summary.FileWriter("./train_log", self.sess.graph)
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)

        if flags.load_model_path is not None:
            print('Loading Model...')
            try:
                ckpt = tf.train.get_checkpoint_state(flags.load_model_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print('Loading Model Succeeded...')
            except:
                print('Loading Model Failed')
                pass
        self.sess.run(tf.global_variables_initializer())
        print('Done initializing variables')

    '''
    params:
         @ imgs: bulk_extracted_feature(positions)
         # usage: queue prediction, self-play
    '''
    def run_many(self,imgs):
        imgs[:][...,16] = (imgs[:][...,16]-0.5)*2
        feed_dict = {self.imgs:imgs,self.model.training: False}
        move_probabilities,value = self.sess.run([self.model.prediction,self.model.value],feed_dict=feed_dict)

        # with multi-gpu, porbs and values are separated in each outputs
        # so vstack will merge them together.
        return np.vstack(move_probabilities), np.vstack(value)

    '''
    params:
         @ training_data
         @ reinforcement direction
         @ use sparse softmax to compute cross entropy
         @ learning rate
         # mini-batch training
    '''
    def train(self, training_data, direction=1.0, use_sparse=True):
        print('Training model...')
        self.num_iter = training_data.data_size // self.batch_num

        # Set default learning rate for scheduling
        for j in range(self.num_epoch):
            print(f'Local epoch {j+1}')

            for i in range(self.num_iter):
                batch = training_data.get_batch(self.batch_num)
                batch = [np.asarray(item).astype(np.float32) for item in batch]
                # convert the last feature: player colour to -1 & 1 rather than 0 & 1
                batch[0][...,16] = (batch[0][...,16]-0.5)*2
                # convert the game result: -1 & 1 rather than 0 & 1
                batch[2] = (batch[2]-0.5)*2

                feed_dict = {self.imgs: batch[0],
                             self.labels: batch[1],
                             self.results: batch[2],
                             self.model.reinforce_dir: direction, # +1 or -1 only used for self-play data, trivial in SL
                             self.model.use_sparse_sotfmax: 1 if use_sparse else -1, # +1 in SL, -1 in RL
                             self.model.training: True,
                             self.model.lrn_rate: self.lr} # scheduled learning rate

                try:
                    _, l, ac, result_ac,summary, lr,temp, global_norm = \
                    self.sess.run([self.model.train_op, self.model.cost,self.model.acc,\
                                   self.model.result_acc , self.summary, self.model.lrn_rate,\
                                   self.model.temp,self.model.norm], feed_dict=feed_dict)
                    global_step = self.sess.run(self.model.global_step)
                    self.train_writer.add_summary(summary,global_step)
                    self.sess.run(self.model.increase_global_step)
                    self.schedule_lrn_rate(global_step)

                    if i % 25 == 0:
                        with open("result.txt","a") as f:
                            f.write('Training...\n')
                            print >>f, f'Step {i} | Training loss {l:.2f} | Temperature {temp:.2f} | Magnitude of global norm {global_norm:.2f} | Total step {global_step} | Play move accuracy {ac:.4f} | Game outcome accuracy {result_ac:.2f}'
                            print >>f, f'Learning rate {"Adam" if self.optimizer_name=="adam" else lr}'
                        if ac > 0.7: # overfitting, abort, check evaluation
                            return
                except KeyboardInterrupt:
                    sys.exit()
                except tf.errors.InvalidArgumentError:
                    print(f'Step {i+1} corrupts. Discard.')
                    continue

    '''
    params:
       @ test_data: test.chunk.gz 10**5 positions
       @ proportion: how much proportion to evaluate
       usage: evaluate
    '''
    def test(self,test_data, proportion=0.1):

        print('Running evaluation...')
        num_minibatches = test_data.data_size // self.batch_num
        test_data.shuffle()
        test_loss, test_acc, test_result_acc , n_batch = 0, 0, 0,0
        for i in range(int(num_minibatches * proportion)):
            batch = test_data.get_batch(self.batch_num)
            batch = [np.asarray(item).astype(np.float32) for item in batch]
            # convert the last feature: player colour to -1 & 1 from 0 & 1
            batch[0][...,16] = (batch[0][...,16]-0.5)*2
            batch[2] = (batch[2]-0.5)*2

            feed_dict_eval = {self.imgs: batch[0],
                              self.labels: batch[1],
                              self.results:batch[2],
                              self.model.training: False}

            loss, ac, result_acc = self.sess.run([self.model.cost, self.model.acc,self.model.result_acc], feed_dict=feed_dict_eval)
            test_loss += loss
            test_acc += ac
            test_result_acc += result_acc
            n_batch += 1

        tot_test_loss = test_loss / (n_batch-1e-2)
        tot_test_acc = test_acc / (n_batch-1e-2)
        test_result_acc = test_result_acc / (n_batch-1e-2)

        with open("result.txt","a") as f:
            f.write('Running evaluation...\n')
            print >>f, f'Test loss: {tot_test_loss:.2f}'
            print >>f, f'Play move test accuracy: {tot_test_acc:.4f}'
            print >>f, f'Win ratio test accuracy: {test_result_acc:.2f}'

        if tot_test_acc > 0.2 or self.force_save_model:
            # if test acc is bigger than 20%, save or force save model
            self.saver.save(self.sess,f'./savedmodels/model-{tot_test_acc:.4f}.ckpt',\
                            global_step=self.sess.run(self.model.global_step))

    def schedule_lrn_rate(self, train_step):
        """train_step equals total number of min_batch updates"""
        if train_step < 200000:
            self.lr = 1e-2
        elif train_step < 400000:
            self.lr = 1e-2
        elif train_step < 600000:
            self.lr = 1e-3
        elif train_step < 700000:
            self.lr = 1e-4
        elif train_step < 800000:
            self.lr = 1e-5
        else:
            self.lr = 1e-5
