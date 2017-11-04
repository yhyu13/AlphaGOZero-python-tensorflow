import tensorflow as tf
import numpy as np
import os
import sys

from model import alphagozero_resnet_model
from model import alphagozero_resnet_elu_model
import utils.features as features

class Network:
    def __init__(self,args,hps,load_model_path=None):

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
        self.lr = args.lr
        self.lr_schedule = args.lr_schedule
        self.lr_factor = args.lr_factor
        self.force_save_model = args.force_save_model
        self.optimizer_name = hps.optimizer

        self.img = tf.placeholder(tf.float32, shape=[None, self.img_row, self.img_col, self.img_channels])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.nb_classes])
        self.results = tf.placeholder(tf.float32,shape=[None,1])

        # whether this example should be positively or negatively reinforced.
        # Set to 1 for positive, -1 for negative.
        self.reinforce_direction = tf.placeholder(tf.float32, shape=[None])

        print('Building Model...')
        if 'elu' in args.model:
            self.model = alphagozero_resnet_elu_model.AlphaGoZeroResNetELU(hps, self.img, self.labels, self.results,'train')
        else:
            self.model = alphagozero_resnet_model.AlphaGoZeroResNet(hps, self.img, self.labels, self.results,'train')
        self.model.build_graph()
        print('Complete...')

        self.merged = self.model.summaries

        if not os.path.exists('./train_log'):
            os.makedirs('./train_log')

        if not os.path.exists('./savedmodels'):
            os.makedirs('./savedmodels')
        
        self.train_writer = tf.summary.FileWriter("./train_log", self.sess.graph)

        self.saver = tf.train.Saver(var_list=[v for v in tf.trainable_variables()],max_to_keep=0)

        if load_model_path is not None:
            print('Loading Model...')
            self.saver.restore(self.sess, load_model_path)
            print('Complete...')
            
        self.sess.run(tf.global_variables_initializer())
        print('Done initializing variables')

    def run_many(self,imgs):
        imgs[:][...,16] = (imgs[:][...,16]-0.5)*2
        move_probabilities,value = self.sess.run([self.model.predictions,self.model.value],feed_dict={self.img:imgs})
        return move_probabilities, value

    def train(self, training_data,direction=1.0):        
        print('Training model...')
        self.model.mode = 'train'
        self.num_iter = training_data.data_size // self.batch_num
        # Set default learning rate for scheduling
        lr = self.lr
        for j in range(self.num_epoch):
            print('Epoch {}'.format(j+1))
            # Decrease learning rate every args.lr_schedule epoch
            # By args.lr_factor
            if (j + 1) % self.lr_schedule == 0:
                lr *= self.lr_factor

            for i in range(self.num_iter):
                batch = training_data.get_batch(self.batch_num)
                batch = [np.asarray(item).astype(np.float32) for item in batch]
                # convert the last feature: player colour to -1 & 1 from 0 & 1
                batch[0][...,16] = (batch[0][...,16]-0.5)*2
                batch[2] = (batch[2]-0.5)*2
                
                feed_dict = {self.img: batch[0],
                             self.labels: batch[1],
                             self.results: batch[2],
                             self.model.reinforce_dir: direction}
                
                try:
                    _, l, ac, result_ac,summary, lr,temp, global_norm = \
                    self.sess.run([self.model.train_op, self.model.cost,self.model.acc,\
                                   self.model.result_acc , self.merged, self.model.lrn_rate,\
                                   self.model.temp,self.model.norm], feed_dict=feed_dict)
                    self.train_writer.add_summary(summary, i)
                    self.sess.run(self.model.increase_global_step)
                    
                    if i % 5 == 0:
                        print('Step {} | Training loss {:.2f} | Temperature {:.2f} | Magnitude of global norm {} | Total step {}'.format(i+1,\
                                                                                                             l,temp,global_norm,\
                                                                                                             self.sess.run(self.model.global_step)))
                        print('Play move training accuracy', ac)
                        print('Win ratio training accuracy', result_ac)
                        print('Learning rate', 'Adam' if self.optimizer_name=='adam' else lr)
                except KeyboardInterrupt:
                    sys.exit()
                except tf.errors.InvalidArgumentError:
                    print('Step {} corrupts. Discard.'.format(i+1))
                    continue

    def test(self,test_data, proportion=0.1):
        
        print('Running evaluation...')
        self.model.mode = 'eval'
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

        tot_test_loss = test_loss / (n_batch-1e2)
        tot_test_acc = test_acc / (n_batch-1e2)
        test_result_acc = test_result_acc / (n_batch-1e2)

        print('   Test loss: {}'.format(tot_test_loss))
        print('   play move test accuracy: {}'.format(tot_test_acc))
        print('   Win ratio test accuracy: {}'.format(test_result_acc))

        if tot_test_acc > 0.2 or self.force_save_model:
            # if test acc is bigger than 20%, save or force save model
            self.saver.save(self.sess,'./savedmodels/model-'+str(round(tot_test_acc,3))+'.ckpt')

