import argparse
from model import alphagozero_resent_model

class GOTrainEnv:
    def __init__(self):

        # The data, shuffled and split between train and test sets
        self.x_train, self.y_train, self.x_test, self.y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

        # Reorder dimensions for tensorflow
        '''
        self.mean = np.mean(self.x_train, axis=0, keepdims=True)
        self.std = np.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std
        '''
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)
        print('y_train shape:', self.y_train.shape)
        print('y_test shape:', self.y_test.shape)

        # For generator
        self.num_examples = self.x_train.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # Basic info
        self.batch_num = args.n_batch
        self.num_epoch = args.n_epoch
        self.img_row = args.n_img_row
        self.img_col = args.n_img_col
        self.img_channels = args.n_img_channels
        self.nb_classes = args.n_classes
        self.num_iter = self.x_train.shape[0] / self.batch_num  # per epoch

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self.batch_size = batch_size

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size

        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]

            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]

    def train(self, hps):
        from time import time
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
    
        img = tf.placeholder(tf.float32, shape=[self.batch_num, 32, 32, 3])
        labels = tf.placeholder(tf.int32, shape=[self.batch_num, ])

        model = alphagozero_resnet_model.AlphaGoZeroResNet(hps, img, labels, zs,'train')
        model.build_graph()

        merged = model.summaries
        train_writer = tf.summary.FileWriter("/tmp/train_log", sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print('Done initializing variables')
        print('Running model...')

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
                batch = self.next_batch(self.batch_num)
                feed_dict = {img: batch[0],
                             labels: batch[1],
                             model.lrn_rate: lr}
                _, l, ac, summary, lr, global_norm = sess.run([model.train_op, model.cost, model.acc, merged, model.lrn_rate,model.norm], feed_dict=feed_dict)
                train_writer.add_summary(summary, i)
                sess.run(model.increase_global_step)
                
                if i % 200 == 0:
                    print('step', i+1)
                    print('Training loss', l)
                    print('Training accuracy', ac)
                    print('Learning rate', round(lr,10))
                    print('Magnitude of global norm', round(global_norm,2))
                    print('Total step', sess.run(model.global_step))

            print('Running evaluation...')

            test_loss, test_acc, n_batch = 0, 0, 0
            for batch in tl.iterate.minibatches(inputs=self.x_test,
                                                targets=self.y_test,
                                                batch_size=self.batch_num,
                                                shuffle=False):
                feed_dict_eval = {img: batch[0], labels: batch[1]}

                loss, ac = sess.run([model.cost, model.acc], feed_dict=feed_dict_eval)
                test_loss += loss
                test_acc += ac
                n_batch += 1

            tot_test_loss = test_loss / n_batch
            tot_test_acc = test_acc / n_batch

            print('   Test loss: {}'.format(tot_test_loss))
            print('   Test accuracy: {}'.format(tot_test_acc))
            print('   Total training hour: {:.2f}'.format((time()-start_time)/3600.))

            if tot_test_acc > 0.4:
                # if test acc is bigger than 40%, save
                saver.save(sess,'savedmodels/model-'+str(round(tot_test_acc,3))+'.ckpt')

        print('Completed training and evaluation. A Go god is born!')

if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Define parameters.')

    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--n_batch', type=int, default=128)
    parser.add_argument('--n_img_row', type=int, default=19)
    parser.add_argument('--n_img_col', type=int, default=19)
    parser.add_argument('--n_img_channels', type=int, default=17)
    parser.add_argument('--n_classes', type=int, default=362)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--n_resid_units', type=int, default=20)
    parser.add_argument('--lr_schedule', type=int, default=60)
    parser.add_argument('--lr_factor', type=float, default=0.1)

    args = parser.parse_args()
    
    run = GoTrainEnv.Supervised_Learning_Env()

    hps = alphagozero_resent_model.HParams(batch_size=run.batch_num,
                               num_classes=run.nb_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=args.lr,
                               num_residual_units=args.n_resid_units,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='adam')

    run.train(hps)
