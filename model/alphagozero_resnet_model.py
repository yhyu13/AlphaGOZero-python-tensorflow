from model.resnet_model import *

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class AlphaGoZeroResNet(ResNet):

    def __init__(self, hps, images, labels, zs, mode):
        self.zs = zs
        self.training = tf.placeholder(tf.bool)
        super(AlphaGoZeroResNet, self).__init__(hps, images, labels, mode)

    # override _batch_norm
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_mean, mean, 0.99))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_variance, variance, 0.99))

            tf.summary.histogram(moving_mean.op.name, moving_mean)
            tf.summary.histogram(moving_variance.op.name, moving_variance)

            def train():
                # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)

            def test():
                return tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, 0.001)

            y = tf.cond(tf.equal(self.training, tf.constant(True)), train, test)
            y.set_shape(x.get_shape())
            return y

    # override _conv to use He initialization with truncated normal to prevent dead neural
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = in_filters + out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.truncated_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    # override _residual block to repliate AlphaGoZero architecture
    def _residual(self, x, in_filter, out_filter, stride):
        """Build a residual block for the model."""
        orig_x = x

        with tf.variable_scope('sub1'):
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
            # Batch normalisation
            x = self._batch_norm('bn1', x)
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub2'):
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
            # Batch normalisation
            x = self._batch_norm('bn2', x)

        with tf.variable_scope('sub_add'):
            # A skip connection that adds the input to the block
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    # override build graph
    def build_graph(self):
        """Build a whole graph for the model."""

        ''' https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

            build_graph
                 |---tower_loss
                      |---average_grads
                           |---add_L2_loss
                   grads---|
            tran_op---|

        '''

        with tf.device('/cpu:0'):

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.increase_global_step = self.global_step.assign_add(1)

            self.lrn_rate = tf.maximum(tf.train.exponential_decay(
                self.hps.lrn_rate, self.global_step, 1e3, 0.66), 1.)
            # self.lrn_rate = tf.Variable(self.hps.lrn_rate, dtype=tf.float32, trainable=False)
            tf.summary.scalar('learning_rate', self.lrn_rate)
            self.reinforce_dir = tf.Variable(1., dtype=tf.float32, trainable=False)

            if self.hps.optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hps.optimizer == 'mom':
                self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            elif self.hps.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lrn_rate)

            image_batches = tf.split(self.images, self.hps.num_gpu, axis=0)
            label_batches = tf.split(self.labels, self.hps.num_gpu, axis=0)
            z_batches = tf.split(self.zs, self.hps.num_gpu, axis=0)
            tower_grads = [None] * self.hps.num_gpu
            self.prediction = []
            self.value = []
            self.cost, self.acc, self.result_acc, self.temp = 0, 0, 0, 0

        with tf.variable_scope(tf.get_variable_scope()):
            """Build the core model within the graph."""
            for i in range(self.hps.num_gpu):
                with tf.device(f'/gpu:{i}'):
                    with tf.name_scope(f'TOWER_{i}') as scope:

                        image_batch, label_batch, z_batch = image_batches[i], label_batches[i], z_batches[i]
                        loss, move_acc, result_acc, temp = self._tower_loss(
                            scope, image_batch, label_batch, z_batch, tower_idx=i)
                        # reuse variable happens here
                        tf.get_variable_scope().reuse_variables()
                        grad = self.optimizer.compute_gradients(loss)
                        tower_grads[i] = grad
                        self.cost += loss
                        self.acc += move_acc
                        self.result_acc += result_acc
                        self.temp += temp

        self.cost /= self.hps.num_gpu
        self.acc /= self.hps.num_gpu
        self.result_acc /= self.hps.num_gpu
        self.temp /= self.hps.num_gpu
        grads = self._average_gradients(tower_grads)

        if self.mode == 'train':
            self._build_train_op(grads)

        self.summaries = tf.summary.merge_all()

    # overrride _build_model to be an empty method
    def _build_model(self):
        pass

    def _tower_loss(self, scope, image_batch, label_batch, z_batch, tower_idx):
        """Build the residual tower within the model."""
        with tf.variable_scope('init'):
            x = image_batch
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('init_conv', x, 3, 17, 256, self._stride_arr(1))
            # Batch normalisation
            x = self._batch_norm('initial_bn', x)
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)

        strides = [1, 1, 1]
        res_func = self._residual
        filters = [256, 256]

        with tf.variable_scope('res_block_0'):
            # _residual block in AlphaGoZero architecture
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]))

        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('res_block_%d' % i):
                # _residual block in AlphaGoZero architecture
                x = res_func(x, filters[1], filters[1], self._stride_arr(1))

        with tf.variable_scope('policy_head'):
            # A convolution of 1 filter of kernel size 1x1 with stride 1
            logits = self._conv('policy_conv', x, 1, 256, 2, self._stride_arr(1))
            # Batch normalisation
            logits = self._batch_norm('policy_bn', logits)
            # A rectifier non-linearity
            logits = self._relu(logits, self.hps.relu_leakiness)
            # A fully connected linear layer that outputs a vector of
            # size 192^2 + 1 = 362 corresponding to logit probabilities
            # for all intersections and the pass move

            # defensive 1 step to temp annealling
            temp = tf.maximum(tf.train.exponential_decay(
                self.hps.temperature, self.global_step, 1e4, 0.8), 1.)
            logits = tf.divide(self._fully_connected(
                logits, self.hps.num_classes, 'policy_fc'), temp)
            prediction = tf.nn.softmax(logits)
            self.prediction.append(prediction)

        with tf.variable_scope('value_head'):
            # A convolution of 1 filter of kernel size 1x1 with stride 1
            value = self._conv('value_conv', x, 1, 256, 1, self._stride_arr(1))
            # Batch normalisation
            value = self._batch_norm('value_bn', value)
            # A rectifier non-linearity
            value = self._relu(value, self.hps.relu_leakiness)
            # A fully connected linear layer to a hidden layer of size 256
            value = self._fully_connected(value, 256, 'value_fc1')
            # A rectifier non-linearity
            value = self._relu(value, self.hps.relu_leakiness)
            # A fully connected linear layer to a scalar
            value = self._fully_connected(value, 1, 'value_fc2')
            # A tanh non-linearity outputting a scalar in the range [1, 1]
            value = tf.tanh(value)
            self.value.append(value)

        with tf.variable_scope('costs'):
            self.use_sparse_sotfmax = tf.constant(1, tf.int32, name="condition")

            def f1(): return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.argmax(label_batch, axis=1))

            def f2(): return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch)
            xent = tf.cond(self.use_sparse_sotfmax > 0, f1, f2)
            squared_diff = tf.squared_difference(z_batch, value)
            ce = tf.reduce_mean(xent, name='cross_entropy')
            mse = tf.reduce_mean(squared_diff, name='mean_square_error')
            cost = ce * self.reinforce_dir + mse + self._decay()
            tf.summary.scalar(f'cost_tower_{tower_idx}', cost)
            tf.summary.scalar(f'ce_tower_{tower_idx}', ce)
            # scale MSE to [0,1]
            tf.summary.scalar(f'mse_tower_{tower_idx}', mse / 4)

        with tf.variable_scope('move_acc'):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.argmax(label_batch, 1))
            acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='move_accu')
            tf.summary.scalar(f'move_accuracy_tower_{tower_idx}', acc)

        with tf.variable_scope('result_acc'):
            correct_prediction_2 = tf.equal(
                tf.sign(value), z_batch)
            result_acc = tf.reduce_mean(
                tf.cast(correct_prediction_2, tf.float32), name='result_accu')
            tf.summary.scalar(f'result_accuracy_tower_{tower_idx}', result_acc)

        return cost, acc, result_acc, temp

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
           Note that this function provides a synchronization point across all towers.
           Args:
              tower_grads: List of lists of (gradient, variable) tuples. The outer list
              is over individual gradients. The inner list is over the gradient
              calculation for each tower.
           Returns:
              List of pairs of (gradient, variable) where the gradient has been averaged
              across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, var in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                # logger.debug(f'Network variables: {var.name}')
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    # override build train op
    def _build_train_op(self, grads_vars):
        """Build training specific ops for the graph."""
        '''
        # Add histograms for trainable variables.
        # Add histograms for gradients.
        for grad, var in grads_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name, var)
                tf.summary.histogram(var.op.name + '/gradients', grad)
        '''
        # defensive step 2 to clip norm
        clipped_grads, self.norm = tf.clip_by_global_norm(
            [g for g, _ in grads_vars], self.hps.global_norm)

        # defensive step 3 check NaN
        # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
        grad_check = [tf.check_numerics(g, message='NaN Found!') for g in clipped_grads]
        with tf.control_dependencies(grad_check):
            apply_op = self.optimizer.apply_gradients(
                zip(clipped_grads, [v for _, v in grads_vars]),
                global_step=self.global_step, name='train_step')

            train_ops = [apply_op] + self._extra_train_ops
            # Group all updates to into a single train op.
            self.train_op = tf.group(*train_ops)
