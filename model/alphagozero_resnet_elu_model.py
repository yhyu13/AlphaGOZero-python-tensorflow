from model.resnet_model import *

class AlphaGoZeroResNet(ResNet):

    def __init__(self, hps, images, labels, zs, mode):
        self.zs = zs
        super().__init__(hps, images, labels, mode)

    # override build graph
    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.increase_global_step = self.global_step.assign_add(self.hps.batch_size)
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _relu(x,leak=0):
        return tf.nn.elu(x)

    def _fully_connected(self, x, out_dim, name=''):
        """FullyConnected layer for final output."""
        x = tf.contrib.layers.flatten(x)
        w = tf.get_variable(
            name+'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable(name+'biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    # override _residual block to repliate AlphaGoZero architecture
    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        
        orig_x = x
        
        with tf.variable_scope('sub1'):
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
            # Batch normalisation
            #x = self._batch_norm('bn1', x)
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
            #x = self._relu(x, self.hps.relu_leakiness)

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    # overrride _build_model to replicate AlphaGoZero Architecture
    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._images
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('init_conv', x, 3, 17, 256, self._stride_arr(1))
            # Batch normalisation
            x = self._batch_norm('initial_bn', x)
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)

        strides = [1, 1, 1]
        activate_before_residual = [False, False, False] # futile
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [256, 512]
        else:
            res_func = self._residual
            filters = [256, 256]

        with tf.variable_scope('res_block_0'):
            # _residual block to repliate AlphaGoZero architecture
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('res_block_%d' % i):
                # _residual block to repliate AlphaGoZero architecture
                x = res_func(x, filters[1], filters[1], self._stride_arr(1),
                             False)

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
            self.temp = tf.maximum(tf.train.exponential_decay(100.,self.global_step,1e6,0.95),1.)
            logits = tf.divide(self._fully_connected(logits, self.hps.num_classes, 'policy_fc'),self.temp)
            self.predictions = tf.nn.softmax(logits)

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
            self.value = tf.tanh(value)
            
        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            squared_diff = tf.squared_difference(self.zs,self.value)
            self.cost = tf.reduce_mean(xent, name='xent') + tf.reduce_mean(squared_diff,name='squared_diff')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope('move_acc'):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.argmax(self.labels,1))
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='move_accu')

            tf.summary.scalar('move_accuracy', self.acc)

        with tf.variable_scope('result_acc'):
            correct_prediction_2 = tf.equal(
                tf.sign(self.value), self.zs)
            self.result_acc = tf.reduce_mean(
                tf.cast(correct_prediction_2, tf.float32), name='result_accu')

            tf.summary.scalar('resutl_accuracy', self.result_acc)

    # override build train op
    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)
        # defensive step 2 to clip norm
        grads,self.norm = tf.clip_by_global_norm(grads,100.)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(1e-4)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
