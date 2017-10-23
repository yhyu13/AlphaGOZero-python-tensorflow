from resnet_model import *

class AlphaGoZeroResNet(ResNet):

    def __init__(self, hps, images, labels, zs, mode):
        self.zs = zs
        super().__init__(hps, images, labels, mode)

    # override _residual block to repliate AlphaGoZero architecture
    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        
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
            logits = self._fully_connected(logits, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('value_head'):
            # A convolution of 1 filter of kernel size 1x1 with stride 1
            value = self._conv('value_conv', x, 1, 256, 1, self._stride_arr(1))
            # Batch normalisation
            value = self._batch_norm('value_bn', logits)
            # A rectifier non-linearity
            value = self._relu(value, self.hps.relu_leakiness)
            # A fully connected linear layer to a hidden layer of size 256
            value = self._fully_connected(value, 256, 'fc1')
            # A rectifier non-linearity
            value = self._relu(value, self.hps.relu_leakiness)
            # A fully connected linear layer to a scalar
            value = self._fully_connected(value, 1, 'fc2')
            # A tanh non-linearity outputting a scalar in the range [1, 1]
            self.value = tf.tanh(value)
            
        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            squared_diff = tf.squared_difference(self.zs,self.value)
            self.cost = tf.reduce_mean(xent, name='xent') + tf.reduce_mean(squared_diff,name='squared_diff')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope('acc'):
            correct_prediction = tf.equal(
                tf.cast(tf.argmax(logits, 1), tf.int32), self.labels)
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accu')

            tf.summary.scalar('accuracy', self.acc)
