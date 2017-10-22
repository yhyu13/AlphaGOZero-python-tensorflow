from resnet_model import *

class AlphaGoZeroResNet(ResNet):

    def __init__(self, hps, images, labels, zs, mode):
        self.zs = zs
        super().__init__(hps, images, labels, mode)

    # override _residual block to repliate AlphaGoZero architecture
    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        
        """Residual unit with 2 sub layers.
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
        """
        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
            x = self._batch_norm('bn1', x)
            x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub2'):
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
            x = self._batch_norm('bn2', x)

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x
            x = self._relu(x, self.hps.relu_leakiness)

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    # overrride _build_model to replicate AlphaGoZero Architecture
    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._images
            x = self._conv('init_conv', x, 3, 17, 256, self._stride_arr(1))
            # 3x3x1 256 followed by batch norm and relu
            x = self._batch_norm('initial_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)

        strides = [1, 1, 1]
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual
            # deep residual
            #filters = [16, 16, 32, 64]
            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # wide residual
            filters = [256, 256, 640, 1280]
            # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1),
                             False)

        '''
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2],
                         self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1),
                             False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3],
                         self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1),
                             False)
        
        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)
        '''
        with tf.variable_scope('policy_head'):
            logits = self._conv('policy_conv', x, 1, 256, 2, self._stride_arr(1))
            logits = self._batch_norm('policy_bn', logits)
            logits = self._relu(logits, self.hps.relu_leakiness)
            logits = self._fully_connected(logits, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('value_head'):
            value = self._conv('value_conv', x, 1, 256, 1, self._stride_arr(1))
            value = self._batch_norm('value_bn', logits)
            value = self._relu(value, self.hps.relu_leakiness)
            value = self._fully_connected(value, 256)
            value = self._relu(value, self.hps.relu_leakiness)
            value = self._fully_connected(value,1)
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
