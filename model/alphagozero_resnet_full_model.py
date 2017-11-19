from model.alphagozero_resnet_model import *

class AlphaGoZeroResNetFULL(AlphaGoZeroResNet):

    def __init__(self, *args, **kwargs):
        super(AlphaGoZeroResNetFULL,self).__init__(*args, **kwargs)

    # override _residual block to be full pre-activation residual block
    # https://arxiv.org/pdf/1603.05027.pdf
    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):

        """
           @ f is now an identity function that makes ensures nonvaninshing gradient
           @ BN first impose stronger regularization which reduce overfitting
        """
        orig_x = x

        with tf.variable_scope('sub1'):
            # Batch normalisation
            x = self._batch_norm('bn1', x)
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            # Batch normalisation
            x = self._batch_norm('bn2', x)
            # A rectifier non-linearity
            x = self._relu(x, self.hps.relu_leakiness)
            # A convolution of 256 filters of kernel size 3x3 with stride 1
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            # A skip connection that adds the input to the block
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x

        #tf.logging.info('image after unit %s', x.get_shape())
        return x

    '''
    # overrride policy and value head to be fully convolutional network
    def _tower_loss(self,scope,image_batch,label_batch,z_batch,tower_idx):

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
            # _residual block to repliate AlphaGoZero architecture
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]))

        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('res_block_%d' % i):
                # _residual block to repliate AlphaGoZero architecture
                x = res_func(x, filters[1], filters[1], self._stride_arr(1))

        with tf.variable_scope('policy_head'):

            # Batch normalisation
            logits = self._batch_norm('policy_bn', x)
            # A rectifier non-linearity
            logits = self._relu(logits, self.hps.relu_leakiness)

            # A convolution of 362 filter of kernel size 1x1 with stride 1
            logits = self._conv('policy_conv', x, 1, 256, self.hps.num_classes , self._stride_arr(1))

            # defensive 1 step to temp annealling
            temp = tf.maximum(tf.train.exponential_decay(self.hps.temperature,self.global_step,1e4,0.8),1.)

            # a global average pool to a hidden layer of 362 classes
            logits = tf.divide(self._global_avg_pool(logits),temp)

            prediction = tf.nn.softmax(logits)
            self.prediction.append(prediction)

        with tf.variable_scope('value_head'):

            # Batch normalisation
            value = self._batch_norm('value_bn', x)
            # A rectifier non-linearity
            value = self._relu(value, self.hps.relu_leakiness)

            # a convolutional net goes from 256 filters to 1 filter
            value = self._conv('value_conv', value, 1, 256, 1, self._stride_arr(1))

            # a global average pool to a single scalar
            value = self._global_avg_pool(value)

            # A tanh non-linearity outputting a scalar in the range [1, 1]
            value = tf.tanh(value)
            self.value.append(value)

        with tf.variable_scope('costs'):
            self.use_sparse_sotfmax = tf.constant(1, tf.int32, name="condition")
            def f1(): return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(label_batch,axis=1))
            def f2(): return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch)
            xent = tf.cond(self.use_sparse_sotfmax > 0, f1 , f2 )
            squared_diff = tf.squared_difference(z_batch,value)
            ce = tf.reduce_mean(xent, name='cross_entropy')
            mse = tf.reduce_mean(squared_diff,name='mean_square_error')
            cost = ce*self.reinforce_dir + mse + self._decay()
            tf.summary.scalar(f'cost_tower_{tower_idx}', cost)
            tf.summary.scalar(f'ce_tower_{tower_idx}', ce)
            # scale MSE to [0,1]
            tf.summary.scalar(f'mse_tower_{tower_idx}', mse/4)

        with tf.variable_scope('move_acc'):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.argmax(label_batch,1))
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
        '''
