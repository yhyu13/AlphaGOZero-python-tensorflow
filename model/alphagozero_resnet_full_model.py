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

        tf.logging.info('image after unit %s', x.get_shape())
        return x
