from model.alphagozero_resnet_model import *


class AlphaGoZeroResNetELU(AlphaGoZeroResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _relu(self, x, leak=0):
        """change relu to elu"""
        return tf.nn.elu(x)

    # override _residual block with ELU model
    # https://arxiv.org/pdf/1604.04112.pdf
    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """split+conv+elu+conv+bach_norm+merge"""
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

        tf.logging.info('image after unit %s', x.get_shape())
        return x
