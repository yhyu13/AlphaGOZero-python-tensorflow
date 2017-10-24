import tensorflow as tf
import math

def conv(inputs, diameter, Nin, Nout, name):
    fan_in = diameter * diameter * Nin
    #stddev = math.sqrt(2.0 / fan_in)
    print "WARNING: USING DIFFERENT STDDEV FOR CONV!"
    stddev = math.sqrt(1.0 / fan_in)
    kernel = tf.Variable(tf.truncated_normal([diameter, diameter, Nin, Nout], stddev=stddev), name=name+'_kernel')
    return tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')

def conv_uniform_bias(inputs, diameter, Nin, Nout, name):
    bias = tf.Variable(tf.constant(0.0, shape=[Nout]), name=name+'_bias')
    return conv(inputs, diameter, Nin, Nout, name) + bias

def conv_pos_dep_bias(inputs, diameter, Nin, Nout, N, name):
    bias = tf.Variable(tf.constant(0.0, shape=[N, N, Nout]), name=name+'_bias')
    return conv(inputs, diameter, Nin, Nout, name) + bias

def ReLU_conv_uniform_bias(inputs, diameter, Nin, Nout, name):
    return tf.nn.relu(conv_uniform_bias(inputs, diameter, Nin, Nout, name))

def ReLU_conv_pos_dep_bias(inputs, diameter, Nin, Nout, N, name):
    return tf.nn.relu(conv_pos_dep_bias(inputs, diameter, Nin, Nout, N, name))

def ELU_conv_pos_dep_bias(inputs, diameter, Nin, Nout, N, name):
    return tf.nn.elu(conv_pos_dep_bias(inputs, diameter, Nin, Nout, N, name))

def linear_layer(inputs, Nin, Nout):
    #stddev = math.sqrt(2.0 / Nin)
    print "WARNING: USING DIFFERENT STDDEV FOR LINEAR!"
    stddev = math.sqrt(1.0 / Nin)
    print "linear layer using stddev =", stddev
    weights = tf.Variable(tf.truncated_normal([Nin, Nout], stddev=0.1))
    bias = tf.Variable(tf.constant(0.0, shape=[Nout]))
    out = tf.matmul(inputs, weights) + bias
    return out

def ReLU_fully_connected_layer(inputs, Nin, Nout):
    return tf.nn.relu(linear_layer(inputs, Nin, Nout))

def ELU_fully_connected_layer(inputs, Nin, Nout):
    return tf.nn.elu(linear_layer(inputs, Nin, Nout))


def preReLU_conv_pos_dep_bias(inputs, diameter, Nin, Nout, N, name):
    return conv_pos_dep_bias(tf.nn.relu(inputs), diameter, Nin, Nout, N, name)

def preELU_conv_pos_dep_bias(inputs, diameter, Nin, Nout, N, name):
    return conv_pos_dep_bias(tf.nn.elu(inputs), diameter, Nin, Nout, N, name)

def residual_block_preReLU_2convs_pos_dep_bias(inputs, diameter, Nfeat, N, name):
    conv1 = preReLU_conv_pos_dep_bias(inputs, diameter, Nfeat, Nfeat, N, name + '_1')
    conv2 = preReLU_conv_pos_dep_bias(conv1, diameter, Nfeat, Nfeat, N, name + '_2')
    return inputs + conv2

def residual_block_preELU_2convs_pos_dep_bias(inputs, diameter, Nfeat, N, name):
    conv1 = preELU_conv_pos_dep_bias(inputs, diameter, Nfeat, Nfeat, N, name + '_1')
    conv2 = preELU_conv_pos_dep_bias(conv1, diameter, Nfeat, Nfeat, N, name + '_2')
    return inputs + conv2

