import tensorflow as tf
from Layers import *

class Conv5PosDepFC1ELU: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/eval_conv5posdepfc1ELU_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 64
        NKfirst = 64
        Nfc = 256
        conv1 = ELU_conv_pos_dep_bias(feature_planes, 5, Nfeat, NKfirst, N, 'conv1')
        conv2 = ELU_conv_pos_dep_bias(conv1, 3, NKfirst, NK, N, 'conv2')
        conv3 = ELU_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = ELU_conv_pos_dep_bias(conv3, 3, NK, NK, N, 'conv4')
        conv5 = ELU_conv_pos_dep_bias(conv4, 3, NK, NK, N, 'conv5')
        conv6 = ELU_conv_pos_dep_bias(conv5, 3, NK, NK, N, 'conv6')
        conv7 = ELU_conv_pos_dep_bias(conv6, 3, NK, NK, N, 'conv7')
        conv8 = ELU_conv_pos_dep_bias(conv7, 3, NK, NK, N, 'conv8')
        conv9 = ELU_conv_pos_dep_bias(conv8, 3, NK, NK, N, 'conv9')
        conv10 = ELU_conv_pos_dep_bias(conv9, 3, NK, NK, N, 'conv10')
        conv11 = ELU_conv_pos_dep_bias(conv10, 3, NK, NK, N, 'conv11')
        conv11_flat = tf.reshape(conv11, [-1, NK*N*N])
        fc = ELU_fully_connected_layer(conv11_flat, NK*N*N, Nfc)
        score = tf.tanh(linear_layer(fc, Nfc, 1))
        return score

class Conv11PosDepFC1ELU: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/eval_conv11posdepfc1ELU_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 256
        NKfirst = 256
        Nfc = 256
        conv1 = ELU_conv_pos_dep_bias(feature_planes, 5, Nfeat, NKfirst, N, 'conv1')
        conv2 = ELU_conv_pos_dep_bias(conv1, 3, NKfirst, NK, N, 'conv2')
        conv3 = ELU_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = ELU_conv_pos_dep_bias(conv3, 3, NK, NK, N, 'conv4')
        conv5 = ELU_conv_pos_dep_bias(conv4, 3, NK, NK, N, 'conv5')
        conv6 = ELU_conv_pos_dep_bias(conv5, 3, NK, NK, N, 'conv6')
        conv7 = ELU_conv_pos_dep_bias(conv6, 3, NK, NK, N, 'conv7')
        conv8 = ELU_conv_pos_dep_bias(conv7, 3, NK, NK, N, 'conv8')
        conv9 = ELU_conv_pos_dep_bias(conv8, 3, NK, NK, N, 'conv9')
        conv10 = ELU_conv_pos_dep_bias(conv9, 3, NK, NK, N, 'conv10')
        conv11 = ELU_conv_pos_dep_bias(conv10, 3, NK, NK, N, 'conv11')
        conv11_flat = tf.reshape(conv11, [-1, NK*N*N])
        fc = ELU_fully_connected_layer(conv11_flat, NK*N*N, Nfc)
        score = tf.tanh(linear_layer(fc, Nfc, 1))
        return score

class Linear:
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/linear_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        features_flat = tf.reshape(feature_planes, [-1, N*N*Nfeat])
        weights = tf.Variable(tf.constant(0.0, shape=[N*N*Nfeat, 1]), name='weights')
        #weights = tf.constant(0.0, shape=[N*N*Nfeat, 1])
        bias = tf.Variable(tf.constant(0.0, shape=[1]))
        out = tf.matmul(features_flat, weights) + bias
        #out = tf.matmul(features_flat, weights)
        score = tf.tanh(out)
        return score

class Zero:
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/zero_N%d_fe%d" % (N, Nfeat)
    def inference(self, feature_planes, N, Nfeat):
        dummy = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='dummy')
        return dummy * tf.constant(0.0, dtype=tf.float32, shape=[128])
