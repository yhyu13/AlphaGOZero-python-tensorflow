import tensorflow as tf
from Layers import *

class Conv4PosDep:
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/influence_conv4posdep_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 16
        NKfirst = 16
        conv1 = ELU_conv_pos_dep_bias(feature_planes, 5, Nfeat, NKfirst, N, 'conv1')
        conv2 = ELU_conv_pos_dep_bias(conv1, 3, NKfirst, NK, N, 'conv2')
        conv3 = ELU_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = conv_pos_dep_bias(conv3, 3, NK, 1, N, 'conv4') 
        logits = tf.reshape(conv4, [-1, N*N])        
        return logits # use with sigmoid and sigmoid_cross_entropy_with_logits


class Conv12PosDepELU: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/influence_conv12posdep_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 192
        NKfirst = 192
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
        conv12 = conv_pos_dep_bias(conv11, 3, NK, 1, N, 'conv12') 
        logits = tf.reshape(conv12, [-1, N*N])        
        return logits

