import tensorflow as tf
from Layers import *

class Linear:
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/ckpts_linear_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        flat_features = tf.reshape(feature_planes, [-1, N*N*Nfeat])
        logits = linear_layer(flat_features, N*N*Nfeat, N*N)
        return logits

class SingleFull: # recommend 9x9, mbs=1000, adam, lr=0.003
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/ckpts_single_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        Nhidden = 1024
        flat_features = tf.reshape(feature_planes, [-1, N*N*Nfeat])
        hidden = fully_connected_layer(flat_features, N*N*Nfeat, Nhidden)
        logits = linear_layer(hidden, Nhidden, N*N)
        return logits

class Conv3Full: # recommend 9x9, mbs=1000, adam, lr=0.003
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/ckpts_conv3_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        NK = 32
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK)
        conv2 = conv_layer(conv1, 3, NK, NK)
        conv3 = conv_layer(conv2, 3, NK, NK)
        conv3_flat = tf.reshape(conv3, [-1, N*N*NK])
        hidden4 = fully_connected_layer(conv3_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden4, Nhidden, N*N)
        return logits

class Conv4Full: 
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/ckpts_conv4_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        NK = 64
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK, stddev=0.01)
        conv2 = conv_layer(conv1, 3, NK, NK, stddev=0.01)
        conv3 = conv_layer(conv2, 3, NK, NK, stddev=0.01)
        conv4 = conv_layer(conv3, 3, NK, NK, stddev=0.01)
        conv4_flat = tf.reshape(conv4, [-1, N*N*NK])
        hidden5 = fully_connected_layer(conv4_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden5, Nhidden, N*N)
        return logits

class Conv5Full: 
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/ckpts_conv5_full_N%d_mb%d_fe_%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK, stddev=0.01)
        conv2 = conv_layer(conv1, 3, NK, NK, stddev=0.01)
        conv3 = conv_layer(conv2, 3, NK, NK, stddev=0.01)
        conv4 = conv_layer(conv3, 3, NK, NK, stddev=0.01)
        conv5 = conv_layer(conv4, 3, NK, NK, stddev=0.01)
        conv5_flat = tf.reshape(conv5, [-1, N*N*NK])
        hidden6 = fully_connected_layer(conv5_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden6, Nhidden, N*N)
        return logits

class Conv8: 
    def __init__(self, N, Nfeat, minibatch_size=1000, learning_rate=0.0003):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/ckpts_conv8_N%d_mb%d_fe%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK, stddev=0.01)
        conv2 = conv_layer(conv1, 3, NK, NK, stddev=0.01)
        conv3 = conv_layer(conv2, 3, NK, NK, stddev=0.01)
        conv4 = conv_layer(conv3, 3, NK, NK, stddev=0.01)
        conv5 = conv_layer(conv4, 3, NK, NK, stddev=0.01)
        conv6 = conv_layer(conv5, 3, NK, NK, stddev=0.01)
        conv7 = conv_layer(conv6, 3, NK, NK, stddev=0.01)
        conv8 = conv_layer(conv7, 1, NK, 1, stddev=0.01) # todo: switch to no_relu
        conv8_flat = tf.reshape(conv8, [-1, N*N])        
        bias = tf.Variable(tf.constant(0, shape=[N*N], dtype=tf.float32)) # position-dependent bias
        logits = conv8_flat + bias
        return logits

class Conv8Full: 
    def __init__(self, N, Nfeat, minibatch_size, learning_rate):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/ckpts_conv8_full_N%d_mb%d_fe%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        Nhidden = 1024
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK)
        conv2 = conv_layer(conv1, 3, NK, NK)
        conv3 = conv_layer(conv2, 3, NK, NK)
        conv4 = conv_layer(conv3, 3, NK, NK)
        conv5 = conv_layer(conv4, 3, NK, NK)
        conv6 = conv_layer(conv5, 3, NK, NK)
        conv7 = conv_layer(conv6, 3, NK, NK)
        conv8 = conv_layer(conv7, 3, NK, NK)
        conv8_flat = tf.reshape(conv8, [-1, N*N*NK])        
        hidden9 = fully_connected_layer(conv8_flat, N*N*NK, Nhidden)
        logits = linear_layer(hidden9, Nhidden, N*N)
        return logits

# didn't get higher than ~33% on KGS :(
class Conv12: # AlphaGo architecture
    def __init__(self, N, Nfeat, minibatch_size=1000, learning_rate=0.0003):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/ckpts_conv12_N%d_mb%d_fe%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        conv1 = conv_layer(feature_planes, 5, Nfeat, NK)
        conv2 = conv_layer(conv1, 3, NK, NK)
        conv3 = conv_layer(conv2, 3, NK, NK)
        conv4 = conv_layer(conv3, 3, NK, NK)
        conv5 = conv_layer(conv4, 3, NK, NK)
        conv6 = conv_layer(conv5, 3, NK, NK)
        conv7 = conv_layer(conv6, 3, NK, NK)
        conv8 = conv_layer(conv7, 3, NK, NK)
        conv9 = conv_layer(conv8, 3, NK, NK)
        conv10 = conv_layer(conv9, 3, NK, NK)
        conv11 = conv_layer(conv10, 3, NK, NK)
        conv12 = conv_layer_no_relu(conv11, 1, NK, 1)
        conv12_flat = tf.reshape(conv12, [-1, N*N])        
        bias = tf.Variable(tf.constant(0, shape=[N*N], dtype=tf.float32)) # position-dependent bias
        logits = conv12_flat + bias
        return logits

# smallest network described in Maddison et al. paper
# They claim 37.5% accuracy on KGS
# One difference is that they have two output planes, one for each color
class MaddisonMinimal: 
    def __init__(self, N, Nfeat, minibatch_size=1000, learning_rate=0.0003):
        self.checkpoint_dir = "/home/greg/coding/ML/go/NN/work/checkpoints/maddison_minimal_N%d_mb%d_fe%d" % (N, minibatch_size, Nfeat)
        self.learning_rate = learning_rate
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 16
        conv1 = relu_conv_pos_dep_bias(feature_planes, 5, Nfeat, NK, self.N)
        conv2 = relu_conv_pos_dep_bias(conv1, 3, NK, NK, self.N)
        conv3 = relu_conv_pos_dep_bias(conv2, 3, NK, NK, self.N)
        conv4 = conv_pos_dep_bias(conv3, 1, NK, 1, self.N)
        logits = tf.reshape(conv4, [-1, N*N])
        return logits

class Conv6PosDep: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv6posdep_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 128
        conv1 = relu_conv_pos_dep_bias(feature_planes, 5, Nfeat, NK, N, 'conv1')
        conv2 = relu_conv_pos_dep_bias(conv1, 3, NK, NK, N, 'conv2')
        conv3 = relu_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = relu_conv_pos_dep_bias(conv3, 3, NK, NK, N, 'conv4')
        conv5 = relu_conv_pos_dep_bias(conv4, 3, NK, NK, N, 'conv5')
        conv6 = conv_pos_dep_bias(conv5, 3, NK, 1, N, 'conv6') 
        logits = tf.reshape(conv6, [-1, N*N])        
        return logits

# Got to ~46.7% on KGS after 58K minibatches of 256 (~12 hours of training), NK=192, learning_rate=0.03
class Conv8PosDep: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv8posdep_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 192
        NKfirst = 192
        conv1 = relu_conv_pos_dep_bias(feature_planes, 5, Nfeat, NKfirst, N, 'conv1')
        conv2 = relu_conv_pos_dep_bias(conv1, 3, NKfirst, NK, N, 'conv2')
        conv3 = relu_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = relu_conv_pos_dep_bias(conv3, 3, NK, NK, N, 'conv4')
        conv5 = relu_conv_pos_dep_bias(conv4, 3, NK, NK, N, 'conv5')
        conv6 = relu_conv_pos_dep_bias(conv5, 3, NK, NK, N, 'conv6')
        conv7 = relu_conv_pos_dep_bias(conv6, 3, NK, NK, N, 'conv7')
        conv8 = conv_pos_dep_bias(conv7, 3, NK, 1, N, 'conv8') 
        logits = tf.reshape(conv8, [-1, N*N])        
        return logits

class Conv10PosDep: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv10posdep_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 192
        NKfirst = 192
        conv1 = relu_conv_pos_dep_bias(feature_planes, 5, Nfeat, NKfirst, N, 'conv1')
        conv2 = relu_conv_pos_dep_bias(conv1, 3, NKfirst, NK, N, 'conv2')
        conv3 = relu_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = relu_conv_pos_dep_bias(conv3, 3, NK, NK, N, 'conv4')
        conv5 = relu_conv_pos_dep_bias(conv4, 3, NK, NK, N, 'conv5')
        conv6 = relu_conv_pos_dep_bias(conv5, 3, NK, NK, N, 'conv6')
        conv7 = relu_conv_pos_dep_bias(conv6, 3, NK, NK, N, 'conv7')
        conv8 = relu_conv_pos_dep_bias(conv7, 3, NK, NK, N, 'conv8')
        conv9 = relu_conv_pos_dep_bias(conv8, 3, NK, NK, N, 'conv9')
        conv10 = conv_pos_dep_bias(conv9, 3, NK, 1, N, 'conv10') 
        logits = tf.reshape(conv10, [-1, N*N])        
        return logits

# ELUs seem to give a ~1.1% increase in training accuracy out to at least 
# 12K minibatches of 256. This is with the centering transformation
#       loaded_feature_planes = (loaded_feature_planes.astype(np.float32) - 0.154) * 2.77
# on KGS data using make_feature_planes_stones_3liberties_4history_ko
# Using ReLUs, that centering transformation produces only a ~0.2% increase in training accuracy
# which seems to be disappearing after 8K minibatches. But with ELUs it is more permanent?
#
# Reached ~52.5% on KGS after ~200K minibatches of 256, using featurewise normalization,
# dropping the learning rate at the very end for a ~1.5% jump
class Conv10PosDepELU: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv10posdep_N%d_fe%d" % (N, Nfeat)
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
        conv10 = conv_pos_dep_bias(conv9, 3, NK, 1, N, 'conv10') 
        logits = tf.reshape(conv10, [-1, N*N])        
        return logits

class Conv12PosDepELU: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv12posdepELU_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 256
        NKfirst = 256
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

class Conv12PosDepELUBig: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv12posdepELUbig_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 512
        NKfirst = 512
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

class Conv16PosDepELU: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv16posdepELU_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 256
        NKfirst = 256
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
        conv12 = ELU_conv_pos_dep_bias(conv11, 3, NK, NK, N, 'conv12')
        conv13 = ELU_conv_pos_dep_bias(conv12, 3, NK, NK, N, 'conv13')
        conv14 = ELU_conv_pos_dep_bias(conv13, 3, NK, NK, N, 'conv14')
        conv15 = ELU_conv_pos_dep_bias(conv14, 3, NK, NK, N, 'conv15')
        conv16 = conv_pos_dep_bias(conv15, 3, NK, 1, N, 'conv16') 
        logits = tf.reshape(conv16, [-1, N*N])        
        return logits

class Conv4PosDepELU: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv4posdepELU_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 48
        NKfirst = 48
        conv1 = ELU_conv_pos_dep_bias(feature_planes, 5, Nfeat, NKfirst, N, 'conv1')
        conv2 = ELU_conv_pos_dep_bias(conv1, 3, NKfirst, NK, N, 'conv2')
        conv3 = ELU_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = conv_pos_dep_bias(conv3, 3, NK, 1, N, 'conv4') 
        logits = tf.reshape(conv4, [-1, N*N])        
        return logits

class Conv12PosDep: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/conv12posdep_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 192
        NKfirst = 192
        conv1 = relu_conv_pos_dep_bias(feature_planes, 5, Nfeat, NKfirst, N, 'conv1')
        conv2 = relu_conv_pos_dep_bias(conv1, 3, NKfirst, NK, N, 'conv2')
        conv3 = relu_conv_pos_dep_bias(conv2, 3, NK, NK, N, 'conv3')
        conv4 = relu_conv_pos_dep_bias(conv3, 3, NK, NK, N, 'conv4')
        conv5 = relu_conv_pos_dep_bias(conv4, 3, NK, NK, N, 'conv5')
        conv6 = relu_conv_pos_dep_bias(conv5, 3, NK, NK, N, 'conv6')
        conv7 = relu_conv_pos_dep_bias(conv6, 3, NK, NK, N, 'conv7') 
        conv8 = relu_conv_pos_dep_bias(conv7, 3, NK, NK, N, 'conv8')
        conv9 = relu_conv_pos_dep_bias(conv8, 3, NK, NK, N, 'conv9')
        conv10 = relu_conv_pos_dep_bias(conv9, 3, NK, NK, N, 'conv10')
        conv11 = relu_conv_pos_dep_bias(conv10, 3, NK, NK, N, 'conv11')
        conv12 = conv_pos_dep_bias(conv11, 3, NK, 1, N, 'conv12') 
        logits = tf.reshape(conv12, [-1, N*N])        
        return logits

class Res5x2PreELU:
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/res5x2_preelu_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 256
        conv_in = conv_pos_dep_bias(feature_planes, 5, Nfeat, NK, N, 'conv_in')
        res1 = residual_block_preELU_2convs_pos_dep_bias(conv_in, 3, NK, N, 'res1')
        res2 = residual_block_preELU_2convs_pos_dep_bias(res1, 3, NK, N, 'res2')
        res3 = residual_block_preELU_2convs_pos_dep_bias(res2, 3, NK, N, 'res3')
        res4 = residual_block_preELU_2convs_pos_dep_bias(res3, 3, NK, N, 'res4')
        res5 = residual_block_preELU_2convs_pos_dep_bias(res4, 3, NK, N, 'res5')
        conv_out = conv_pos_dep_bias(res5, 3, NK, 1, N, 'conv_out')
        logits = tf.reshape(conv_out, [-1, N*N])        
        return logits

class Res10x2PreELU:
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/res10x2_preelu_N%d_fe%d" % (N, Nfeat)
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        NK = 256
        conv_in = conv_pos_dep_bias(feature_planes, 5, Nfeat, NK, N, 'conv_in')
        res1 = residual_block_preELU_2convs_pos_dep_bias(conv_in, 3, NK, N, 'res1')
        res2 = residual_block_preELU_2convs_pos_dep_bias(res1, 3, NK, N, 'res2')
        res3 = residual_block_preELU_2convs_pos_dep_bias(res2, 3, NK, N, 'res3')
        res4 = residual_block_preELU_2convs_pos_dep_bias(res3, 3, NK, N, 'res4')
        res5 = residual_block_preELU_2convs_pos_dep_bias(res4, 3, NK, N, 'res5')
        res6 = residual_block_preELU_2convs_pos_dep_bias(res5, 3, NK, N, 'res6')
        res7 = residual_block_preELU_2convs_pos_dep_bias(res6, 3, NK, N, 'res7')
        res8 = residual_block_preELU_2convs_pos_dep_bias(res7, 3, NK, N, 'res8')
        res9 = residual_block_preELU_2convs_pos_dep_bias(res8, 3, NK, N, 'res9')
        res10 = residual_block_preELU_2convs_pos_dep_bias(res9, 3, NK, N, 'res10')
        conv_out = conv_pos_dep_bias(res10, 3, NK, 1, N, 'conv_out')
        logits = tf.reshape(conv_out, [-1, N*N])        
        return logits

class FirstMoveTest: 
    def __init__(self, N, Nfeat):
        self.train_dir = "/home/greg/coding/ML/go/NN/work/train_dirs/first_move_test"
        self.N = N
        self.Nfeat = Nfeat
    def inference(self, feature_planes, N, Nfeat):
        bias = tf.Variable(tf.constant(0.0, shape=[self.N, self.N, 1]), name='single_bias')
        logits = tf.reshape(bias, [-1, N*N])
        return logits




