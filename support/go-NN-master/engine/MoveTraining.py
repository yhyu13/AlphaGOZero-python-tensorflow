import tensorflow as tf
import numpy as np
import random
import Symmetry

def apply_random_symmetries(many_feature_planes, many_move_arrs):
    N = many_feature_planes.shape[1]
    for i in range(many_feature_planes.shape[0]):
        s = random.randint(0, 7)
        Symmetry.apply_symmetry_planes(many_feature_planes[i,:,:,:], s)
        Symmetry.apply_symmetry_vertex(many_move_arrs[i,:], N, s)


def build_feed_dict(loader, apply_normalization, feature_planes, move_indices):
    batch = loader.next_minibatch(('feature_planes', 'moves')  )
    loaded_feature_planes = batch['feature_planes'].astype(np.float32)
    loaded_move_arrs = batch['moves'].astype(np.int32) # BIT ME HARD.

    apply_normalization(loaded_feature_planes)

    apply_random_symmetries(loaded_feature_planes, loaded_move_arrs)

    N = loaded_feature_planes.shape[1]
    loaded_move_indices = N * loaded_move_arrs[:,0] + loaded_move_arrs[:,1] 

    return { feature_planes: loaded_feature_planes.astype(np.float32),
             move_indices: loaded_move_indices }

def loss_func(logits):
    move_indices = tf.placeholder(tf.int64, shape=[None])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, move_indices)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    correct_prediction = tf.equal(tf.argmax(logits,1), move_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return move_indices, cross_entropy_mean, accuracy



