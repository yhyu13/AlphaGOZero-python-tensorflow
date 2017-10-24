import tensorflow as tf
import numpy as np
import random
import Symmetry

def apply_random_symmetries(many_feature_planes, many_final_maps):
    for i in range(many_feature_planes.shape[0]):
        s = random.randint(0, 7)
        Symmetry.apply_symmetry_planes(many_feature_planes[i,:,:,:], s)
        Symmetry.apply_symmetry_plane(many_final_maps[i,:,:], s)

def build_feed_dict(loader, apply_normalization, feature_planes, final_maps):
    batch = loader.next_minibatch(('feature_planes', 'final_maps'))
    loaded_feature_planes = batch['feature_planes'].astype(np.float32)
    loaded_final_maps = batch['final_maps'].astype(np.float32)

    apply_normalization(loaded_feature_planes)

    apply_random_symmetries(loaded_feature_planes, loaded_final_maps)

    minibatch_size = loaded_feature_planes.shape[0]
    N = loaded_feature_planes.shape[1]
    return { feature_planes: loaded_feature_planes,
             final_maps: loaded_final_maps.reshape((minibatch_size, N*N)) }

def loss_func(logits):
    final_maps = tf.placeholder(tf.float32, shape=[None, 361])

    # final maps are originally -1 to 1. rescale them to 0 to 1 probabilities:
    final_prob_maps = final_maps * tf.constant(0.5) + tf.constant(0.5)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=final_prob_maps)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    correct_prediction = tf.equal(tf.sign(logits), tf.sign(final_maps))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return final_maps, cross_entropy_mean, accuracy
