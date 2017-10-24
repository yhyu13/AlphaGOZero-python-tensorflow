import tensorflow as tf
import numpy as np
import time
import multiprocessing
import random
import Symmetry
import NPZ
import gc

def apply_random_symmetries(many_feature_planes):
    N = many_feature_planes.shape[1]
    for i in range(many_feature_planes.shape[0]):
        s = random.randint(0, 7)
        #Symmetry.apply_symmetry_planes(many_feature_planes[i,:,:,:], s)
        Symmetry.apply_symmetry_features_example(many_feature_planes, i, s)


"""
def build_feed_dict(loader, apply_normalization, feature_planes, final_scores):
    a = time.time()
    batch = loader.next_minibatch(('feature_planes', 'final_scores'))
    b = time.time()
    loaded_feature_planes = batch['feature_planes'].astype(np.float32)
    loaded_scores = batch['final_scores'].astype(np.float32) # BIT ME HARD.
    c = time.time()

    loaded_scores = np.ravel(loaded_scores) # flatten to 1D
    d = time.time()

    apply_normalization(loaded_feature_planes)
    e = time.time()

    #print "WARNING: NOT APPLYING SYMMETRIES!!!!!!!!!!!!!!!!"
    apply_random_symmetries(loaded_feature_planes)
    f = time.time()

    print "b-a = %f, c-b = %f, d-c = %f, e-d = %f, f-e = %f" % ((b-a,c-b,d-c,e-d,f-e))

    #N = loaded_feature_planes.shape[1]

    #print "loaded_feature_planes ="
    #print loaded_feature_planes
    #print "loaded_scores ="
    #print loaded_scores

    return { feature_planes: loaded_feature_planes,
             final_scores: loaded_scores }
"""

def build_feed_dict_strings(loader, apply_normalization):
    a = time.time()
    batch = loader.next_minibatch(('feature_planes', 'final_scores'))
    b = time.time()
    loaded_feature_planes = batch['feature_planes'].astype(np.float32)
    loaded_scores = batch['final_scores'].astype(np.float32) # BIT ME HARD.
    loaded_scores = np.ravel(loaded_scores) # flatten to 1D
    c = time.time()
    apply_normalization(loaded_feature_planes)
    d = time.time()
    apply_random_symmetries(loaded_feature_planes)
    e = time.time()
    print "b-a=%f, c-b=%f, d-c=%f, e-d=%f" % (b-a, c-b, d-c, e-d)
    return { 'feature_planes': loaded_feature_planes,
             'final_scores': loaded_scores }

def dict_strings_to_ops(feed_dict_strings, feature_planes_ph, final_scores_ph):
    return { feature_planes_ph: feed_dict_strings['feature_planes'],
             final_scores_ph: feed_dict_strings['final_scores'] }

def build_feed_dict(loader, apply_normalization, feature_planes_ph, final_scores_ph):
    return dict_strings_to_ops(build_feed_dict_strings(loader, apply_normalization), feature_planes_ph, final_scores_ph)

def async_worker(q, npz_dir, minibatch_size, apply_normalization):
    print "Hello from EvalTraining async_worker process!!!"
    gc.set_debug(gc.DEBUG_STATS)
    loader = NPZ.RandomizingLoader(npz_dir, minibatch_size)
    names = ('feature_planes', 'final_scores')
    while True:
        feed_dict_strings = build_feed_dict_strings(loader, apply_normalization)
        q.put(feed_dict_strings, block=True) # will block if queue is full

class AsyncRandomBatchQueue:
    def __init__(self, feature_planes_ph, final_scores_ph, npz_dir, minibatch_size, apply_normalization):
        self.q = multiprocessing.Queue(maxsize=5)
        self.process = multiprocessing.Process(target=async_worker, args=(self.q, npz_dir, minibatch_size, apply_normalization))
        self.process.daemon = True
        self.process.start()
        self.feature_planes_ph = feature_planes_ph
        self.final_scores_ph = final_scores_ph
    def next_feed_dict(self):
        feed_dict_strings = self.q.get(block=True, timeout=30)
        feed_dict = dict_strings_to_ops(feed_dict_strings, self.feature_planes_ph, self.final_scores_ph)
        return feed_dict


def loss_func(score_op):
    final_scores = tf.placeholder(tf.float32, shape=[None])

    squared_errors = tf.square(tf.reshape(score_op, [-1]) - final_scores)
    #mean_sq_err = tf.reduce_mean(squared_errors, name='mean_sq_err')
    cross_entropy_ish_loss = tf.reduce_mean(-tf.log(tf.constant(1.0) - tf.constant(0.5) * tf.abs(tf.reshape(score_op, [-1]) - final_scores), name='cross-entropy-ish-loss'))

    correct_prediction = tf.equal(tf.sign(tf.reshape(score_op, [-1])), tf.sign(final_scores))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    #return final_scores, mean_sq_err, accuracy, squared_errors
    return final_scores, cross_entropy_ish_loss, accuracy



