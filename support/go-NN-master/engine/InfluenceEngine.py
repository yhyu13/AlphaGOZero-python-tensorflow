import tensorflow as tf
import numpy as np
import os
from Engine import *
from Board import *
import Features
import Symmetry
import Checkpoint

class InfluenceEngine(BaseEngine):
    def name(self):
        return "InfluenceEngine"

    def version(self):
        return "1.0"

    def __init__(self, model):
        BaseEngine.__init__(self) 
        self.model = model
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.logits = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
                Checkpoint.restore_from_checkpoint(self.sess, saver, checkpoint_dir)

    def make_influence_map(self):
        if self.model.Nfeat == 15:
            board_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(self.board, self.board.color_to_play)
            assert False, "for some reason I commented out the normalization???"
            #Normalization.apply_featurewise_normalization_B(board_feature_planes)
        else: 
            assert False
        feature_batch = make_symmetry_batch(board_feature_planes)
        feed_dict = {self.feature_planes: feature_batch}
        logit_batch = self.sess.run(self.logits, feed_dict)
        move_logits = Symmetry.average_plane_over_symmetries(logit_batch, self.model.N)
        move_logits = move_logits.reshape((self.model.N, self.model.N))
        influence_map = np.tanh(move_logits)
        if self.board.color_to_play == Color.White:
            influence_map *= -1
        #influence_map = -1 * np.ones((self.model.N, self.model.N), dtype=np.float32)
        return influence_map


    def pick_move(self, color):
        for i in xrange(10000):
            x = np.random.randint(0, self.board.N-1)
            y = np.random.randint(0, self.board.N-1)
            if self.board.play_is_legal(x, y, color):
                return Move(x,y)
        return Move.Pass()


