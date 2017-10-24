#!/usr/bin/python
import tensorflow as tf
import numpy as np
import os
from Engine import *
from Board import *
import Features
import Normalization
import Symmetry
import Checkpoint

def average_probs_over_symmetries(probs):
    assert probs.size == 8
    return probs.mean()

class EvalEngine(BaseEngine):
    def name(self):
        return "EvalEngine"

    def version(self):
        return "1.0"

    def __init__(self, model):
        BaseEngine.__init__(self) 
        self.model = model
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.probs_op = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
                Checkpoint.restore_from_checkpoint(self.sess, saver, checkpoint_dir)

    def get_position_eval(self):
        #assert self.model.Nfeat == 21
        #board_feature_planes = Features.make_feature_planes_stones_4liberties_4history_ko_4captures(self.board, self.board.color_to_play).astype(np.float32)
        #Normalization.apply_featurewise_normalization_C(board_feature_planes)
        assert self.model.Nfeat == 22
        board_feature_planes = Features.make_feature_planes_stones_4liberties_4history_ko_4captures_komi(self.board, self.board.color_to_play, self.komi).astype(np.float32)
        Normalization.apply_featurewise_normalization_D(board_feature_planes)
        feature_batch = Symmetry.make_symmetry_batch(board_feature_planes)
        feed_dict = {self.feature_planes: feature_batch}
        probs_batch = self.sess.run(self.probs_op, feed_dict)
        prob = average_probs_over_symmetries(probs_batch)
        if self.board.color_to_play == Color.White:
            prob *= -1
        return prob

    def pick_move(self, color):
        for i in xrange(10000):
            x = np.random.randint(0, self.board.N-1)
            y = np.random.randint(0, self.board.N-1)
            if self.board.play_is_legal(x, y, color):
                return Move(x,y)
        return Move.Pass


if __name__ == '__main__':
    import GTP
    fclient = GTP.redirect_all_output("log_engine.txt")

    import EvalModels
    
    engine = EvalEngine(EvalModels.Conv11PosDepFC1ELU(N=19, Nfeat=22))
    
    gtp = GTP.GTP(engine, fclient)
    gtp.loop()
