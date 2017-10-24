import tensorflow as tf
import numpy as np
import os.path
import Features
import Normalization


def restore_from_checkpoint(sess, saver, ckpt_dir):
    print "Trying to restore from checkpoint in dir", ckpt_dir
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print "Checkpoint file is ", ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print "Restored from checkpoint %s" % global_step
    else:
        print "No checkpoint file found"
        assert False

class TFEval:
    def __init__(self, model):
        self.model = model

        # build the graph
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.score_op = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
                restore_from_checkpoint(self.sess, saver, checkpoint_dir)

    def evaluate(self, board):
        board_feature_planes = Features.make_feature_planes_stones_4liberties_4history_ko_4captures(board, board.color_to_play).astype(np.float32)
        Normalization.apply_featurewise_normalization_C(board_feature_planes)
        feed_dict = {self.feature_planes: board_feature_planes.reshape(1,self.model.N,self.model.N,self.model.Nfeat)}
        score = np.asscalar(self.sess.run(self.score_op, feed_dict))
        return score
