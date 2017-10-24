import tensorflow as tf
import numpy as np
import os.path
import Checkpoint
import Features
import Normalization

class AllPolicy:
    def suggest_moves(self, board):
        moves = []
        for x in xrange(board.N):
            for y in xrange(board.N):
                if board.play_is_legal(x, y, board.color_to_play):
                    moves.append((x,y))
        return moves

def softmax(E, temp):
    #print "E =\n", E
    expE = np.exp(temp * (E - max(E))) # subtract max to avoid overflow
    return expE / np.sum(expE)

class TFPolicy:
    def __init__(self, model, threshold_prob, softmax_temp):
        self.model = model
        self.threshold_prob = threshold_prob
        self.softmax_temp = softmax_temp

        # build the graph
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

    def suggest_moves(self, board):
        board_feature_planes = Features.make_feature_planes_stones_4liberties_4history_ko_4captures(board, board.color_to_play).astype(np.float32)
        Normalization.apply_featurewise_normalization_C(board_feature_planes)
        feed_dict = {self.feature_planes: board_feature_planes.reshape(1,self.model.N,self.model.N,self.model.Nfeat)}
        move_logits = self.sess.run(self.logits, feed_dict).ravel() # ravel flattens to 1D
        # zero out illegal moves
        for x in xrange(self.model.N):
            for y in xrange(self.model.N):
                ind = self.model.N * x + y 
                if not board.play_is_legal(x, y, board.color_to_play):
                    move_logits[ind] = -1e99
        move_probs = softmax(move_logits, self.softmax_temp)
        sum_probs = np.sum(move_probs)
        if sum_probs == 0: return [] # no legal moves
        move_probs /= sum_probs # re-normalize probabilities

        good_moves = []
        cum_prob = 0.0
        while cum_prob < self.threshold_prob:
            ind = np.argmax(move_probs)
            x,y = ind / self.model.N, ind % self.model.N
            good_moves.append((x,y))
            prob = move_probs[ind]
            cum_prob += prob
            move_probs[ind] = 0

        return good_moves



