import tensorflow as tf
import numpy as np
import random
import os
from Engine import *
import Book
import Features
import Normalization
import Symmetry
import Checkpoint
from GTP import Move, true_stderr
from Board import *

def softmax(E, temp):
    #print "E =\n", E
    expE = np.exp(temp * (E - max(E))) # subtract max to avoid overflow
    return expE / np.sum(expE)

def sample_from(probs):
    cumsum = np.cumsum(probs)
    r = random.random()
    for i in xrange(len(probs)):
        if r <= cumsum[i]: 
            return i
    assert False, "problem with sample_from" 


class TFEngine(BaseEngine):
    def __init__(self, eng_name, model):
        super(TFEngine,self).__init__() 
        self.eng_name = eng_name
        self.model = model
        self.book = Book.load_GoGoD_book()

        self.last_move_probs = np.zeros((self.model.N, self.model.N,))
        self.kibitz_mode = False

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


    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def set_board_size(self, N):
        if N != self.model.N:
            return False
        return BaseEngine.set_board_size(self, N)

    def pick_book_move(self, color):
        if self.book:
            book_move = Book.get_book_move(self.board, self.book)
            if book_move:
                print "playing book move", book_move
                return Move(book_move[0], book_move[1])
            print "no book move"
        else:
            print "no book"
        return None

    def pick_model_move(self, color):
        if self.model.Nfeat == 15:
            board_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(self.board, color)
            Normalization.apply_featurewise_normalization_B(board_feature_planes)
        elif self.model.Nfeat == 21:
            board_feature_planes = Features.make_feature_planes_stones_4liberties_4history_ko_4captures(self.board, color).astype(np.float32)
            Normalization.apply_featurewise_normalization_C(board_feature_planes)
        else:
            assert False
        feature_batch = Symmetry.make_symmetry_batch(board_feature_planes)

        feed_dict = {self.feature_planes: feature_batch}

        logit_batch = self.sess.run(self.logits, feed_dict)
        move_logits = Symmetry.average_plane_over_symmetries(logit_batch, self.model.N)
        softmax_temp = 1.0
        move_probs = softmax(move_logits, softmax_temp)

        # zero out illegal moves
        for x in xrange(self.model.N):
            for y in xrange(self.model.N):
                ind = self.model.N * x + y 
                if not self.board.play_is_legal(x, y, color):
                    move_probs[ind] = 0
        sum_probs = np.sum(move_probs)
        if sum_probs == 0: return Move.Pass() # no legal moves, pass
        move_probs /= sum_probs # re-normalize probabilities

        pick_best = True
        if pick_best:
            move_ind = np.argmax(move_probs)
        else:
            move_ind = sample_from(move_probs)
        move_x = move_ind / self.model.N
        move_y = move_ind % self.model.N

        self.last_move_probs = move_probs.reshape((self.board.N, self.board.N))

        return Move(move_x, move_y)

    def pick_move(self, color):
        book_move = self.pick_book_move(color)
        if book_move:
            if self.kibitz_mode: # in kibitz mode compute model probabilities anyway
                self.pick_model_move(color) # ignore the model move
            return book_move
        return self.pick_model_move(color)

    def get_last_move_probs(self):
        return self.last_move_probs

    def stone_played(self, x, y, color):
        # if we are in kibitz mode, we want to compute model probabilities for ALL turns
        if self.kibitz_mode:
            self.pick_model_move(color)
            true_stderr.write("probability of played move %s (%d, %d) was %.2f%%\n" % (color_names[color], x, y, 100*self.last_move_probs[x,y]))

        BaseEngine.stone_played(self, x, y, color)

    def toggle_kibitz_mode(self):
        self.kibitz_mode = ~self.kibitz_mode
        return self.kibitz_mode




