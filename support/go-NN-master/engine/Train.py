#!/usr/bin/python

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import os
import random
import time
from datetime import datetime
import gc
import MoveModels
import MoveTraining
import InfluenceModels
import InfluenceTraining
import EvalModels
import EvalTraining
import NPZ
import Normalization
import Checkpoint

def train_step(total_loss, learning_rate, momentum=None):
    return tf.train.MomentumOptimizer(learning_rate, momentum).minimize(total_loss)

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def read_float_from_file(filename, default):
    try: 
        with open(filename, 'r') as f:
            x = float(f.read().strip())
            return x
    except:
        print "failed to read from file", filename, "; using default value", default
        return default

def append_line_to_file(filename, s):
    with open(filename, 'a') as f:
        f.write(s)
        f.write('\n')

class MovingAverage:
    def __init__(self, name, time_constant):
        self.name = name
        self.time_constant = time_constant
        self.num_samples = 0
        self.avg = 0.0
        self.last_sample = 0
    def add(self, sample):
        sample = float(sample)
        self.num_samples += 1
        weight = 1.0 / min(self.num_samples, self.time_constant)
        self.avg = weight * sample + (1 - weight) * self.avg
        self.last_sample = sample
    def write(self, summary_writer, step):
        summary_writer.add_summary(make_summary(self.name+' (avg)', self.avg), step)
        summary_writer.add_summary(make_summary(self.name+' (raw)', self.last_sample), step)


"""
def async_batch_worker(batch_queue, loader, build_feed_dict, apply_normalization):
    while True:
        feed_dict = build_feed_dict(loader, apply_normalization)
        batch_queue.put(feed_dict, block=True) # will block if queue is full

class BatchQueue:
    def __init__(loader, build_feed_dict, apply_normalization, placeholder_names):
        self.batch_queue = multiprocessing.Queue(maxsize=5)
        self.process = multiprocessing.Process(target=async_batch_worker, args=(self.batch_queue, loader, build_feed_dict, apply_normalization))
        self.process.daemon = True
        self.process.start()
    def pop_feed_dict():
        str_feed_dict = self.batch_queue.get(block=True, timeout=5)
        # new dict with TF placeholder variables as keys instead of strings:
        op_feed_dict = dict((ph, str_feed_dict[placeholder_names[ph]]) for ph in placeholder_names)
        return op_feed_dict
"""


def train_model(model, N, Nfeat, build_feed_dict, normalization, loss_func, train_data_dir, val_data_dir, lr_base, lr_half_life, max_steps, just_validate=False):
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        momentum_ph = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N, N, Nfeat])

        model_outputs = model.inference(feature_planes, N, Nfeat)
        outputs_ph, total_loss, accuracy = loss_func(model_outputs)
        train_op = train_step(total_loss, learning_rate_ph, momentum_ph)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2.0)

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(os.path.join(model.train_dir, 'summaries', datetime.now().strftime('%Y%m%d-%H%M%S')), graph=sess.graph, flush_secs=5)
        accuracy_avg = MovingAverage('accuracy', time_constant=1000)
        total_loss_avg = MovingAverage('total_loss', time_constant=1000)

        def run_validation(): # run the validation set
            val_loader = NPZ.Loader(val_data_dir)
            mean_loss = 0.0
            mean_accuracy = 0.0
            mb_num = 0
            print "Starting validation..."
            while val_loader.has_more():
                if mb_num % 100 == 0: print "validation minibatch #%d" % mb_num
                feed_dict = build_feed_dict(val_loader, normalization, feature_planes, outputs_ph)
                loss_value, accuracy_value = sess.run([total_loss, accuracy], feed_dict=feed_dict)
                mean_loss += loss_value
                mean_accuracy += accuracy_value
                mb_num += 1
            mean_loss /= mb_num
            mean_accuracy /= mb_num
            print "Validation: mean loss = %.3f, mean accuracy = %.2f%%" % (mean_loss, 100*mean_accuracy)
            summary_writer.add_summary(make_summary('validation_loss', mean_loss), step)
            summary_writer.add_summary(make_summary('validation_accuracy_percent', 100*mean_accuracy), step)
    
        last_training_loss = None

        if just_validate: # Just run the validation set once
            Checkpoint.restore_from_checkpoint(sess, saver, model.train_dir)
            run_validation()
        else: # Run the training loop
            #step = 0
            step = Checkpoint.optionally_restore_from_checkpoint(sess, saver, os.path.join(model.train_dir, 'checkpoints'))
            #step = optionally_restore_from_checkpoint(sess, saver, model.train_dir)
            #print "WARNING: CHECKPOINTS TURNED OFF!!"
            print "WARNING: WILL STOP AFTER %d STEPS" % max_steps
            print "WARNING: IGNORING lr.txt and momentum.txt"
            print "lr_base = %f, lr_half_life = %f" % (lr_base, lr_half_life)
            #loader = NPZ.AsyncRandomizingLoader(train_data_dir, minibatch_size=128)
            minibatch_size = 128
            batch_queue = EvalTraining.AsyncRandomBatchQueue(feature_planes, outputs_ph, train_data_dir, minibatch_size, normalization)
            #loader = NPZ.RandomizingLoader(train_data_dir, minibatch_size=128)
            #loader = NPZ.GroupingRandomizingLoader(train_data_dir, Ngroup=1)
            #loader = NPZ.SplittingRandomizingLoader(train_data_dir, Nsplit=2)
            last_step_ref_time = 0
            while True:
                if step % 10000 == 0 and step != 0: 
                    run_validation()

                start_time = time.time()
                #feed_dict = build_feed_dict(loader, normalization, feature_planes, outputs_ph)
                feed_dict = batch_queue.next_feed_dict()
                load_time = time.time() - start_time

                if step % 1 == 0:
                    #learning_rate = read_float_from_file('../work/lr.txt', default=0.1)
                    #momentum = read_float_from_file('../work/momentum.txt', default=0.9)
                    if step < 100:
                        learning_rate = 0.0003 # to stabilize initially
                    else:
                        learning_rate = lr_base * 0.5**(float(step-100)/lr_half_life)
                    momentum = 0.9
                    summary_writer.add_summary(make_summary('learningrate', learning_rate), step)
                    summary_writer.add_summary(make_summary('momentum', momentum), step)
                feed_dict[learning_rate_ph] = learning_rate
                feed_dict[momentum_ph] = momentum
    
                start_time = time.time()
                _, loss_value, accuracy_value, outputs_value = sess.run([train_op, total_loss, accuracy, model_outputs], feed_dict=feed_dict)
                train_time = time.time() - start_time

                total_loss_avg.add(loss_value)
                accuracy_avg.add(100 * accuracy_value)
                #print "outputs_value ="
                #print outputs_value.flatten()
                #print "feed_dict[outputs_ph] ="
                #print feed_dict[outputs_ph].flatten()

                if np.isnan(loss_value):
                    print "Model diverged with loss = Nan"
                    return
                #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step >= max_steps: 
                    return

                if step % 10 == 0:
                    total_loss_avg.write(summary_writer, step)
                    accuracy_avg.write(summary_writer, step)

                full_step_time = time.time() - last_step_ref_time
                last_step_ref_time = time.time()

                if step % 1 == 0:
                    minibatch_size = feed_dict[feature_planes].shape[0]
                    examples_per_sec = minibatch_size / full_step_time
                    print "%s: step %d, lr=%.6f, mom=%.2f, loss = %.4f, accuracy = %.2f%% (mb_size=%d, %.1f examples/sec), (load=%.3f train=%.3f total=%0.3f sec/step)" % \
                            (datetime.now(), step, learning_rate, momentum, loss_value, 100*accuracy_value, minibatch_size, examples_per_sec, load_time, train_time, full_step_time)
                    if step % 10 == 0:
                        summary_writer.add_summary(make_summary('examples/sec', examples_per_sec), step)
                        summary_writer.add_summary(make_summary('step', step), step)
    
                if step % 1000 == 0 and step != 0:
                    #print "WARNING: CHECKPOINTS TURNED OFF!!"
                    saver.save(sess, os.path.join(model.train_dir, "checkpoints", "model.ckpt"), global_step=step)

                step += 1



if __name__ == "__main__":
    N = 19
    #Nfeat = 15
    #Nfeat = 21
    Nfeat = 22
    
    """
    #model = Models.Conv6PosDep(N, Nfeat) 
    #model = Models.Conv8PosDep(N, Nfeat) 
    #model = Models.Conv10PosDep(N, Nfeat) 
    #model = MoveModels.Conv10PosDepELU(N, Nfeat) 
    #model = MoveModels.Conv12PosDepELU(N, Nfeat) 
    model = MoveModels.Conv12PosDepELUBig(N, Nfeat) 
    #model = MoveModels.Conv16PosDepELU(N, Nfeat) 
    #model = MoveModels.Res5x2PreELU(N, Nfeat) 
    #model = MoveModels.Res10x2PreELU(N, Nfeat) 
    #model = MoveModels.Conv4PosDepELU(N, Nfeat) 
    #model = Models.FirstMoveTest(N, Nfeat) 
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/train-rand-2"
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/val-small"
    #normalization = Normalization.apply_featurewise_normalization_B
    train_data_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/val-small"
    normalization = Normalization.apply_featurewise_normalization_C
    build_feed_dict = MoveTraining.build_feed_dict
    loss_func = MoveTraining.loss_func
    """

    #model = EvalModels.Conv5PosDepFC1ELU(N, Nfeat)
    model = EvalModels.Conv11PosDepFC1ELU(N, Nfeat)
    #model = EvalModels.Zero(N, Nfeat)
    #model = EvalModels.Linear(N, Nfeat)
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_Nf21/train"
    train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_komi_Nf22/train"
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_Nf21/val-small"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_komi_Nf22/val-small"
    #normalization = Normalization.apply_featurewise_normalization_C
    normalization = Normalization.apply_featurewise_normalization_D
    build_feed_dict = EvalTraining.build_feed_dict
    loss_func = EvalTraining.loss_func

    """
    #model = InfluenceModels.Conv4PosDep(N, Nfeat)
    model = InfluenceModels.Conv12PosDepELU(N, Nfeat)
    train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/examples/stones_3lib_4hist_ko_Nf15/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/examples/stones_3lib_4hist_ko_Nf15/val"
    build_feed_dict = InfluenceTraining.build_feed_dict
    loss_func = InfluenceTraining.loss_func
    normalization = Normalization.apply_featurewise_normalization_B
    """

    #gc.set_debug(gc.DEBUG_STATS)

    print "Training data = %s\nValidation data = %s" % (train_data_dir, val_data_dir)

    #for lr_half_life in [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]:
    #for lr_half_life in [1e4, 3e4, 1e5, 3e5, 1e6]:
    #    max_steps = lr_half_life * 7
    #    #for lr_base in [0.01, 0.003, 0.001, 0.0003]:
    #    #lr_base = 0.008
    #    lr_base = 0.002 # seems to be the highest useful learning rate for eval_conv11fc1
    lr_base = 0.002
    lr_half_life = 1e5 #3e4
    max_steps = 1e9
    train_model(model, N, Nfeat, build_feed_dict, normalization, loss_func, train_data_dir, val_data_dir, lr_base, lr_half_life, max_steps, just_validate=False)

