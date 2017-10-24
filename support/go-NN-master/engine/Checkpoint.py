import tensorflow as tf

def restore_from_checkpoint(sess, saver, ckpt_dir):
    print "Trying to restore from checkpoint in dir", ckpt_dir
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print "Checkpoint file is ", ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print "Restored from checkpoint %s" % global_step
        return global_step
    else:
        print "No checkpoint file found"
        assert False

def optionally_restore_from_checkpoint(sess, saver, train_dir):
    while True:
        response = raw_input("Restore from checkpoint [y/n]? ").lower()
        if response == 'y': 
            return restore_from_checkpoint(sess, saver, train_dir)
        if response == 'n':
            return 0

