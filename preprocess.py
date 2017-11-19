import os
import sys
_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

import argh

from utils.load_data_sets import *


# Credit: Brain Lee

# this method doesn't read .sgf recursively from a folder to subfolder
# make sure dataset looks like this:
# --|-folder1/.sgfs
#   |-folder2/.sgfs
#   ...
#   |-folderN/.sgfs
def preprocess(*data_sets, processed_dir="processed_data",one_big_training_chunck=True):
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    test_chunk, training_chunks = parse_data_sets(*data_sets)
    print("Allocating %s positions as test; remainder as training" % len(test_chunk), file=sys.stderr)

    print("Writing test chunk")
    test_dataset = DataSet.from_positions_w_context(test_chunk, is_test=True)
    test_filename = os.path.join(processed_dir, "test.chunk.gz")
    test_dataset.write(test_filename)
    test_dataset = None

    training_datasets = map(DataSet.from_positions_w_context, training_chunks)
    for i, train_dataset in enumerate(training_datasets):
        if i % 10 == 0:
            print("Writing training chunk %s" % i)

        """write all training data into one big chunck"""
        if one_big_training_chunck:
            train_filename = os.path.join(processed_dir, "train0.chunk.gz")
            train_dataset.write(train_filename,firts_time=i==0)
        else:
            train_filename = os.path.join(processed_dir, "train%s.chunk.gz" % i)
            train_dataset.write(train_filename,firts_time=True)

    print("%s chunks written" % (i+1))


def tfrecord(*data_sets, processed_dir="processed_data"):
    import tensorflow as tf
    CHUNK_SIZE = 4096

    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    sgf_files = list(find_sgf_files(*data_sets))
    print("%s sgfs found." % len(sgf_files), file=sys.stderr)
    positions_w_context = itertools.chain(*map(get_positions_from_sgf, sgf_files))

    shuffled_positions = utils.shuffler(positions_w_context)
    training_chunks = utils.iter_chunks(CHUNK_SIZE, shuffled_positions)
    training_datasets = map(DataSet.from_positions_w_context, training_chunks)

    path2file = os.path.join(processed_dir, "train.tfrecords")
    writer = tf.python_io.TFRecordWriter(path2file)
    for i, train_dataset in enumerate(training_datasets):
        if i % 10 == 0:
            print("Writing training chunk %s with size %d" % (i,CHUNK_SIZE))

        example = tf.train.Example(features=tf.train.Features(feature={
            'labels': tf.train.Feature(bytes_list=tf.train.Int64List(value=[train_dataset.next_moves])),
            'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_dataset.pos_features])),
            'results':tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_dataset.results]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

    print("%s chunks written" % (i+1))
    print("%s positions written" % (i+1)*CHUNK_SIZE)


if __name__=="__main__":

    p = argh.ArghParser()
    p.add_commands([preprocess,tfrecord])
    p.dispatch()
