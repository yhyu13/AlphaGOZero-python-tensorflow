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
def preprocess(*data_sets, processed_dir="processed_data"):
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)
        
    '''
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
        train_filename = os.path.join(processed_dir, "train%s.chunk.gz" % i)
        train_dataset.write(train_filename)
    print("%s chunks written" % (i+1))'''
    
    sgf_files = list(find_sgf_files(*data_sets))
    print("%s sgfs found." % len(sgf_files), file=sys.stderr)
    est_num_positions = len(sgf_files) * 200 # about 200 moves per game
    positions_w_context = itertools.chain(*map(get_positions_from_sgf, sgf_files))
    
    positions_w_context = list(positions_w_context)
    test_size = 10**5
    
    print("Writing test chunk")
    test_dataset = DataSet.from_positions_w_context(positions_w_context[:test_size], is_test=True)
    test_filename = os.path.join(processed_dir, "test.chunk.gz")
    test_dataset.write(test_filename)
    
    print("Writing train chunk")
    test_dataset = DataSet.from_positions_w_context(positions_w_context[test_size:], is_test=False)
    test_filename = os.path.join(processed_dir, "train0.chunk.gz")
    test_dataset.write(test_filename)
    test_dataset = None


if __name__=="__main__":

    p = argh.ArghParser()
    p.add_commands([preprocess])
    p.dispatch()
