import argparse
import os
import sys

from utils.load_data_sets import DataSet,parse_data_sets


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

    test_chunk, training_chunks = parse_data_sets(*data_sets)
    print("Allocating %s positions as test; remainder as training" % len(test_chunk), file=sys.stderr)

    print("Writing test chunk")
    test_dataset = DataSet.from_positions_w_context(test_chunk, is_test=True)
    test_filename = os.path.join(processed_dir, "test.chunk.gz")
    test_dataset.write(test_filename)

    training_datasets = map(DataSet.from_positions_w_context, training_chunks)
    for i, train_dataset in enumerate(training_datasets):
        if i % 10 == 0:
            print("Writing training chunk %s" % i)
        train_filename = os.path.join(processed_dir, "train%s.chunk.gz" % i)
        train_dataset.write(train_filename)
    print("%s chunks written" % (i+1))            

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Define parameters.')

    parser.add_argument('--dataset', dest='data_sets')

    args = parser.parse_args()
    
    preprocess(args.data_sets)
