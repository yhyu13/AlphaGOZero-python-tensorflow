#!/bin/bash

if [ "$#" -ne 1 ] ; then
  echo "Must specify a directory."
  exit
fi

dir=$1
cd "$dir"

echo "Going to divide $dir into training, validation, and test sets."

tmp_list=/tmp/npz_list.txt

ls | grep '.npz' | shuf > "$tmp_list"

num_mb=$(wc -l < "$tmp_list")
num_val=$(($num_mb / 10))
num_test=$(($num_mb / 10))
num_train=$(($num_mb - $num_val - $num_test))

echo "Total number of minibatches is $num_mb"
echo "$num_train to training set."
echo "$num_val to validation set."
echo "$num_test to test set."

mkdir -p train
mkdir -p val
mkdir -p test

head "-$num_val" "$tmp_list" | while read fn ; do
  mv "$fn" val
done

tail "-$num_test" "$tmp_list" | while read fn ; do
  mv "$fn" test
done

mv *.npz train

echo "Done."
