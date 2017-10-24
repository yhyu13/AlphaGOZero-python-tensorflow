#!/bin/bash

# Search directories under argument 1 for conclusive SGF games
# Make symbolic links to all of them in directory 2

if [ "$#" -ne 2 ] ; then
  echo "Must specify a source directory and a destination directory."
  exit
fi

sgf_dir=$1
dest_dir=$2

cd "$dest_dir"

find "$sgf_dir" -name '*.sgf' -print | while read fn ; do
  if grep 'RE\[' "$fn" | egrep -v 'Time|Resign' ; then
      ln -s "$fn"
  fi
done
