#!/bin/bash

outfile=corrupt_sgfs.txt

base_dir=/home/greg/coding/ML/go/NN/data/GoGoD/GoGoDWinter2015/Database

ls "$base_dir" | while read sub_dir ; do
    ls "$base_dir/$sub_dir" | while read sgf ; do
        #if grep -H --color=auto 'corrupt\|Corrupt\|illegal\|Illegal' "$base_dir/$sub_dir/$sgf" ; then 
        if grep -H --color=auto 'corrupt\|Corrupt' "$base_dir/$sub_dir/$sgf" ; then 
            echo "$sgf" >> "$outfile"
        fi
    done
done

