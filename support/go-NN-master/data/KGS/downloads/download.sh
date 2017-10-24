#!/bin/bash

mkdir downloads
cd downloads

# Get a list of SGF archive file URLs
wget --output-document=- http://u-go.net/gamerecords/ | grep Download | grep bz2 | sed 's/.*"\(.*\)".*/\1/' > urls.txt

while read url ; do
    echo "fetching $url"
    wget "$url"    
done < urls.txt

echo "Done fetching archive files. Extracting..."

cd ..
mkdir SGFs 
cd SGFs

for archive in ../downloads/*.tar.bz2 ; do
    tar xjf "$archive"
done

echo "Done!"

