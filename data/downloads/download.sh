#!/bin/bash

# Get a list of SGF archive file URLs
#wget --output-document=- 'http://www.u-go.net/gamerecords-4d' | grep 'Download' | sed #’s/.*”\(.*\)”.*/\1/‘ > urls.txt

while read url ; do
    echo "fetching $url"
    wget "$url"    
done < urls.txt

echo "Done fetching archive files. Extracting..."

cd ..
mkdir SGFs 
cd SGFs

for archive in ../downloads/*.tar.gz ; do
    tar xzf "$archive"
done

echo "Done!"

