#-!./bin/bash

# Search directories under argument 1 for SGF games
# Have gnugo play them out to the end and write them to 
# the directory given as argument 2

if [ "$#" -ne 2 ] ; then
    echo "Must specify a source directory and a destination directory."
    exit
fi

sgf_dir=$1
dest_dir=$2

cd "$sgf_dir"

ls | while read fn ; do
    outfile="$dest_dir/played_out_$fn"
    if [ -e "$outfile" ] ; then
        echo "$outfile already exists, skipping"
    else
        echo "playing out $fn"
        gnugo --infile "$fn" --outfile "$outfile" --score aftermath --capture-all-dead --chinese-rules

        # play out the game a SECOND time. The reason is that the second time
        # gnugo writes out a file consisting solely of handicap-style stone
        # placements, meaning we don't have to compute captures or anything.
        # So parsing the final result will be very fast.
        gnugo --infile "$outfile" --outfile "$outfile" --score aftermath --capture-all-dead --chinese-rules
    fi
done
    
