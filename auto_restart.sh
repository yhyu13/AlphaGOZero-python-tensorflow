#!/bin/bash

train(){
    source activate py3dl
    python main.py --mode=selfplay
}

until train; do
    echo "'train' crashed with exit code $?. Restarting..." >&2
    sleep 1
done
