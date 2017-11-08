#!/bin/bash

train(){
    python main.py --mode=train --force_save
}

until train; do
    echo "'train' crashed with exit code $?. Restarting..." >&2
    sleep 1
done
