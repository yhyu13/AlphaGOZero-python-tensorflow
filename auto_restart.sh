#bin/env/bash
TRAIN= "python main.py --mode=train --dataset=processed_data/ --force_save > resutl.txt"
TEST= "lol"
while 1; do
    until echo "${TEST}";do
	echo "Restart training"
	sleep 1
    done
    sleep 1
done
