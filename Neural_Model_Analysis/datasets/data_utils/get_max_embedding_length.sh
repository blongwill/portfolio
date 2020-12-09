#!/bin/sh
#Arg1 is path to dataset 

### No arguments needed, script collects dataset paths from config files directly

#paths=$(cat ~/ling575_neural_models/configs/*txt |grep dataset |cut -d" " -f2 | awk '$0="~/ling575_neural_models/datasets/"$0".txt"' | sed -e 's/\r//g')
paths=$(cat ~/ling575_neural_models/configs/*txt |grep dataset |cut -d" " -f2 | tr -d '\r' | awk -v data_dir=$(echo ~/ling575_neural_models/datasets/) '$0=data_dir$0".txt"')
read -a ARRAY <<< "$paths"
./get_max_embedding_length.py  ${paths}
