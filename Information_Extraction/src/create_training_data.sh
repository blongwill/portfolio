#!/bin/sh

./src/extract_wiki_data.py ./rec/name_lists/training_names.txt ./rec/training_vecs/class_training.vectors.txt ./rec/training_vecs/binary_training.vectors.txt 700 True

#### These really only need trimmed if there are a lot more positive instances than negative instances because the classifier should choose absent if all words are unkonwn
#./src/trim_data.py ./rec/training_vecs/binary_training.vectors.txt ./rec/training_vecs/trimmed_binary_training.vectors.txt 6000

./src/trim_data.py ./rec/training_vecs/class_training.vectors.txt ./rec/training_vecs/trimmed_class_training.vectors.txt 680
 

