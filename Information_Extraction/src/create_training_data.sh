#!/bin/sh

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project

#ssh file that calls all the methods necessary with the hardcode file paths needed to create training data for wiki extractor
./src/extract_wiki_data.py ./rec/name_lists/training_names.txt ./rec/training_vecs/class_training.vectors.txt ./rec/training_vecs/binary_training.vectors.txt 700 True
./src/trim_data.py ./rec/training_vecs/class_training.vectors.txt ./rec/training_vecs/trimmed_class_training.vectors.txt 680
 

