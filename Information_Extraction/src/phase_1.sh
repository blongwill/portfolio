#!/bin/sh

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project

#sh file used to call all scripts that were involved in the first phase:\
#   text processing, candidate sentence for training, featurization for training data

./src/pre-process_names.sh
./src/create_training_data.sh
./src/create_testing_data.sh
./src/create_language_model.py ./rec/training_vecs/trimmed_class_training.vectors.txt ./rec/language_models/class_language_model.txt
./src/create_language_model.py ./rec/training_vecs/binary_training.vectors.txt ./rec/language_models/binary_language_model.txt
./src/extract_semantic_relations.py ./rec/testing_vecs/binary_test_sentences.txt ./output/test_semantic_relations.txt False



