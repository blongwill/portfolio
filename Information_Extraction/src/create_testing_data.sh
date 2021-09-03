#!/bin/sh

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project
#ssh file that calls all the methods with hardcoded file paths to create testing data vectors for later use with extraction

##### Extracts training sentences from wikipedia 
./src/extract_wiki_data.py ./rec/name_lists/testing_names.txt ./rec/testing_vecs/class_testing.vectors.txt ./rec/testing_vecs/binary_testing.vectors.txt 80 False

##### Trim original output  from testing vectors to 80 of each class and sort them

#multi-class vectors
./src/trim_data.py ./rec/testing_vecs/class_testing_vecs/class_testing.vectors.txt ./rec/testing_vecs/class_testing_vecs/trimmed_class_testing.vectors.txt 80

#binary vectors
./src/trim_data.py ./rec/testing_vecs/binary_testing_vecs/binary_testing.vectors.txt ./rec/testing_vecs/binary_testing_vecs/trimmed_binary_testing.vectors.txt 80

#Remove labels and output just test sentences

cut -f 2- -d ' ' ./rec/testing_vecs/class_testing_vecs/trimmed_class_testing.vectors.txt  > ./rec/testing_vecs/class_test_sentences.txt
cut -f 2- -d ' ' ./rec/testing_vecs/binary_testing_vecs/trimmed_binary_testing.vectors.txt  > ./rec/testing_vecs/binary_test_sentences.txt


#Extract relations
./src/extract_semantic_relations.py ./rec/testing_vecs/class_test_sentences.txt ./rec/testing_vecs/class_testing_vecs/class_test_relations.txt False
./src/extract_semantic_relations.py ./rec/testing_vecs/binary_test_sentences.txt ./rec/testing_vecs/binary_testing_vecs/binary_test_relations.txt False


