#!/bin/sh

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project

#Sh file used to call method that extracts semantic relations from sentences\
#ses a binary Naive Bayes classifier to first detect if a relation contained any form of the target relations
#Relations classified as having one of the target relations were input to a second multi-class classifier that returns the class label

./src/run_semantic_extraction.py ./rec/testing_vecs/binary_test_sentences.txt ./rec/testing_vecs/test_semantic_relations.txt ./rec/language_models/binary_language_model.txt ./output/binary_sys_output.txt ./output/binary_acc.txt False ./rec/language_models/class_language_model.txt ./output/class_sys_output.txt ./output/class_acc.txt False

