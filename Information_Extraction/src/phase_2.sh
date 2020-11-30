#!/bin/sh

./src/run_semantic_extraction.py ./rec/testing_vecs/binary_test_sentences.txt ./rec/testing_vecs/test_semantic_relations.txt ./rec/language_models/binary_language_model.txt ./output/binary_sys_output.txt ./output/binary_acc.txt False ./rec/language_models/class_language_model.txt ./output/class_sys_output.txt ./output/class_acc.txt False

