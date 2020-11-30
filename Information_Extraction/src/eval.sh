#!/bin/sh


./src/run_classifier.py ./rec/testing_vecs/binary_testing_vecs/annotated_binary_10.txt ./rec/language_models/binary_language_model.txt ./output/binary_sys_output_10.txt ./output/binary_acc_10.txt True
./src/run_classifier.py ./rec/testing_vecs/binary_testing_vecs/annotated_binary_25.txt ./rec/language_models/binary_language_model.txt ./output/binary_sys_output_25.txt ./output/binary_acc_25.txt True
./src/run_classifier.py ./rec/testing_vecs/binary_testing_vecs/annotated_binary_50.txt ./rec/language_models/binary_language_model.txt ./output/binary_sys_output_50.txt ./output/binary_acc_50.txt True

./src/run_classifier.py ./rec/testing_vecs/class_testing_vecs/annotated_class_10.txt ./rec/language_models/class_language_model.txt ./output/class_sys_output_10.txt ./output/class_acc_10.txt True
./src/run_classifier.py ./rec/testing_vecs/class_testing_vecs/annotated_class_25.txt ./rec/language_models/class_language_model.txt ./output/class_sys_output_25.txt ./output/class_acc_25.txt True
./src/run_classifier.py ./rec/testing_vecs/class_testing_vecs/annotated_class_50.txt ./rec/language_models/class_language_model.txt ./output/class_sys_output_50.txt ./output/class_acc_50.txt True

./src/compare_relation_counts.py ./output/openie.txt ./output/sentence_file_count.txt ./output/openie_comparison.txt




