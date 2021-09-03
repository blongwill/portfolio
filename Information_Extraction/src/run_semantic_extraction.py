#!/opt/python-3.6/bin/python3
# -*- coding: utf-8 -*-

import sys
from extract_semantic_relations import extract_to_file
from run_classifier import naive_bayes_classifier

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project
# Script for extracting semantic relations from

def main():

    #### Gets sentence file ########################
    test_sentence_file=sys.argv[1]


    #### Run extractor ########################
    semantic_relation_output_file=sys.argv[2]

    extract_to_file(test_sentence_file, semantic_relation_output_file, str(False))

    semantic_relation_output_lines = open(semantic_relation_output_file).readlines()



    #### Instantiate and Run binary classifier ########################
    binary_language_model_file=sys.argv[3]
    binary_sys_output_file=sys.argv[4]
    binary_acc_output_file=sys.argv[5]
    binary_classifier_input_is_labeled=bool(eval(sys.argv[6]))

    nbc = naive_bayes_classifier(binary_language_model_file)

    nbc.classify(test_vectors=semantic_relation_output_lines, data_is_labeled=binary_classifier_input_is_labeled, sys_file_path=binary_sys_output_file, acc_file_path=binary_acc_output_file)


    semantic_relations_present=[semantic_relation_output_lines[index] for index,line in enumerate(open(binary_sys_output_file).readlines()) if (binary_classifier_input_is_labeled==True and line.split()[2]=="present") or (binary_classifier_input_is_labeled==False and line.split()[1]=="present")]


    ##### Write out to file


    #### Run relation classifier ########################

    class_language_model_file = sys.argv[7]
    class_sys_output_file = sys.argv[8]
    class_acc_output_file = sys.argv[9]
    class_classifier_input_is_labeled = bool(eval(sys.argv[10]))

    nbc = naive_bayes_classifier(class_language_model_file)


    nbc.classify(test_vectors=semantic_relations_present, data_is_labeled=class_classifier_input_is_labeled, sys_file_path=class_sys_output_file,acc_file_path=class_acc_output_file)


