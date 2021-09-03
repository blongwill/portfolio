#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project

#Python script that creates a language model that displays features as word frequencies in addition to the class priors
#Uses add-delta smoothing

#Script dependencies
from collections import defaultdict
import math
import sys

'''
Params:
    1.) training_vectors_file_path:str - File path to a file that displays training vectors in the form of  class_name feat_i:count_i
    2.) output_file_name:str - File path to output file that contains the final language model
example calls:
#./create_language_model.py input_file output_file
#./create_language_model.py /Users/BenLongwill/Desktop/openio/train.vectors.txt /Users/BenLongwill/Desktop/openio/language_model.txt
#./create_language_model.py /Users/BenLongwill/Desktop/openio/binary_relation.vectors.txt /Users/BenLongwill/Desktop/openio/binary_language_model.txt
'''

training_vectors=open(sys.argv[1]).readlines()
output_file=open(sys.argv[2],'w+')

#Hardcoded delta values used in probability calculations with add delta smoothing
COND_PROB_DELTA=.5
PRIOR_DELTA=1

vector_instance_counts_by_class= defaultdict(lambda:0)
feature_instance_tallies_by_class= defaultdict(lambda:defaultdict(lambda:0))
relation_tally_total=defaultdict(lambda:0)
vocabulary_list=set()
conditional_probabilities_by_class= defaultdict(lambda:defaultdict(lambda:0))
class_priors=defaultdict(lambda:0)

#Iterates file line by line and splits to separate class from word-frequency pairs
#uses dictionary to tally frequencies by class
for index, vector in enumerate(training_vectors):
    relation , word_pairs = vector.split(" ",maxsplit=1)[0], vector.strip().split()[1:]
    vector_instance_counts_by_class[relation]+=1

    for word_pair in word_pairs:
        if word_pair:
            try:
                word,count=word_pair.split(":")
                vocabulary_list.add(word)
                feature_instance_tallies_by_class[relation][word] += int(count)
            except Exception as e:
                print(word_pair)
                print(e)
#### Update unseen words for each class with a count of zero
for relation , feature_set in list(feature_instance_tallies_by_class.items()):
    unseen_words=set(vocabulary_list).difference(feature_set.keys())
    feature_instance_tallies_by_class[relation].update(dict.fromkeys(unseen_words,0))

    #Calculate class prior and store in dictionary Uses add delta smoothing
    class_priors[relation] = (PRIOR_DELTA + vector_instance_counts_by_class[relation]) / (len(vector_instance_counts_by_class.keys()) + sum(vector_instance_counts_by_class.values()))
    #Get sum of all features in class
    total_feature_instance_tally_sum = sum(feature_set.values())

    #divide using delta values in order to get the conditional probability
    for feature, feature_instance_tallies in feature_set.items():
        conditional_probabilities_by_class[relation][feature] = float((COND_PROB_DELTA + feature_instance_tallies)) / (COND_PROB_DELTA * len(vocabulary_list) + total_feature_instance_tally_sum)


#Write all the probbabilities to file. Formatting is important for downstream parsing
output_file.write("{: >25}\n".format("%%%%% prior prob P(c) %%%%%\n" ))
for relation , prior in sorted(class_priors.items(), key=lambda x: x[1], reverse=True):
    output_file.write("{: >15} {: >20} {: >20}\n".format(relation,prior,math.log10(prior)))
output_file.write("{: >25}\n".format("\n%%%%% conditional prob P(f|c) %%%%%" ))
for relation, features in sorted(conditional_probabilities_by_class.items()):
    output_file.write("{: >15}\n".format("\n%%%%% conditional prob P(f|c) c='"+ relation + "' %%%%%\n" ))
    for feature, cond_prob in sorted(features.items(), key=lambda x: x[1], reverse=True):
        output_file.write("{: >15} {: >20} {: >20}\n".format(feature, cond_prob, math.log10(cond_prob)))
