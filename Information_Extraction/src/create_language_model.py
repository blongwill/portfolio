#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import math
import sys

#./create_language_model.py input_file output_file
#./create_language_model.py /Users/BenLongwill/Desktop/openio/train.vectors.txt /Users/BenLongwill/Desktop/openio/language_model.txt
#./create_language_model.py /Users/BenLongwill/Desktop/openio/binary_relation.vectors.txt /Users/BenLongwill/Desktop/openio/binary_language_model.txt

training_vectors=open(sys.argv[1]).readlines()
COND_PROB_DELTA=.5
PRIOR_DELTA=1

output_file=open(sys.argv[2],'w+')


vector_instance_counts_by_class= defaultdict(lambda:0)
feature_instance_tallies_by_class= defaultdict(lambda:defaultdict(lambda:0))
relation_tally_total=defaultdict(lambda:0)
vocabulary_list=set()
conditional_probabilities_by_class= defaultdict(lambda:defaultdict(lambda:0))
class_priors=defaultdict(lambda:0)

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


#### Update unseen words for each class
for relation , feature_set in list(feature_instance_tallies_by_class.items()):
    unseen_words=set(vocabulary_list).difference(feature_set.keys())
    feature_instance_tallies_by_class[relation].update(dict.fromkeys(unseen_words,0))

    class_priors[relation] = (PRIOR_DELTA + vector_instance_counts_by_class[relation]) / (len(vector_instance_counts_by_class.keys()) + sum(vector_instance_counts_by_class.values()))


    total_feature_instance_tally_sum = sum(feature_set.values())

    for feature, feature_instance_tallies in feature_set.items():
        conditional_probabilities_by_class[relation][feature] = float((COND_PROB_DELTA + feature_instance_tallies)) / (COND_PROB_DELTA * len(vocabulary_list) + total_feature_instance_tally_sum)



#, key=lambda x: x[1], reverse=True
output_file.write("{: >25}\n".format("%%%%% prior prob P(c) %%%%%\n" ))
for relation , prior in sorted(class_priors.items(), key=lambda x: x[1], reverse=True):
    output_file.write("{: >15} {: >20} {: >20}\n".format(relation,prior,math.log10(prior)))
output_file.write("{: >25}\n".format("\n%%%%% conditional prob P(f|c) %%%%%" ))
for relation, features in sorted(conditional_probabilities_by_class.items()):
    output_file.write("{: >15}\n".format("\n%%%%% conditional prob P(f|c) c='"+ relation + "' %%%%%\n" ))
    for feature, cond_prob in sorted(features.items(), key=lambda x: x[1], reverse=True):
        output_file.write("{: >15} {: >20} {: >20}\n".format(feature, cond_prob, math.log10(cond_prob)))
