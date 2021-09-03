#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project

'''
File contains NB classifier class object and main runs the classifier

Params:
    1.) test_vector_file_path:str - path to file containing word sequences separated by newline
    2.) language_model_file_path:str - path to file containing class class prior and feature log probabilities
    3.) sys_file_path:str - path to output file for sample index and confidence of classification by class
    4.) acc_file_path:str - path to output file for confusion matrix of testing data.
    5.) data_is_labeled:bool - Allows classifier to know when to self evaluate

Features used were real value numbers of occurrences and included add-one smoothing in or-der to return a smooth confidence percentage for each classification.
Unknownw ords were skipped when calculating classification

Example call:
./run_classifier.py test_vector binary_language_model.txt, class_language_model, binary_sys_out_file_name:str binary_acc_file_name:str data_is_labele:bool class_language_model.txt class_sys_out_file_name class_acc_file_name

'''


#Script dependenecies
import math
from collections import defaultdict
import sys
import nltk
#Uses the porterstemmer for stemming test sentences
ps = nltk.PorterStemmer()

class naive_bayes_classifier:
    def __init__(self, language_model_file):
        self.class_priors,self.cond_probs=self.configure_classifier(language_model_file)

    #Configures the naive bayes classifier based by reading in a language model file
    #Parses file by splitting and then reading lines
    #Stores probabilities in default dictionary data structure -- Which are later stored as class attributes
    def configure_classifier(self, language_model_file):
        class_priors = {}
        cond_probs = defaultdict(lambda: defaultdict(lambda:0))

        prior_section, conditional_section = open(language_model_file).read().split("%%%%% conditional prob P(f|c) %%%%%",
                                                                            maxsplit=1)
        for line in prior_section.split("\n\n")[1].split("\n"):
            relation, class_prior, log_class_prior = line.strip().split()
            class_priors.update({relation: log_class_prior})

        for section in conditional_section.split("%%%%% conditional prob P(f|c) c="):
            relation, cond_probs_section = section.split("\n", maxsplit=1)
            relation = relation.split(" ")[0].strip("'")

            for line in cond_probs_section.strip().splitlines():
                feature, cond_prob, log_cond_prob = line.strip().split()
                cond_probs[relation][feature] = log_cond_prob

        return class_priors, cond_probs

    #### Used for classifier evaluation purposes for test time
    #binary or mutliclass relation depending on language model input file
    #This is a decoding procedure
    def classify(self,test_vectors:list, data_is_labeled:bool, sys_file_path:str, acc_file_path:str):
        with open(sys_file_path, 'w') as sys_file:

            total_test_vectors = len(test_vectors)
            prediction_counts = defaultdict(lambda: defaultdict(lambda: 0))

            #Itterates test word sequences
            for vector in test_vectors:

                test_conditional_log_sum = defaultdict(lambda: 0)
                final_probabilities = defaultdict(lambda: 0)

                #Gets relation index -- e.g 1-2 for the wikipedia entry and sequence number
                relation_index, vector=vector.split(" ",maxsplit=1)

                if data_is_labeled:
                    gold_label, features = vector.split(" ", maxsplit=1)
                else:
                    features = vector
                    gold_label=None

                #Itterates across each class to create probability distribution
                for relation_class, prior in self.class_priors.items():

                    test_conditional_log_sum[relation_class] = float(prior)

                    #Adds log probabilities for each word (The same as multiplying regular propabilities)
                    for word in nltk.word_tokenize(features):
                        word = ps.stem(word).lower()  # Stems the tests words
                        test_conditional_log_sum[relation_class] += float(self.cond_probs[relation_class][word])

                #### "Funny ones" method to avoid underflow
                for relation, current_value in test_conditional_log_sum.items():
                    final_probability = 1 / sum([math.pow(10, math.fabs(current_value) + class_value) for class_value in
                                                 test_conditional_log_sum.values()])
                    final_probabilities[relation] = final_probability

                sorted_probabilities = sorted(final_probabilities.items(), key=lambda x: x[1], reverse=True)

                #Relation index kept the same in output file
                sys_file.write(relation_index + " ")

                if data_is_labeled:
                    prediction_counts[gold_label][sorted_probabilities[0][0]] += 1
                    sys_file.write(gold_label + " ") ### Adds in gold label tag to sys output

                for relation, final_probability in sorted_probabilities:
                    sys_file.write(relation + " " + str(final_probability) + " ")
                sys_file.write("\n")

            #If data is labeled classifier can then produce accuracy file
            #Self Evaluation takes place below
            if data_is_labeled:
                with open(acc_file_path, 'w') as acc_file:
                    accuracy_count = 0
                    precisions={}
                    recalls=defaultdict(lambda:0)
                    acc_file.write("Confusion matrix for the testing data:\n row is the truth, column is the system output\n")
                    acc_file.write("\t" + "\t".join(sorted(self.class_priors.keys())) + "\n")


                    for gold_label, predictions in sorted(prediction_counts.items()):

                        acc_file.write(gold_label[:3] +":")
                        for prediction in sorted(self.class_priors.keys()):

                            recalls[gold_label]+=(prediction_counts[prediction][gold_label])

                            if gold_label == prediction:
                                accuracy_count += predictions[prediction]
                                precisions[gold_label]=float(predictions[prediction])/sum(predictions.values())

                            acc_file.write("\t" + str(predictions[prediction]))

                        acc_file.write("\t |" + gold_label[:3] + ":")
                        current_recall=float(predictions[gold_label])/recalls[gold_label]
                        acc_file.write(" P=" + str(precisions[gold_label]) + " R=" + str(current_recall) + " F1=" + str((2.0*precisions[gold_label]*current_recall)/(precisions[gold_label]+current_recall)) )
                        acc_file.write("\n")

                    acc_file.write("\t Test accuracy=" + str(accuracy_count / total_test_vectors) + "\n")






def main():

    test_vectors = open(sys.argv[1]).readlines()

    language_model_file = sys.argv[2]
    sys_file_path=sys.argv[3]
    acc_file_path=sys.argv[4]
    data_is_labeled=bool(eval(sys.argv[5]))

    nbc = naive_bayes_classifier(language_model_file)

    nbc.classify(test_vectors=test_vectors, data_is_labeled=data_is_labeled,
                 sys_file_path=sys_file_path, acc_file_path=acc_file_path)


if __name__ == '__main__':
    main()