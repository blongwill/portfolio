#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project

#Script used to parse famous actor's names from a large Movie Datase file
#Splits names into different percentages based on target test vs training percentage
#Shuffles and removes accents from names


#Script Dependencies
import unicodedata
from random import shuffle
import sys

'''
Params:
    1.) Name file called name.basics.tsv from IMDB
    2.) Training output file path names
    3.) Testing output file path names
Example call:./parse_names.py name_file training_names_output_file testing_names_output_file
'''
names_file_path= sys.argv[1]
training_output_file_path = sys.argv[2]
testing_output_file_path = sys.argv[3]

##Only 10% of names are available for testing
percentage_of_training=.9
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

with open(training_output_file_path, "w+") as training_output, open(testing_output_file_path, "w+") as testing_output:

    list_of_names=[]
    #Doesn't include first line bc of header
    for line in open(names_file_path).readlines()[1:]:
        line_split=line.split()
        index=1 #Names started in first column
        full_name=[]
        #Names could be variable length and sometimes include numbers, these were skipped
        while not str.isnumeric(line_split[index]) and line_split[index]!=r'\N':
            full_name.append(line_split[index])
            index += 1
        if len(full_name)>1:
            list_of_names.append(" ".join(full_name))


    #Shuffle and write names to file
    shuffle(list_of_names)
    training_divide=int(len(list_of_names)*percentage_of_training)
    training_names=list_of_names[:training_divide]
    testing_names=list_of_names[training_divide:]
    training_output.write("\n".join(training_names))
    testing_output.write("\n".join(testing_names))

