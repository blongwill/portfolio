#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unicodedata
from random import shuffle
import sys




#./parse_names.py name_file training_names_output_file testing_names_output_file
#./parse_names.py name.basics.tsv training_names.txt testing_names.txt
import unidecode as unidecode

percentage_of_training=.9


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


with open(sys.argv[2], "w+") as training_output, open(sys.argv[3], "w+") as testing_output:

    list_of_names=[]
    for line in open(sys.argv[1]).readlines()[1:]: #First 30,000 names divided on a 30/70 split
        #line=unidecode.unidecode(line)
        #line=str(line.encode("UTF-8"),"UTF-8")
        #line=remove_accents(line)
        line_split=line.split()

        index=1
        full_name=[]
        while not str.isnumeric(line_split[index]) and line_split[index]!=r'\N':
            full_name.append(line_split[index])
            index += 1
        if len(full_name)>1:
            list_of_names.append(" ".join(full_name))


    shuffle(list_of_names)
    training_divide=int(len(list_of_names)*percentage_of_training)
    training_names=list_of_names[:training_divide]
    testing_names=list_of_names[training_divide:]

    training_output.write("\n".join(training_names))
    testing_output.write("\n".join(testing_names))

