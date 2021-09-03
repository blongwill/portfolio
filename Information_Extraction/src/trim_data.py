#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict


#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project
# A data processing script that reads data into a dictionary object and trims the count of raw documents within class to match a document quota max


'''
Parameters 
    file_name:str - File of data in the format 'class\tfeature_i' to be read 
    output_name:str - File name used to create output file 
    document_quota:int - amount of documents required for each class. 
'''
data_dictionary=defaultdict(lambda:[])
file_name=sys.argv[1]
output_name=sys.argv[2]
document_quota=int(sys.argv[3])

# Opens and Reads input file into dictionary object and trims with respect to max value
with open(file_name) as file:
    for line in file.readlines():
        relation , features=line.split(" ",maxsplit=1)
        data_dictionary[relation].append(line)
    with open(output_name,"w") as trimmed_file:
        print("The quota is : " + str(document_quota))
        for relation, features_list in data_dictionary.items():
            #Removes extraneous classes
            if relation != "resting_place" and relation != 'partner(s)':
                #In the case that there were less documents than quote, class ommited
                #additional data collection may be required
                if len(features_list)<document_quota:
                    print(relation + ": List is less than document quota at " + str(len(features_list)) + " : relation ignored from output")
                else:
                    print("Trimming " + relation + ": from " + str(len(features_list)) + " to " + str(document_quota))

                    #Write documents to output file
                    for document in features_list[:document_quota]:
                        trimmed_file.write(document)






