#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project
#Compares relation counts between current system and OPENIE system

#Script dependency
import sys

'''
Params:
    openie_file_path:str - Path to file that contains extracted semantic relation output from of the openIE extractor
    semantic_relation_file_path:str - Path to file containing semantic relations and counts from the current system
    output_file_path:str - Path to output file that displays sentence index and recall for that sentence
'''

openie_file_path=sys.argv[1]
semantic_relation_file_path=sys.argv[2]
output_file_path=sys.argv[3]

with open(openie_file_path) as openie_file, open(semantic_relation_file_path) as my_file, open(output_file_path , "w") as output_file:

    #Parses openIe output by splitting on a blank line
    #If the block is not empty then it skips the first line bc it's a sentence
    #Then counts the number of lines for each relation and stores in dictionary
    openie_counts={}
    for block in openie_file.read().split("\n\n"):
        split_block =block.split("\n")
        if len(split_block) > 0:
            openie_counts[split_block[0].strip()]=len(split_block[1:])
    #For each line in the extracted output file split to get the semantic relation count in first field
    #USes dictionary to lookup sentence count to compare with OPenIE
    #Computes recall using openIE as gold standard
    for index, line in enumerate(my_file.readlines()):
        if len(line)>1:
            sem_rel_count, rest=line.split(" " , maxsplit=1)
            sem_rel_count=float(sem_rel_count.split(":")[1])

            sent_index, rest = rest.split(" ", maxsplit=1)
            sent_index=sent_index.split(":")[1]

            rest=rest.strip().replace("\n", "")
            openie_count=openie_counts.get(rest)

            #Calculations for recall take place here
            if openie_count is not None:
                if float(openie_count)<1:
                    openie_count=1.0
                recall = (sem_rel_count / float(openie_count))

                #If more than relations are found than in openIE then recall is clamped to 1.0
                if recall > 1.0:
                    recall = 1.0
                output_file.write("Sent:"+sent_index + " R=" + str(recall) + "\n" )
            else:
                output_file.write("Sent:" + sent_index + " Sentnece mismatch or Zero Division \n")
