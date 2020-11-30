#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict


openie_file_path=sys.argv[1]
semantic_relation_file_path=sys.argv[2]
output_file_path=sys.argv[3]

with open(openie_file_path) as openie_file, open(semantic_relation_file_path) as my_file, open(output_file_path , "w") as output_file:

    openie_counts={}
    for block in openie_file.read().split("\n\n"):
        split_block =block.split("\n")
        if len(split_block) > 0:
            openie_counts[split_block[0].strip()]=len(split_block[1:])

    for index, line in enumerate(my_file.readlines()):
        if len(line)>1:
            sem_rel_count, rest=line.split(" " , maxsplit=1)
            sem_rel_count=float(sem_rel_count.split(":")[1])

            sent_index, rest = rest.split(" ", maxsplit=1)
            sent_index=sent_index.split(":")[1]

            rest=rest.strip().replace("\n", "")
            openie_count=openie_counts.get(rest)


            if openie_count is not None:
                if float(openie_count)<1:
                    openie_count=1.0
                recall = (sem_rel_count / float(openie_count))

                if recall > 1.0:
                    recall = 1.0
                output_file.write("Sent:"+sent_index + " R=" + str(recall) + "\n" )
            else:
                output_file.write("Sent:" + sent_index + " Sentnece mismatch or Zero Division \n")
