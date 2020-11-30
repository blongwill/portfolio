#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

data_dictionary=defaultdict(lambda:[])
file_name=sys.argv[1]
output_name=sys.argv[2]
document_quota=int(sys.argv[3])

#### Trims count of document in each class to match a document quota max

with open(file_name) as file:

    for line in file.readlines():
        relation , features=line.split(" ",maxsplit=1)
        data_dictionary[relation].append(line)
    with open(output_name,"w") as trimmed_file:
        print("The quota is : " + str(document_quota))
        for relation, features_list in data_dictionary.items():
            if relation != "resting_place" and relation != 'partner(s)':
                if len(features_list)<document_quota:
                    print(relation + ": List is less than document quota at " + str(len(features_list)) + " : relation ignored from output")
                else:
                    print("Trimming " + relation + ": from " + str(len(features_list)) + " to " + str(document_quota))

                    for document in features_list[:document_quota]:
                        trimmed_file.write(document)






