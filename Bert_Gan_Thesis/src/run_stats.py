#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Quickly devloped helper script to collect numerical output from output files and calculate/print mean and standard deviation for paper
#Navigates directory and saves data in dictionary objects for later analysis
#Can be used by .src/run_stats.py out_directory

#Dependencies
import sys
import numpy as np
from collections import defaultdict


#Collects data by reading lines containing ':' characters values into dictionary as key/value
def grab_info(section_of_doc, d):

    for line in section_of_doc.split("\n"):
        if ":" in line:
            k, v = line.split(":")
            d[k].append(float(v.strip()))

    return d

#Calculates the means and standard deviations and prints them to main.
def print_stats(info_d):

    for k,v in info_d.items():
        print(k + ": {:0.2f}  +- {:0.2f}".format(np.mean(v),  np.std(v,ddof=1)))


prelim_d = defaultdict(lambda :list())
eval_d = defaultdict(lambda :list())

for f_path in sys.argv[1:]:

    #Opens all files in the output directory
    with open(f_path) as cur_f:
        #Processes files by splitting headers
        prelim, eval = cur_f.read().split("****************** Evaluation ******************")

        #Collects both preliminary and final evaluation values
        prelim_d=grab_info(prelim, prelim_d)
        eval_d=grab_info(eval, eval_d)

#Prints to main
print("Preliminary:\n")
print_stats(prelim_d)
print()
print("Post Fine-tuning:\n")
print_stats(eval_d)
print()



