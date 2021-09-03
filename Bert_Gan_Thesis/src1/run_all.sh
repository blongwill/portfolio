#!/bin/sh

#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#sh script for running all modules of thesis in correct order


src/data_collection.py $1

src/prelim_eval.py $1

src/train.py $1

src/evaluate.py $1
