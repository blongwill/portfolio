#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Module designed to call evaluation procedure on BERT Generation model and GPT model before fine-tuning takes place
#New Generators are instantiated each time and not saved in pre-evaluation
#prelim_eval.py can be run from the base directory by calling ./src/prelim_eval.py config_file_name (not including .txt extension)

#Dependencies
from configuration import settings
from utility_models import device
from gan_evaluation import evaluate_bert, evaluate_gpt
from gan_model import GAN


def main():

    settings.write_result("****************** Preliminary Evaluation ******************\n")

    generator=GAN.Generator().to(device) #Instantiate new BERT generator

    evaluate_bert(generator=generator, output_file_path=settings.get_bert_prelim_out_path()) #Calls evaluation on BERT generator

    evaluate_gpt(output_file_path=settings.get_gpt_prelim_out_path()) ###Instantiate and evaluates new GPT generator

if __name__ == "__main__":
    main()
