#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Script reads in trained generator BERT model
#Runs evaluation procedure to obtain post-training measures


#Dependencies
from configuration import settings
from utility_models import device
from gan_evaluation import evaluate_bert, evaluate_gpt
from gan_model import GAN


def main():
    settings.write_result("****************** Evaluation ******************\n")
    generator=GAN.Generator(settings.get_generator_path()).to(device)
    evaluate_bert(generator=generator, output_file_path=settings.get_bert_eval_out_path())

if __name__ == "__main__":
    main()
