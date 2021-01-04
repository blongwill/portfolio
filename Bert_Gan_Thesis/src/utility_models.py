#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Helper module to instantiate required objects: device, tokenizer, and perplexity_model
#Import this file to use objects

#Dependencies
import torch
from transformers import BertTokenizer
from configuration import settings


#Helper method used to get GPU or CPU object for selecting model attachement
def get_device():
    #Check if there is a GPU available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        settings.write_debug('Using GPU:' + torch.cuda.get_device_name(0))
    # Uses the CPU if no GPU available
    else:
        settings.write_debug('Using CPU.')
        device = torch.device("cpu")

    return device

def load_perplexity_model():
    settings.write_debug("Loading pre-trained transformer_lm.wiki103.adaptive")
    # Load an English LM trained on wiki103 data
    en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wiki103.adaptive', tokenizer='moses')
    en_lm.requires_grad=False
    en_lm.eval()  # disable dropout
    settings.write_debug('Finished Loading pre-trained transformer_lm.wiki103.adaptive')
    return en_lm


settings.write_debug('Loading Utility models')
device = get_device() #Using for loading models to GPU
model_type=settings.get_model_type() #bert_base_uncased
tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=model_type.endswith("uncased"))
perplexity_model = load_perplexity_model().to(device) #Outside model used for evaluating perplexity of BERT models
settings.write_debug('Finished loading utility models')
