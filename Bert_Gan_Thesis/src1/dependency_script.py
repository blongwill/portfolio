#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Copy and paste lines in command line in order to ensure models are correctly downloaded prior to running


# Requirements = ( ( machine == "patas-gn2.ling.washington.edu" ) )

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8


from transformers import BertTokenizer
BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
from transformers import BertForSequenceClassification
BertForSequenceClassification.from_pretrained("bert-base-cased")
BertForSequenceClassification.from_pretrained("bert-base-uncased")
from transformers import BertForMaskedLM
BertForMaskedLM.from_pretrained("bert-base-uncased")
BertForMaskedLM.from_pretrained("bert-base-cased")
import torch
torch.hub.load('pytorch/fairseq', 'transformer_lm.wiki103.adaptive', tokenizer='moses')


'''
