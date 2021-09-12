#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Script collects and processing raw data into batches
#Writes batches to file in order to be used later


###### Acknowledgments ##########################
#- Prepare_data, prepare wiki, prepare_tbc copied for reproducibility from:
#  @inproceedings{wang2019bert,
    #title = "{BERT} has a Mouth, and It Must Speak: {BERT} as a {M}arkov Random Field Language Model",
    #author = "Wang, Alex  and  Cho, Kyunghyun", month = jun, year = "2019"
###############################

#Dependencies
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
from utility_models import settings, tokenizer
import torch
import random

# Set the seed value all over the place to make this reproducible.
seed_val = settings.get_random_state()
random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#Method used to collect, process, and divide raw data into bathes for training and testing
def get_data():

    wiki103_file_path = settings.get_raw_wiki_path()
    tbc_file_path = settings.get_raw_tbc_path()

    #Data cleaning and preparation according to previous work
    wiki_data=prepare_wiki(wiki103_file_path)
    tbc_data= prepare_tbc(tbc_file_path)
    input_sents = wiki_data + tbc_data

    #Method groups batches of data by normalizing length of BERt tokens
    smart_batches = create_smart_batches(input_sents, batch_size=settings.get_batch_size())

    # Divide data into training and validation
    train_inputs, validation_inputs = train_test_split(smart_batches[:settings.get_num_batches()], random_state=settings.get_random_state(), test_size=settings.get_test_size_ratio() )

    #Writes prepared data to file to be saved
    with open(settings.get_train_inputs_path(), 'w+') as f:
        [f.write(" ".join(tokenizer.convert_ids_to_tokens(sent)[1:-1]) + "\n") for batch in train_inputs for sent in batch]

    with open(settings.get_validation_inputs_path(), 'w+') as f:
        [f.write(" ".join(tokenizer.convert_ids_to_tokens(sent)[1:-1]) + "\n") for batch in validation_inputs for sent in batch]

    with open(settings.get_proc_wiki_path(), 'wb') as f:
        pickle.dump(wiki_data, f)

    with open(settings.get_proc_tbc_path(), 'wb') as f:
        pickle.dump(tbc_data, f)
    #######################################

#Method that takes lists of word sequences, tokenizes, removes lists with OOV tokens and those with few than minimum sample length.
#tokens are organized into the specified number of batches by length of sequence.
def create_smart_batches(input_sents , batch_size:int) -> list:

    # Enocdes sentences into bert tokens
    batched_ids = tokenizer.batch_encode_plus(
        input_sents,
        pad_to_max_length=False,
        truncation=True,  ##### Currently truncates a sentence
        max_length=settings.get_sample_size() + 2,
        # Max length is sentence length , Must add two for CLS and final SEP token
        add_special_tokens=True)['input_ids']

    samples = [sent_ids for sent_ids in batched_ids if len(sent_ids) > settings.get_min_sample_len() and tokenizer.unk_token_id not in sent_ids]

    d=defaultdict(list)
    [d[len(sample)].append(sample) for sample in samples]

    batch_ordered_sentences = []

    while len(d)>0: ### This loop will allow for batches without any padding
        key,samples=random.sample(list(d.items()), 1)[0]

        while len(samples) >= batch_size:
            to_take = min(batch_size, len(samples))
            select = random.randint(0, len(samples) - to_take)
            batch = samples[select:(select + to_take)]

            max_size = max([len(sen) for sen in batch])

            for sent_i in range(len(batch)):
                sen=batch[sent_i]
                num_pads = max_size - len(sen)
                padded_input = sen + [tokenizer.pad_token_id] * num_pads
                batch[sent_i]=padded_input

            batch_ordered_sentences.append(torch.tensor(batch))
            del samples[select:select + to_take]

        del d[key]

    return batch_ordered_sentences


##### Preparation code was kept the same from previous paper for reproducibility
def prepare_data(data_file, replacements={}, uncased=True):
    data = [d.strip().split() for d in open(data_file, 'r').readlines()]
    if uncased:
        data = [[t.lower() for t in sent] for sent in data]
    for k, v in replacements.items():
        data = [[t if t != k else v for t in sent] for sent in data]
    return data

def prepare_wiki(data_file, uncased=True):
    replacements = {"@@unknown@@": "[UNK]"}
    return prepare_data(data_file, replacements=replacements, uncased=uncased)

def prepare_tbc(data_file):
    replacements = {"``": "\"", "\'\'": "\""}
    return prepare_data(data_file, replacements=replacements)
##############################################################################

if __name__ == "__main__":
    get_data()
