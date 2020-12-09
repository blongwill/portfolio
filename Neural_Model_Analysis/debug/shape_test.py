#!/usr/bin/env python3


import sys
from transformers import BertTokenizer


#./test_toy_shape.py /Users/BenLongwill/Desktop/toy_dataset.txt

# Output: an array of token ids
def encode_sentences(tokenizer: BertTokenizer, sentences: list) -> (list,list):
    # Use the pretrained BERT transfer model
    #return as an array of token id's
    encoding_dict = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences, pad_to_max_length=True, max_length=512, add_special_tokens=True, return_tensors='pt')                  #converts tokens to id's and includes CLS and SEP
                                      # can be converted back with convert_ids_to_tokens
    print(encoding_dict.keys())
    #print(encoding_dict['tokens'])
    return encoding_dict['input_ids'], encoding_dict['attention_mask']



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def get_tweets_and_labels(dataset_path:str)->(list,list):
    lines = open(dataset_path).readlines()
    tweets=[]
    labels=[]
    for line in lines:
        tweet, label=line.split("\t")
        tweets.append(tweet)
        labels.append(labels)
    return tweets, labels


dataset_path=sys.argv[1]
tweets,labels=get_tweets_and_labels(dataset_path)

input_ids, attention_masks=encode_sentences(tokenizer,tweets)

for id in input_ids:
    print(tokenizer.convert_ids_to_tokens(id))

print("input_ids tensor shape ----------------------> " + str(input_ids.size()))
print("attention_masks tensor shape ----------------------> " + str(input_ids.size()))




