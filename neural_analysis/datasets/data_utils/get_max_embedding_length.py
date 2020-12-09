#!/usr/bin/env python3

from transformers import BertTokenizer
import pandas as pd
import sys
import numpy as np

#### Must activate conda environment specific using yml file prior to running this script
#### Run on dataset to calculate the optimal maximum embedding token length to be used with BERT
#### Accepts multiple dataset path arguments

####### To Run: ./get_max_embedding_length.py dataset_path ##################

def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

paths = sys.argv[1:]

for dataset_path in paths:


    # Load the dataset into a pandas dataframe.
    df = pd.read_csv(dataset_path, delimiter='\t', header=None, names=['sentence', 'label'], quoting=3)
    df = df[df['sentence'].notnull()]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    #Tokenizes script in batch without padding or special characters
    encoding_dict = tokenizer.batch_encode_plus(df.sentence.values, pad_to_max_length=False, add_special_tokens=False)

    #Gets the length of each tokenized tweet
    lens= [len(mask) for mask in encoding_dict['attention_mask'] ]

    #output
    print(dataset_path + " -------------> " + str(np.max(removeOutliers(lens, 1.5))))



    ###Other useful information
    #print("95th percentile tokenized embedding length: " + str(np.percentile(lens, 95)))
    #print("Average size without outliars: " + str(np.average(removeOutliers(lens, 1.5))))

