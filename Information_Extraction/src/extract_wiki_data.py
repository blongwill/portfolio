#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
import urllib.request as urllib2

import nltk
from bs4 import BeautifulSoup
import re
import random
import sys
from nltk.corpus import stopwords
from urllib.parse import quote
from unidecode import unidecode
import re

ps = nltk.PorterStemmer() #Used for stemming
hdr = {"User-Agent": "My Agent"} # For HTML headers


#./extract_wiki_data entity_list_file:File class_output_file_name:str binary_output_file_name:str document_quota:int is_training_data:Bool

list_of_entities=open(sys.argv[1]).readlines()
document_quota=int(sys.argv[4])
is_training_data = bool(eval(sys.argv[5]))

stop_words = set(stopwords.words('english'))

''' ##### Used for parsing semantic relations using shallow features
def get_relation_contexts(sentence):
    print(type(sentence))
    print(sentence)
    tagged_sent = nltk.pos_tag(nltk.word_tokenize(sentence))

    points_of_interest = [index for index, element in enumerate(tagged_sent) if
                          'NN' in element[1] or 'PRP' in element[1]]

    if len(points_of_interest) > 1:
        for index in range(1, len(points_of_interest)):
            if any('VB' in pos for word, pos in
                   tagged_sent[points_of_interest[index - 1]:points_of_interest[index] + 1]):
                yield " ".join([word for word, pos in tagged_sent[points_of_interest[index - 1]:points_of_interest[index] + 1]])

def get_entity_relation_contexts(sentence,entity1_list,entity2):

    # Checked in outside method
    for possible_context in re.split(pattern="(?<="+ entity2+")",string=sentence,flags=re.IGNORECASE)[:-1]:
        print(possible_context)
        for entity1 in entity1_list:
            lower_possible_context=possible_context.lower()
            if entity1 in lower_possible_context.split():
                print("context accepted")
                yield possible_context[lower_possible_context.index(entity1):]
                break
'''

def print_vector(relation_name, sentence, output_file):
    print("Entered Vector print method")
    sys.stdout.flush()

    sentence=sentence.replace("\n", " ").replace("\r","") # Remove white spaces
    current_relation = Relation(relation_name)
    if is_training_data:
        print("Vector is training_data")
        sys.stdout.flush()
        for token in [token[0].lower() for token in nltk.pos_tag(nltk.word_tokenize(sentence)) if
                       token[1] != 'NNP' and token[1] != 'NNPS' and token[1] != 'JJ' and token[1] != 'CD' and 'PRP' not in
                       token[1]]:
            if token not in stop_words and token.isalnum() and token != entity2 and token not in entity_name.lower():
                token = ps.stem(token)
                current_relation.features.update({token: current_relation.features.get(token, 0) + 1})

        print("Printing vector")
        sys.stdout.flush()
        if len(current_relation.features)>0:
            output_file.write(current_relation.name)
            for feature, frequency in current_relation.features.items():
                output_file.write(" " + feature + ":" + str(frequency))
            output_file.write("\n")
    else:
        print("Vector is not training_data")
        sys.stdout.flush()
        print("Printing vector")
        sys.stdout.flush()
        output_file.write(current_relation.name + " " + sentence + "\n")

    output_file.flush()

class Relation:
    #Relation objects hold a name and the features
    def __init__(self, name:str=None):
        self.name=name
        self.features={} # 'word'-freq bag of words and 'speaker' - freq count and 'episode-name' -freq count
    def __repr__(self):
        return self.name

def clean(input):
    #REmoves parens and brackets along with their contents
  cleanr = re.compile('\[.*?\]|\(.*?\)|/.*?/')
  cleantext = re.sub(cleanr, "", input)
  return cleantext

def extract_table_information(bs):
    #Takes beatiful soup object parse and collects table semantic relation and argument data
    relation_to_entity2 = {}
    ### Removes citations
    body = bs.find("body")
    for element in body.findAll("sup", {'class': 'reference'}):
        element.decompose()

        table = bs.find("table", {'class': 'infobox'})

        if table:
            table_rows = table.find("tbody").findAll("tr")

            for row in table_rows:

                for element in row.findAll("table", {'class': 'wikitable'}):
                        element.decompose()

                th = row.find('th')
                td = row.find('td')

                if th and td:

                    relation = th.get_text().strip().replace(" ", "_").lower()
                    entity = td.get_text().strip().lower()


                        ### Locations take first attribute which is usually most specific
                    if relation == "born" or relation == "died" \
                              or relation == 'spouse(s)' or relation == 'education': # relation == 'partner(s)' or relation == "resting_place"
                        attributes = td.findAll("a")
                        if len(attributes) > 0:
                            entity = attributes[0].get_text().split(",")[0].lower()


                    elif relation == "occupation":
                        #### Take only the first occupation
                        entity = entity.replace("\n", ",").split(",")[0]


                    if relation == "born" or relation == "died" or relation == "occupation" or relation == 'spouse(s)'\
                                or relation == "education": #or relation == "resting_place" \ or relation == 'partner(s)':
                            relation_to_entity2.update({relation: clean(entity)})
    return relation_to_entity2

def extract_wiki_entry(url:str):
    print("Entered extracting wiki entries")
    sys.stdout.flush()
    retries=2
    while retries>0:
        print("In while loop for extracting wiki page")
        sys.stdout.flush()
        try:
            print("Within try of while loop")
            sys.stdout.flush()

            req = urllib2.Request(url, headers=hdr)
            html = urllib2.urlopen(req)

            bs = BeautifulSoup(unicodedata.normalize("NFKD", html.read().decode('utf-8')).replace('\0','') ,"html.parser")

            if html.getcode() == 200:
                return bs
        except Exception as inst:
            print(inst)
            retries-=1
    return None

def extract_and_print_negative_examples(negative_training_quota:int):
    print("Entered extract NEGATIVE EXAPMLES")
    sys.stdout.flush()

    bs=extract_wiki_entry("https://en.wikipedia.org/wiki/Special:Random")

    unrelated_text = " ".join([clean(p.get_text().strip()) for p in bs.findAll("p")])

    unrelated_sentences = nltk.sent_tokenize(unrelated_text)

    while int(negative_training_quota) > 0:
        print("Entered while loops for negative_traing quota")
        sys.stdout.flush()


        while len(unrelated_sentences)<1:
            print("Entered while loop if unrelated sentences empty and getting new one  ")
            sys.stdout.flush()

            bs=extract_wiki_entry("https://en.wikipedia.org/wiki/Special:Random") ##Gets random wikipedia article

            unrelated_text = " ".join([clean(p.get_text().strip()) for p in bs.findAll("p")])

            unrelated_sentences = nltk.sent_tokenize(unrelated_text)

        ''' Checked if relations exist in sentence, Removed because it caused a never ending loop
        while not relations.isdisjoint(set(unrelated_sentence.lower().split())):
            print("Entered while loop to choose new sentence if disjoint ")
            sys.stdout.flush()
            unrelated_sentence = unrelated_sentences[random.randint(0, len(unrelated_sentences) - 1)]
        '''

        unrelated_sentence = unrelated_sentences[random.randint(0, len(unrelated_sentences) - 1)]

        unrelated_sentences.remove(unrelated_sentence)

        print("Going to print Negative vector ")

        sys.stdout.flush()
        print_vector("absent", unrelated_sentence, binary_output_file)  # text, output_file)

        negative_training_quota -= 1


with open(sys.argv[2], "w", encoding='utf8') as class_output_file, open(sys.argv[3], "w", encoding='utf8') as binary_output_file:
    document_counter= nltk.defaultdict(lambda:0)

    print("start program")
    sys.stdout.flush()

    for entity_name in list_of_entities:
        print("New Entity1")
        sys.stdout.flush()

        entity1 = entity_name.strip().replace(" ", "_")

        print("https://en.wikipedia.org/wiki/" + entity1)
        sys.stdout.flush()

        bs = extract_wiki_entry("https://en.wikipedia.org/wiki/" + quote(entity1))
        if bs is not None:
            print("bs is not none")
            sys.stdout.flush()

            try:
                relation_to_entity2 = extract_table_information(bs)
            except Exception as e:
                print(e)
                continue

            ### separates wiki_text into a list of sentences
            wiki_text = " ".join([clean(p.get_text().strip()) for p in bs.findAll("p")])

            wiki_sentences = nltk.sent_tokenize(wiki_text)

            if len(wiki_sentences)>0 and len(relation_to_entity2)>0:

                print("Wiki text is within lenght and there are relations from the wiki page ")
                sys.stdout.flush()

                entity1_list = set(entity_name.lower().split() + ['he'] + ['she']) # Splits the name and only uses he or she

                negative_training_quota = 0  # Used for making absence the same amoutn of times
                for relation_name, entity2 in relation_to_entity2.items():

                    if document_counter[relation_name] < document_quota:

                        for sentence in wiki_sentences:
                            if sentence:
                                #print("Checking each sentence")
                                sys.stdout.flush()

                                lower_sentence = sentence.lower()

                                if (entity2 in lower_sentence) and any(entity1_list for word in lower_sentence):
                                    print("sentence found")
                                    sys.stdout.flush()

                                    print("Going to output the PRESENT vectors")
                                    sys.stdout.flush()

                                    print_vector('present', sentence, binary_output_file)
                                    print_vector(relation_name, sentence, class_output_file)

                                    negative_training_quota += 1
                                    document_counter[relation_name] += 1

                                    if relation_name !="occupation":
                                        break

                print("Going to Extract NEGATIVE vectors")
                sys.stdout.flush()
                #relations = set(relation_to_entity2.keys()).union(relation_to_entity2.values()) # Relations not used in negative training
                extract_and_print_negative_examples(negative_training_quota)

        print(document_counter)
        sys.stdout.flush()
        ### Allows program to end once quota is reached
        if len(document_counter.values())>0 and not any(document_count < document_quota for document_count in document_counter.values()):
            break


