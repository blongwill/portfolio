#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project
# Script for scraping wikipedia data and extracting semantic relations.
# Creates output files containing either binary or multiclass semantic relations for famous actors.
# Additional processing is added to training output vs. evaluation output

#Script dependencies
import unicodedata
import urllib.request as urllib2
import nltk
from bs4 import BeautifulSoup
import random
import sys
from nltk.corpus import stopwords
from urllib.parse import quote
import re

'''
Program Parameters:
    1.) list_of_entites:str - File containing list of candidate actors' names to search wikipedia for
    2.) class_output_file_path:str - File name used to create output for multiclass classification -- Includes class names relations
    3.) binary_output_file_path:str -File name used to create output for binary classification -- Only includes present/absent relations
    4.) document_quota:int - integer representing the number of documents requested for each semantic relation
    5.) is_training_data:bool - indicates to the program whether data is designated for training, 
        that is whether or not it should be stemmed, certain parts of speech removed, is alphanumeric,

#Eample call to method: ./extract_wiki_data entity_list_file:File class_output_file_name:str binary_output_file_name:str document_quota:int is_training_data:Bool
'''
list_of_entities=open(sys.argv[1]).readlines()
class_output_file_path=sys.argv[2]
binary_output_file_path=sys.argv[3]
document_quota=int(sys.argv[4])
is_training_data = bool(eval(sys.argv[5]))

#List of stop words collected from nltk corpus
stop_words = set(stopwords.words('english'))
#Object used for stemming data
ps = nltk.PorterStemmer()
#Hardcoded html header for scraping website
hdr = {"User-Agent": "My Agent"}



'''
Takes a single sentence or document and the relation name that is represented in the sentence.
Writes to output_file as a single document vector in the format: relation\sfeature:frequency
If used for training data is additionally proccessed to capture the context of the relation. \
#           In training data is stemmed and removes the relation/entities to the relation, specific parts of speech, numbers and stop words
'''
def print_vector(relation_name, sentence, output_file):
    print("Entered Vector print method")
    sys.stdout.flush()

    sentence=sentence.replace("\n", " ").replace("\r","") # Remove white spaces
    current_relation = Relation(relation_name)
    if is_training_data:
        for token in [token[0].lower() for token in nltk.pos_tag(nltk.word_tokenize(sentence)) if
                       token[1] != 'NNP' and token[1] != 'NNPS' and token[1] != 'JJ' and token[1] != 'CD' and 'PRP' not in
                       token[1]]:
            if token not in stop_words and token.isalnum() and token != entity2 and token not in entity_name.lower():
                token = ps.stem(token)
                current_relation.features.update({token: current_relation.features.get(token, 0) + 1})

        if len(current_relation.features)>0:
            output_file.write(current_relation.name)
            for feature, frequency in current_relation.features.items():
                output_file.write(" " + feature + ":" + str(frequency))
            output_file.write("\n")
    else:
        output_file.write(current_relation.name + " " + sentence + "\n")

    output_file.flush()

#Relation wrapper object used to hold a class name and the features
class Relation:
    def __init__(self, name:str=None):
        self.name=name
        self.features={}
    def __repr__(self):
        return self.name

#Method takes input and removes parens and brackets along with their contents using regex
def clean(input):
  clnr = re.compile('\[.*?\]|\(.*?\)|/.*?/')
  clntxt = re.sub(clnr, "", input)
  return clntxt


# Parses beautiful soup object parse and collects infobox table from famous actor wikipedias
# Labels of the table are taken as semantic relations
# Corresponding information is taken as argument data
def extract_table_information(bs):
    #Dictionary that stores relation as class and scraped entity as value
    relation_to_entity2 = {}
    ### Removes citations
    body = bs.find("body")
    for element in body.findAll("sup", {'class': 'reference'}):
        element.decompose()

        table = bs.find("table", {'class': 'infobox'})

        #if a table is found on the page gets the table row elements
        if table:
            table_rows = table.find("tbody").findAll("tr")

            #Itterates the rows of the table
            for row in table_rows:
                for element in row.findAll("table", {'class': 'wikitable'}):
                        element.decompose()

                #Table header and table data
                th = row.find('th')
                td = row.find('td')
                #Both must be present as a relation and second entity to the relation
                if th and td:

                    relation = th.get_text().strip().replace(" ", "_").lower()
                    entity = td.get_text().strip().lower()

                    #Processing for specific relations
                    ### Locations take first attribute which is usually most specific
                    # Other relations that were considered: relation == 'partner(s)' or relation == "resting_place"
                    if relation == "born" or relation == "died" \
                              or relation == 'spouse(s)' or relation == 'education':
                        attributes = td.findAll("a")
                        if len(attributes) > 0:
                            entity = attributes[0].get_text().split(",")[0].lower()
                    #Occupation requires special treatment because usually wikipedia has several occupations
                    elif relation == "occupation":
                        #### Take only the first occupation
                        entity = entity.replace("\n", ",").split(",")[0]

                    #Adds these features to dictionary under class name
                    if relation == "born" or relation == "died" or relation == "occupation" or relation == 'spouse(s)'\
                                or relation == "education":
                            relation_to_entity2.update({relation: clean(entity)})
    return relation_to_entity2

#Given a URL, extracts wikipedia entry using beautiful soup
#Allows for two calls to api, if successful returns beautiful soup html object
def extract_wiki_entry(url:str):
    retries=2
    while retries>0:
        try:
            req = urllib2.Request(url, headers=hdr)
            html = urllib2.urlopen(req)
            bs = BeautifulSoup(unicodedata.normalize("NFKD", html.read().decode('utf-8')).replace('\0','') ,"html.parser")

            if html.getcode() == 200:
                return bs
        except Exception as inst:
            print(inst)
            retries-=1
    return None

#Collects negative samples (randomly selected sentences from random pages that don't contain target relations)
def extract_and_print_negative_examples(negative_training_quota:int):

    bs=extract_wiki_entry("https://en.wikipedia.org/wiki/Special:Random")

    unrelated_text = " ".join([clean(p.get_text().strip()) for p in bs.findAll("p")])

    unrelated_sentences = nltk.sent_tokenize(unrelated_text)

    #Counts down until correct number of samples is obtained
    while int(negative_training_quota) > 0:

        #Gets a new randomly selected article if there aren't any possible sentences
        while len(unrelated_sentences)<1:

            bs=extract_wiki_entry("https://en.wikipedia.org/wiki/Special:Random") ##Gets random wikipedia article

            unrelated_text = " ".join([clean(p.get_text().strip()) for p in bs.findAll("p")])

            unrelated_sentences = nltk.sent_tokenize(unrelated_text)

        #gets randomly selected sentence and removes it from the list
        unrelated_sentence = unrelated_sentences[random.randint(0, len(unrelated_sentences) - 1)]
        unrelated_sentences.remove(unrelated_sentence)

        #prints as a data vector using "absent" as the negative semantic relation class
        print_vector("absent", unrelated_sentence, binary_output_file)

        #decrements the counter
        negative_training_quota -= 1

#Itterates a list of pssible semantic relations and the wikipedia pages from a list of famous actor entities
#Parses infobox for second argument 'entity2' of semantic relation
#Itterates remaining sentences of wikipedia body text for sentences that contain the relation and both entiteies
with open(class_output_file_path, "w", encoding='utf8') as class_output_file, open(binary_output_file_path, "w", encoding='utf8') as binary_output_file:
    document_counter= nltk.defaultdict(lambda:0)

    #Iterates list of entity candidates
    for entity_name in list_of_entities:

        #Program considers the entity to be the first argument of semantic relation
        entity1 = entity_name.strip().replace(" ", "_")

        #Gets beautiful soup html object form wikipedia entry
        bs = extract_wiki_entry("https://en.wikipedia.org/wiki/" + quote(entity1))
        if bs is not None:
            try:
                #Parses beautiful soup object into dictionary of semantic relations with entity 2
                relation_to_entity2 = extract_table_information(bs)
            except Exception as e:
                print(e)
                continue

            ### separates wiki_text into a list of sentences
            wiki_text = " ".join([clean(p.get_text().strip()) for p in bs.findAll("p")])
            wiki_sentences = nltk.sent_tokenize(wiki_text)

            #Must have at least one sentence and relation
            if len(wiki_sentences)>0 and len(relation_to_entity2)>0:
                #Splits the name and includes he or she as possible first entity
                entity1_list = set(entity_name.lower().split() + ['he'] + ['she'])

                negative_training_quota = 0  # Used for making absence the same amoutn of times
                #Itterates the relation and entities from infobox
                for relation_name, entity2 in relation_to_entity2.items():

                    if document_counter[relation_name] < document_quota:

                        #Checking each candidate sentences of the wikipedia body for the semantic relation entity arguments
                        for sentence in wiki_sentences:
                            if sentence:
                                lower_sentence = sentence.lower()

                                if (entity2 in lower_sentence) and any(entity1_list for word in lower_sentence):
                                    #Sentence is found first output present vectors
                                    print_vector('present', sentence, binary_output_file)
                                    print_vector(relation_name, sentence, class_output_file)

                                    negative_training_quota += 1
                                    document_counter[relation_name] += 1

                                    #Only collects one occupation sample per article otherwise an explosion of samples occurs
                                    if relation_name !="occupation":
                                        break
                #Get random sentence from random wiki article
                extract_and_print_negative_examples(negative_training_quota)


        ### Allows program to end once quota is reached
        if len(document_counter.values())>0 and not any(document_count < document_quota for document_count in document_counter.values()):
            break
