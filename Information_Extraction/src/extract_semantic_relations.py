#!/opt/python-3.6/bin/python3
# -*- coding: utf-8 -*-

#Benny Longwill
#07/10/2019
#Ling575 Information Extraction Final Project
#Python script that is used to extract semantic relations from sentences

#Script Dependencies
import sys
import spacy
import re
from spacy.parts_of_speech import NOUN, VERB
import textacy.spacier.utils

'''
Params:
    1.) test_sentences_file_path: str - Input vectors of word sequences separated by new line character
    2.) semantic_relation_output_file_path: str - File output path for listing extracted semantic relations to file
    3.) data_is_labeled:bool - Tells program to expect a clsss label at the beginning of each input vector
'''

#Hardcoded file path -- at the time of writing this I couldn't import spacy
sp = spacy.load('/home2/longwill/en_core_web_md/en_core_web_md-2.1.0')

# removes parens and brackets with content using regex
def clean(input):
  clnr = re.compile('\[.*?\]|\(.*?\)|/.*?/')
  clntxt = re.sub(clnr, "", input)
  return clntxt

'''
Aknowledgment: Modified Method From Spacy's Textacy
Textacy.extract's subject_verb_object_triples() method extracts an ordered squence of SVO triples\
 from a spacy-parsed doc adn was the foundation of the current system's extractor.
    This method was modified to allow PP phrase constituents and binary in cases of intransitive verbs
'''
def get_relations(doc):

    sents = doc.sents

    for sent in sents:

        start_i = sent[0].i

        verbs = textacy.spacier.utils.get_main_verbs_of_sent(sent)

        for verb in verbs:
            subjs = textacy.spacier.utils.get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = textacy.spacier.utils.get_objects_of_verb(verb)

            verb_type = check_verb(verb)

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = textacy.spacier.utils.get_span_for_verb_auxiliaries(verb)

            verb = sent[verb_span[0] - start_i : verb_span[1] - start_i + 1]

            #code added to allow for prepositional phrases to be identified with verb
            preps = [child for child in verb.rights if child.dep_=="prep"]

            for subj in subjs:
                subj = sent[
                       textacy.spacier.utils.get_span_for_compound_noun(subj)[0]
                    - start_i : subj.i
                    - start_i
                    + 1
                ]
                if objs:
                    for obj in objs:
                        if obj.pos == NOUN:
                            span = textacy.spacier.utils.get_span_for_compound_noun(obj)
                        elif obj.pos == VERB:
                            span = textacy.spacier.utils.get_span_for_verb_auxiliaries(obj)
                        else:
                            span = (obj.i, obj.i)
                        obj = sent[span[0] - start_i : span[1] - start_i + 1]
                        yield (subj.text, verb.text, obj.text)

                ### The following code was added to include prepositions
                elif len(preps)>0:
                    for prep in preps:
                        if len(list(prep.rights))>0:
                            yield (subj.text, verb.text+" "+ prep.text, " ".join([element.text for element in list(prep.rights)[0].subtree]))

                #Added to include intransitive verbs e.g., I walked
                elif verb_type =="INTRANVERB":
                    right_children=list(verb.rights)

                    if len(right_children)>0 and right_children[0].pos_=="VERB":
                            yield (subj.text, verb.text + " " + right_children[0].text)
                    else:
                        yield (subj.text, verb.text)

#Method that extracts semantic relations from test sentences and writes relations to file

def extract_to_file(test_sentences_file_path:str, semantic_relation_output_file_path:str, data_is_labeled ):
    test_vectors = open(test_sentences_file_path).readlines()
    relation_output_file_path=semantic_relation_output_file_path
    data_is_labeled=bool(eval(data_is_labeled))

    #Keeps a tally of relations by sentence for use in evaluation
    sentence_count_file_path="./output/sentence_file_count.txt"

    with open(relation_output_file_path, 'w') as relation_output_file, open(sentence_count_file_path, 'w') as sentence_count_file :

        #Itterates through test_vector word sequences
        # Make sure there's no empty lines before counting
        for index, vector in enumerate(vec for vec in test_vectors if len(vec) > 0):
            vector=clean(vector)
            if data_is_labeled:
                gold_label, features = vector.split(" ", maxsplit=1)
            else:
                features = vector

            #Spacy parses the features
            parsed_features = sp(features)
            relation_count=0
            #Uses get_relations to extract semantic relations and writes to file
            for relation_index, semantic_relation in enumerate(get_relations(parsed_features)):
                relation_output_file.write(str(index) + "-" +str(relation_index) + ": " + "(" +", ".join(semantic_relation).replace("\n", " ") +")" + "\n")
                relation_count=relation_index+1
            sentence_count_file.write("SemRelCnt:" + str(relation_count)+ " Sentence#:"+ str(index) +"  " + vector + "\n" )
def main():
    test_sentences_file_path= sys.argv[1]
    semantic_relation_output_file_path = sys.argv[2]
    data_is_labeled = sys.argv[3]

    extract_to_file(test_sentences_file_path,semantic_relation_output_file_path , data_is_labeled)

if __name__ == "__main__":
    main()


'''
Acknowledgement: The following code was not written by me, it ws found on StackOverflow 
#Method used to determine the type of verb given a token
'''
def check_verb(token):
    """Check verb type given spacy token"""
    if token.pos_ == 'VERB':
        indirect_object = False
        direct_object = False
        for item in token.children:
            if(item.dep_ == "iobj" or item.dep_ == "pobj"):
                indirect_object = True
            if (item.dep_ == "dobj" or item.dep_ == "dative"):
                direct_object = True
        if indirect_object and direct_object:
            return 'DITRANVERB'
        elif direct_object and not indirect_object:
            return 'TRANVERB'
        elif not direct_object and not indirect_object:
            return 'INTRANVERB'
        else:
            return 'VERB'
    else:
        return token.pos_