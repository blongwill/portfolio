#!/opt/python-3.6/bin/python3
#/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys
import spacy
import re
from spacy.parts_of_speech import NOUN, VERB
import textacy.spacier.utils

#sp=spacy.load('en')
sp = spacy.load('/home2/longwill/en_core_web_md/en_core_web_md-2.1.0')

def clean(input):
    #removes parens and brackets with contents
  cleanr = re.compile('\[.*?\]|\(.*?\)|/.*?/')
  cleantext = re.sub(cleanr, "", input)
  return cleantext

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


def get_relations(doc):
    """
    Extract an ordered sequence of subject-verb-object (SVO) triples from a
    spacy-parsed doc. Note that this only works for SVO languages.

    Args:
        doc (:class:`spacy.tokens.Doc` or :class:`spacy.tokens.Span`)

    Yields:
        Tuple[:class:`spacy.tokens.Span`]: The next 3-tuple of spans from ``doc``
        representing a (subject, verb, object) triple, in order of appearance.
    """
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


            preps = [child for child in verb.rights if child.dep_=="prep"]

            for subj in subjs: #if subj.pos_!="AUX":
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
                elif len(preps)>0:
                    for prep in preps:
                        #print((subj.text, verb.text+" "+ prep.text, " ".join([element.text for element in list(prep.rights)[0].subtree])))
                        if len(list(prep.rights))>0:
                            yield (subj.text, verb.text+" "+ prep.text, " ".join([element.text for element in list(prep.rights)[0].subtree]))
                elif verb_type =="INTRANVERB":
                    right_children=list(verb.rights)

                    if len(right_children)>0 and right_children[0].pos_=="VERB":
                            yield (subj.text, verb.text + " " + right_children[0].text)
                    else:
                        yield (subj.text, verb.text)

#./extract_semantic_relations.py  test_relations relation_output_file_path  data_is_labeled
def extract_to_file(test_sentences_file_path:str, semantic_relation_output_file_path:str, data_is_labeled ):
    test_vectors = open(test_sentences_file_path).readlines()
    relation_output_file_path=semantic_relation_output_file_path
    data_is_labeled=bool(eval(data_is_labeled))

    sentence_count_file_path="./output/sentence_file_count.txt"

    with open(relation_output_file_path, 'w') as relation_output_file, open(sentence_count_file_path, 'w') as sentence_count_file :

        for index, vector in enumerate(vec for vec in test_vectors if len(vec) > 0):  # Make sure there's no empty lines before counting
            vector=clean(vector)
            if data_is_labeled:
                # print(vector.split(" ", maxsplit=1))
                gold_label, features = vector.split(" ", maxsplit=1)
            else:
                features = vector
                gold_label = None

            parsed_features = sp(features)
            relation_count=0
            for relation_index, semantic_relation in enumerate(get_relations(parsed_features)):
                relation_output_file.write(str(index) + "-" +str(relation_index) + ": " + "(" +", ".join(semantic_relation).replace("\n", " ") +")" + "\n")
                relation_count=relation_index+1
            sentence_count_file.write("SemRelCnt:" + str(relation_count)+ " Sentence#:"+ str(index) +"  " + vector + "\n" )
def main():
   #test_sentences_file_path: str, semantic_relation_output_file_path: str, data_is_labeled
    extract_to_file(sys.argv[1] , sys.argv[2] , sys.argv[3])


if __name__ == "__main__":
    main()
