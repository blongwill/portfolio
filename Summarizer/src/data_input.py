#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Benny Longwill"
__email__ = "longwill@uw.edu"


from nltk import word_tokenize, sent_tokenize, pos_tag, PorterStemmer

from nltk.corpus import stopwords

from bs4 import BeautifulSoup
import document_retriever
from content_realization import get_compressed_sentences
from math import log
import os  # os module imported here to open multiple files at once
import spacy

spacy_parser = spacy.load('/home/longwill/en_core_web_md/en_core_web_md-2.1.0')
stop_words = set(stopwords.words('english'))


#### Put none for non-existing text
###############################
class Topic:
    idf_type=None #### Default value
    tf_type=None
    def __init__(self, topic_id:str=None ,docsetA_id:str=None,  title:str=None , narrative:str=None, category=None):
        self.doc_count = 0
        self.sent_count = 0
        self.topic_id=topic_id
        self.docsetA_id=docsetA_id
        self.raw_counts = {}  ##### Used for counts across all sentences
        self.tf_norm_values= {}
        ############### Creates and adds title and narrative and category Sentence Objects To Topic
        self.title = Sentence.create_sentence(self, title)
        self.narrative = Sentence.create_sentence(self, narrative) ##### *** Not all topics have this attribute ***
        self.category=Sentence.create_sentence(self,category) ##### *** Not all topics have this attribute ***
        ##########################################################
        self.document_list=[]
        self.summary = []
        self.idf={}


    #Returns list of all sentences in Topic, including Title sentence, narrative sentences, and headlines
    def all_sentences(self)->list:

        total_sentences=[sentence for document in self.document_list for sentence in document.sentence_list]

        # checks the title in the topic
        if self.title:
            total_sentences.append(self.title)
        # checks the narrative in the topic
        if self.narrative:
            total_sentences.append(self.narrative)

        if self.category:
            total_sentences.append(self.category)

            # checks all headlines if available in every document in the topic
        total_sentences += [document.headline for document in self.document_list if document.headline]

        return total_sentences

    ##### Counts the sentences under this topic that contains a token parameter
    def n_containing(self, token):
        # checks all tokens in each sentence of every document in the topic
        count = sum(1 for document in self.document_list for sentence in document.sentence_list if
            token in sentence)

        # checks all headlines if available in every document in the topic
        count += sum(1 for document in self.document_list if document.headline if token in document.headline)

        # checks the title in the topic
        if self.title and token in self.title:
            count+=1
        # checks the narrative in the topic
        if self.narrative and token in self.narrative:
            count+=1

        return count


    #Take the ratio of the total number of documents to the number of documents containing any word.
    #Then it adds 1 to the devisor to avoid zero division and then takes the log otherwise the weight will be too high
    def get_smooth_idf(self, token):
        if token in self.idf:
           return self.idf[token]
        else:
            current_idf = log(self.sent_count / (1 + self.n_containing(token)))
            self.idf[token]=current_idf
            return current_idf

    #AKA unary IDF score, i.e. don't use IDF
    def get_unary_idf(self):
        return 1

    #he logarithm of the number of documents in the corpus divided by the number of documents the term appears
    # in (this will lead to negative scores for terms appearing in all documents in the corpus)
    def get_standard_idf(self, token):
        if token in self.idf:
            return self.idf[token]
        else:
            current_idf = -log(self.n_containing(token)/self.sent_count)

            self.idf[token] = current_idf
            return current_idf

    #similar to inverse but substracting the number of documents the term appears in from the
    ## total number of documents in the training corpus (this can lead to positive and negative scores)
    def get_probabilistic_idf(self, token):
        if token in self.idf:
            return self.idf[token]
        else:
            current_idf = log(self.sent_count-self.n_containing(token) / self.n_containing(token))
            self.idf[token] = current_idf
            return current_idf

    # Must be used after all Documents, Sentences, and Tokens have been filled.
    def compute_tf_idf(self):

        cluster_count=sum(self.raw_counts.values())

        #for sentence in doc.sentence_list + [self.title, self.narrative,  doc.headline]:
        for sentence in self.all_sentences():
            if sentence:

                for token in sentence.token_list:
                    token_value=token.token_value

                    if token_value not in self.tf_norm_values.keys():
                        raw_token_count=self.raw_counts.get(token_value)
                        try:
                            self.tf_norm_values.update({token_value: eval(
                                'Sentence.get_' + Topic.tf_type + '(cluster_count=cluster_count,raw_token_count=raw_token_count)')})
                        except:
                            self.tf_norm_values.update({token_value: Sentence.get_term_frequency(cluster_count=cluster_count, raw_token_count=raw_token_count)})


                    try:
                        idf=eval('self.get_' + Topic.idf_type +'(token_value)')
                    except:
                        idf=self.get_smooth_idf(token_value)

                    sentence.tf_idf[token_value] = token.raw_count * idf #self.get_standard_idf(token)
                    sentence.tf_idf_norm[token_value] = sentence.tf_norm_values[token_value] * idf

class Document:
    def __init__(self, parent_topic:Topic=None , doc_id:str=None, headline:str=None,date:str=None, category:str=None, document_text:str=None):
        self.parent_topic = parent_topic
        self.sent_count = 0
        self.doc_id=doc_id
        self.date=date
        self.category=category ##### *** Not all topics have this attribute ***
        self.sentence_list = self.create_sentence_list(document_text)
        ############### Creates and adds headline Sentence Objects To Topic
        self.headline = Sentence.create_sentence(self,headline) ##### *** Not all topics have this attribute ***

        if parent_topic:
            self.parent_topic.doc_count+=1 ### Increments parent count When initialized

    def __repr__(self):
        return str(self.sentence_list)
    # Less than method allows for sorting
    def __lt__(self, other):
        return (self.date < other.date)

    # Takes Document object and the text from doc file. The block of text is separated into sentences as sentence objects and also tokenized using NLTK.
    def create_sentence_list(self, doc_text)->list:
        sentence_list=[]

        for doc_sentence in sent_tokenize(doc_text):

            # Get compressed versions of the original sentence
            compressed_sentences = get_compressed_sentences(doc_sentence, spacy_parser, Document.remove_header, Document.remove_parens, Document.remove_quotes, Document.remove_appos, Document.remove_advcl, Document.remove_relcl, Document.remove_acl)
            # Add a sentence object for each compressed sentence
            for sent in compressed_sentences:

                current_sentence=Sentence(self, sent)  #Creates sentence object
                current_sentence.index=len(sentence_list)
                sentence_list.append(current_sentence)

        return sentence_list

#All sentences are part of a document of either Topic or Document type
class Sentence:
    ps = PorterStemmer()

    stemming=False
    lower=False

    def __init__(self,parent_doc:Document or Topic, original_sentence:str):
        self.score=0
        self.parent_doc=parent_doc
        self.original_sentence = original_sentence.replace("\n", " ").strip().replace("  ", " ")
        self.nouns=set()
        self.sent_len = original_sentence.count(" ") + 1   #Counts words in original sentence
        self.raw_counts = {}
        self.tf_norm_values={}
        self.token_list=self.create_token_list(self.original_sentence)  ##### *** List of non-duplicate Tokens as Objects ***
        self.tf_idf={}
        self.tf_idf_norm={}

        # Increments parent count when initialized. If parent_doc is Document, then Topic sent_count is also incremented
        self.parent_doc.sent_count+=1
        if type(self.parent_doc) is Document and self.parent_doc.parent_topic:
            self.parent_doc.parent_topic.sent_count+=1
            self.index = -1

    def __repr__(self):
        return self.original_sentence
    # Less than method allows for sorting
    def __lt__(self, other):
        return (self.score < other.score)
    def __eq__(self, other):
        return (self.score == other.score)
    # allows the use of 'is' operator on Sentence object
    def __contains__(self, param):
        return any(token.token_value == param for token in self.token_list)

    ### Wrapper initializer that includes validation
    @classmethod
    def create_sentence(cls, self:Topic or Document, original_sentence: str):
        if original_sentence:
            sentence = Sentence(self, original_sentence)
        else:
            sentence = None
        return sentence

    @classmethod
    def get_term_frequency(cls,cluster_count,raw_token_count):
        return raw_token_count / cluster_count
    @classmethod
    def get_raw_count(cls,tokenized_sentence,token):
        return tokenized_sentence.count(token)
    @classmethod
    def get_log_normalization(cls,cluster_count, raw_token_count):
        return log(1+ raw_token_count )

    ##### I think this would be good to be implemented but reuqires extra planning
    #def get_augmented_frequency(self,tokenized_sentence,token):
    #    return .5 + (.5 *tokenized_sentence.count(token)/ argmax word count in sentence    )



    # Tokenizes a sentences and populates the sentence object with a token object list
    def create_token_list(self, original_sentence: str)->list:

        # Edits the original sentence to remove article formatting
        # This location was chosen so that all sentences that are tokenized (i.e., Title sentence) could also be effected by this
        token_list = []

        '''LOTS OF TOKEN LOOPING WITHIN SENT REQUIRED BECAUSE OF POS TAGGING'''
        #Tokenize sentence
        tokenized_sent = word_tokenize(original_sentence)

        #Must tokenize sentence before pos tagging
        #Captures only Nouns and adds to self.nouns set
        [self.nouns.add(token) for token,pos in pos_tag(tokenized_sent) if 'NN' in pos]

        #Lowercasing before pos tagging affects tags, so must do after as list
        if Sentence.lower:
            tokenized_sent = [token.lower() for token in tokenized_sent]

        #Can only stem one word at a time, can't stem whole sentence
        if Sentence.stemming:  ##### THe stemmer also lowercases
            tokenized_sent = [Sentence.ps.stem(token) for token in tokenized_sent]

        for token in set(tokenized_sent):
            ''''######### Removes stop words #########'''''
            if token not in stop_words:

                raw_token_count = Sentence.get_raw_count(tokenized_sent, token)

                #Adds raw count to sentence
                self.raw_counts.update({token:raw_token_count})

                try:
                    tf_norm = eval('Sentence.get_' + Topic.tf_type + '(cluster_count=len(tokenized_sent),raw_token_count=raw_token_count)')
                except:
                    tf_norm = Sentence.get_term_frequency(len(tokenized_sent), raw_token_count)

                #Adds normalalized tf to sentence
                self.tf_norm_values.update({token: tf_norm})

                ##### Adds tokens raw counts to Parent Topic
                if type(self.parent_doc)==Topic:
                    current_topic=self.parent_doc
                else:
                    current_topic=self.parent_doc.parent_topic

                current_topic.raw_counts.update({token: current_topic.raw_counts.get(token,0)+raw_token_count})
                ###############################################

                token_list.append(Token(self, token, raw_token_count ,tf_norm))

        return token_list


class Token:
    def __init__(self,parent_sentence:Sentence, token_value:str, raw_count, tf_norm):
        self.raw_count=raw_count
        self.tf_norm=tf_norm
        self.parent_sentence=parent_sentence
        self.token_value=token_value
    def __repr__(self):
        return self.token_value

###############################
### Tag Variables used to avoid hardcoding later in the document or in case of tag changes

title_tag='title'
narrative_tag='narrative'
topic_category_tag='category'
docsetA_tag='docseta'
id_tag='id'
doc_tag='doc'
topic_tag='topic'
parser_tag='lxml'
categories_file_tag='/categories.txt'
headline_tag='headline'

###############################

# Takes a file path, collects all data and stores into a list of class object 'Topic' data structures
def get_data(file_path:str, stemming:bool, lower:bool, idf_type, tf_type, remove_header, remove_parens, remove_quotes, remove_appos, remove_advcl, remove_relcl, remove_acl)->list:
    """Extracts database documents and creates data structure objects to hold them. Returns a list of Topic objects

    Args:
        file_path:str file path on patas that leads to directory that holds training or testing data
        stemming:bool True enables each sentence to be stored with a stem representation in objects and tokens, False does nothing
        lower:bool True enables each sentence to be stored in lower case, False does nothing.
        idf_type:str String input dictates idf representation in objects. Options are: 'smooth_idf', 'probabilistic_idf' , 'standard_idf' , and 'unary_idf'
        tf_type:str String input dictates tf representation in objects. Options are: 'term_frequency', 'log_normalization'
        remove_header: True if the header should be removed from the sentence
        remove_parens: True if parenthetical information should be removed from the sentence
        remove_quotes: True if unpaired quotes should be removed from the sentence
        remove_appos: True if appositional modifier should be removed from the sentence
        remove_advcl: True if adverbial clause modifier should be removed from the sentence
        remove_relcl: True if relative clause modifier should be removed from the sentence
        remove_acl: True if a finite or non-finite clausal modifier shoule be removed from the sentence

    """

    # Set all the hyperparamaters
    configure_class_objects(stemming, lower, idf_type, tf_type, remove_header, remove_parens, remove_quotes, remove_appos, remove_advcl, remove_relcl, remove_acl)

    with open(file_path) as f1:
        task_data = f1.read()

        soup = BeautifulSoup(task_data, parser_tag)
        raw_topics = soup.findAll(topic_tag)

    return get_topics_list(raw_topics, get_categories(file_path))

# unary_idf smooth_idf standard_idf probabilistic_idf
def configure_class_objects(stemming:bool,lower:bool, idf_type:str, tf_type:str, remove_header, remove_parens, remove_quotes, remove_appos, remove_advcl, remove_relcl, remove_acl):

    if stemming:
        Sentence.stemming = stemming
    if lower:
        Sentence.lower=lower
    if idf_type:
        Topic.idf_type=idf_type
    if tf_type:
        Topic.tf_type=tf_type

    Document.remove_header = remove_header
    Document.remove_parens = remove_parens
    Document.remove_quotes = remove_quotes
    Document.remove_appos = remove_appos
    Document.remove_advcl = remove_advcl
    Document.remove_relcl = remove_relcl
    Document.remove_acl = remove_acl

def get_categories(file_path:str):

    try:
        with open(file_path.rsplit("/",maxsplit=1)[0] + categories_file_tag) as f2:
            categories=f2.read().replace(":","").replace("(","").replace(")","").replace("/"," ")
            #Splits on two empty lines in a row
            bag_of_words={}
            for category in categories.split("\n\n\n")[1:]:
                lines=category.strip().split("\n")
                index, words = lines[0].split(" ", maxsplit= 1)
                index=index.rstrip(".")

                bag_of_words.update({index : words.strip().split(" ")})
                for line in lines[1:]:
                    bag_of_words[index]+=line.split()[1:]

            return bag_of_words

    except FileNotFoundError:
        return None


#Takes a Topic class object, an xml or html document set element, and a document retriever object
# Itterates all document Id's in html/xml element and uses the doc retriever to get the raw document from database
#Extracts attributes from raw document and creates Document class object, fills it with sentence objects and adds it to the current topic object
def populate_document_list(current_topic, docsetA, doc_ret:document_retriever.Document_Retriever):

    for doc in docsetA.findAll(doc_tag):

        doc_id = doc.attrs[id_tag]

        raw_doc = doc_ret.retrieve_doc(doc_id)
        headline, category, dateline, doc_text = get_doc_attributes(raw_doc, doc_ret.headline_tag, doc_ret.category_tag, doc_ret.dateline_tag, doc_ret.text_tag)

        current_doc = Document(parent_topic=current_topic, doc_id=doc_id, date=doc_ret.date,headline=headline, category=category, document_text=doc_text)  ########## Creates document object

        current_topic.document_list.append(current_doc)

# Takes the raw topic xml/html and a set of title, narrative, and docset TAGS according to the format of file
# extracts the respective text including document IDs for Aqcuaint(2) database
def get_topic_attributes(raw_topic, title_tag:str , narrative_tag:str, topic_category_tag:str, docsetA_tag:str):

    narrative=None
    title=None
    topic_category=None

    topic_id = raw_topic.attrs[id_tag]

    if title_tag:
        title_element = raw_topic.find(title_tag)
        if title_element is not None:
            title = title_element.text

    if narrative_tag:
        narrative_element = raw_topic.find(narrative_tag)
        if narrative_element is not None:
            narrative = narrative_element.text

    if topic_category_tag:
        category_element=raw_topic.attrs.get(topic_category_tag)
        if category_element is not None:
            topic_category=category_element



    docsetA = raw_topic.find(docsetA_tag)

    docsetA_id = docsetA.attrs[id_tag]

    return topic_id, title, narrative, docsetA_id, docsetA, topic_category

# Requires raw document and the specific tags used according to current HTML or XML.
# Extracts the headline category dateline and text if available in raw document.
def get_doc_attributes(raw_doc, headline_tag:str="", category_tag:str="", dateline_tag:str="", text_tag:str=""):

    ### Instantiate variables in case text not available in raw doc
    headline = None
    category = None
    dateline = None
    doc_text = None



    if headline_tag:
        headline_element = raw_doc.find(headline_tag)
        if headline_element is not None:
            headline = headline_element.text


    if category_tag:
        category_element = raw_doc.find(category_tag)
        if category_element is not None:
            category = category_element.text


    if dateline_tag:
        dateline_element = raw_doc.find(dateline_tag)
        if dateline_element is not None:
            dateline = dateline_element.text


    if text_tag:
        doc_text=' '.join(raw_doc.find(text_tag).itertext())


    return headline, category, dateline, doc_text

#Parses Raw data XML file argument and extracts the topic elements.
# Creates Topic elements and fills withtopic attributes and Document objects
# Returns these topic objects as a list.
def get_topics_list(raw_topics, topic_categories)->list:
    topics=[]
    doc_ret = document_retriever.Document_Retriever()

    for raw_topic in raw_topics:
        topic_id, title, narrative, docsetA_id, docsetA, topic_category = get_topic_attributes(raw_topic, title_tag, narrative_tag,topic_category_tag, docsetA_tag)

        if topic_category:
            topic_category = " ".join(topic_categories[topic_category])

        current_topic= Topic(topic_id = topic_id,docsetA_id = docsetA_id, title = title, narrative = narrative, category=topic_category) ########### Creates topic object

        populate_document_list(current_topic, docsetA, doc_ret)

        current_topic.compute_tf_idf()

        topics.append(current_topic)

    return topics

###############################


'''''''''''''''''''''''''''''''''''''''''''''
Method used to create dummy data structures for testing purposes : Requires formatted Topic file
'''''''''''''''''''''''''''''''''''''''''''''
def build_pseudo_topic(pseudo_document_file_path, stemming:bool=False, lower:bool=False, idf_type:str=None, tf_type:str=None):

    configure_class_objects(stemming,lower, idf_type, tf_type)

    #```doc_id = 1a \n date = 20110506 \n ### \n Sentence 1 would be here. \n Sentence 2 would be here, etc.```
    # (where ### is a metadata separator and everything below that would be a sentence on its own line)
    with open(pseudo_document_file_path) as f1:
        #Parses whole document by empty line
        topic_meta_data, pseudo_docs=f1.read().split("\n\n", maxsplit=1)


        #Puts topic meta-data into dictionary
        topic_meta_data = dict((x.strip(),y.strip()) for x,y in [data.split("=") for data in topic_meta_data.split("\n")])

        #Loads empty topic object with topic metadata from dictionary
        current_topic=None
        try:
            current_topic=Topic(**topic_meta_data)
        except Exception as e:

            print(e)

        #Separates documents then document metadata from sentences and loads them into a dictionary
        #Calls populate sentences to finish filling remaining data structures
        for doc in pseudo_docs.split("\n\n"):
            doc_meta_data, sentences=doc.split("###")
            doc_meta_data=dict([data.split("=") for data in doc_meta_data.replace(" ","").split()])


            try:
                current_doc = Document(parent_topic=current_topic, document_text=sentences.strip().replace("\n", " "),**doc_meta_data)
                current_topic.document_list.append(current_doc)
            except Exception as e:
                print(e)


        current_topic.compute_tf_idf()


        return current_topic


'''''''''''''''''''''''''''''''''''''''''''''
Method used to create dummy data structures for the Gold Standard data
'''''''''''''''''''''''''''''''''''''''''''''
def get_gold_standard_docs(file_path:str)->list:
    return [Document(parent_topic=Topic(),document_text=open(file_path+"/"+file_name).read()) for file_name in os.listdir(file_path)]
