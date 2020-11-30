#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from data_input import get_gold_standard_docs
from svm import build_svm_model, get_svm_best_index

from operator import itemgetter
import numpy as np
import itertools

"""Information ordering for multi-document text summarization. Two different approaches: entity-based & chronological order"""

__author__ = "Amina Venton"
__email__ = "aventon@uw.edu"


def get_entity_grids(sentence_list, nouns_list, num_permutations):
    """
    This function takes in a list of sentence tuples (sent_pos, sent_nouns)
    and list of document nouns and creates an entity grid of the gold document

    Given a number of permutations, it generates and returns the number
    of entity grid permutations of the document and the permutations of the sentence indices

    For training vectors, the first permutation in each array will be the gold summary ordering in the gold document
    For testing vectors, the first permutation in each array will be the ranked summary ordering in the Topic summary
    """

    # List to hold the original summary grid
    # The grid will become n * m
    # Where n = num of sents in doc & m = num of nouns in the doc
    original_summary_entity_grid = []


    # List to hold the original sentence ordering of the doc/summary
    # Indices will be used to grab the sentence strings for ordering
    original_sentence_positions = []

    for sentence in sentence_list:

        # Get individual sentence nouns
        (sent_index, sent_nouns) = sentence

        # Add sentence index to keep track of
        original_sentence_positions.append(sent_index)

        # List to store entity
        sent_row = []

        # Fill out the gold entity grid
        # Nouns for each sentence either:
        # 1 = present 0 = absent
        for noun in nouns_list:

            if noun in sent_nouns:
                sent_row.append(1)
            else:
                sent_row.append(0)

        # Add sentence row of entities to grid
        original_summary_entity_grid.append(sent_row)


    # Get a list of possible sentence orderings in the topic from original sentence indices
    sentence_index_permutations = [list(doc_perm_tuple) for doc_perm_tuple in
                                itertools.islice(itertools.permutations(original_sentence_positions), num_permutations)]

    # Build the entity grid permutations using the original sentence indices and first entity grid
    entity_grid_permutations = []

    for perm in sentence_index_permutations:
        permutation = []
        for i in range(len(perm)):
            sent_index = perm[i]

            entity_grid_cell = original_summary_entity_grid[sent_index]
            permutation.append(entity_grid_cell)

        entity_grid_permutations.append(permutation)


    # Return numpy array of 10  n * m entity grid permutations for the document and the permutations of sentence indices
    return np.array(entity_grid_permutations), sentence_index_permutations


def get_doc_vectors(entity_grid_permutations):
    """
    This function takes in a list of entity grid permutations and generates
    a feature vector representation of each grid

    A list of document feature representations is returned
    """

    # List to store all transition probability vectors
    # First vector is the original gold standard sentence ordering
    doc_vectors = []

    # Iterate through each permutation
    for entity_grid in entity_grid_permutations:

        # Create a sequences dictionary to keep track of counts
        sequences_dict = {"00": 0, "01": 0, "10": 0, "11": 0}

        # Tally up sequences vertically in each grid
        # Transpose entity grid to get sequences horizontally instead
        entity_grid_transpose = entity_grid.transpose()

        # Tally seqs in each row
        for row in entity_grid_transpose:
            for i in range(1, entity_grid_transpose.shape[1]):
                seq = str(row[i - 1]) + str(row[i])
                sequences_dict[seq] = sequences_dict[seq] + 1

        # Create an ordered sequence list for all vectors
        # ["00", "01", "10", "11"]
        sequence_list = sorted(sequences_dict.keys())

        # Get the total transitions, it should be the same for each vector of the document
        total_transitions = sum(sequences_dict.values())

        # Get feature vector representation of each grid
        # Dividing by the total number of transitions will be done last in one step
        feat_vector = []

        for seq in sequence_list:
            feat_vector.append(sequences_dict[seq])

        # Add each feature vector to the list of document vectors
        doc_vectors.append(np.array(feat_vector))

    # Divide doc vectors by the total number of transitions
    doc_vectors = np.array(doc_vectors) / total_transitions

    # Return list of document vectors
    return doc_vectors


def get_doc_data(gold_summ_docs):
    """
    This function takes in a list of gold summary Document objects
    and returns a dictionary of the data.

    Keys are document object and values are list of tuples of (sent, set of nouns)
    for each sent in the doc. {doc_obj: [(sent_index, sent_noun_set)]
    """

    doc_dict = dict()

    for summary_doc in gold_summ_docs:

        # List to hold (sent_position, sent_noun_set) tuples
        doc_list = []
        
        # Get sentence index and set of nouns for each sentence object
        sent_index = 0
        for sent_obj in summary_doc.sentence_list:
            doc_list.append((sent_index, sent_obj.nouns))
            sent_index += 1

        # Add the list of sentence tuples to the dictionary
        doc_dict[summary_doc] = doc_list

    # Return the dictionary
    return doc_dict


def get_training_vectors(gold_summ_docs, num_permutations):
    """
    This function takes in a list of gold summary Document objects
    and makes entity grids and feature vector representations
    for each summary document.

    It returns a list of 2D numpy arrays where each array consists of
    the gold summary feature vector and all of its vector permutations.
    """

    # Master list of 2D feature vector permuations of each document
    all_train_vectors = []

    # Get a dictionary of document data
    doc_dict = get_doc_data(gold_summ_docs)

    # Iterate through all the documents to make grids and vectors
    for document in doc_dict:
        # Get list of all nouns for each document
        all_nouns_set = set()
        all_nouns_sets = [all_nouns_set.union(sent_noun_set) for (sent_ind, sent_noun_set) in doc_dict[document]]
        doc_nouns_list = sorted(list(set.union(*all_nouns_sets)))

        # Get entity grids for variable amount of permutations of the document
        # sentence_index_permutations not used for training
        entity_grid_permutations, sentence_index_permutations = get_entity_grids(doc_dict[document], doc_nouns_list, num_permutations)

        # Get probability transition vectors and feature vector representations for each
        # permutation of the document
        doc_vectors = get_doc_vectors(entity_grid_permutations)

        # Append numpy feature vectors to master list of document vectors
        all_train_vectors.append(doc_vectors)

    # Return list of all feature vectors
    return all_train_vectors


def get_topic_data(topics_with_summaries):
    """
        This function takes in a list of ranked topic objects
        and returns a dictionary of the data.

        Keys are document object and values are list of tuples of (sent, set of nouns)
        for each sent in the doc. {doc_obj: [(sent_index, sent_noun_set)]
        """

    # Master list of 2D feature vector permuations of each document
    all_training_vectors = []

    topic_dict = dict()

    for topic in topics_with_summaries:
        sentences = topic.summary

        # List to hold (sent_pos, sent_noun_set) tuples
        topic_list = []


        for sent_obj in sentences:
            topic_list.append((sentences.index(sent_obj), sent_obj.nouns))

        # Add the list of sentence tuples to the dictionary
        topic_dict[topic] = topic_list

    # Return the dictionary
    return topic_dict


def get_testing_vectors(topics_with_summaries, num_permutations):
    """
    This function takes in a list of ranked topic objects
    and makes entity grids and feature vector representations
    for each summary in the Topic.

    It returns a list of 2D numpy arrays where each array consists of
    its vector permutations.
    """

    # Master list of 2D feature vector permutations of each topic
    all_test_vectors = []

    # Master list of sentence indices
    test_vectors_sentence_indices = []

    # Master list of topic objects in each vector
    topic_objects_of_test_vectors = []

    # Get a dictionary of topic data
    topic_dict = get_topic_data(topics_with_summaries)


    # Iterate through all the topics to make grids and vectors
    for topic in topic_dict:
        # Get list of all nouns for each document
        all_nouns_set = set()
        all_nouns_sets = [all_nouns_set.union(sent_noun_set) for (sent_ind, sent_noun_set) in topic_dict[topic]]
        topic_nouns_list = sorted(list(set.union(*all_nouns_sets)))

        # Get entity grids for variable amount of permutations of the document
        # Add sentence indices for each test vector
        entity_grid_permutations, sentence_index_permutations = get_entity_grids(topic_dict[topic], topic_nouns_list, num_permutations)
        test_vectors_sentence_indices.append(sentence_index_permutations)

        # Get probability transition vectors and feature vector representations for each
        # permutation of the document
        doc_vectors = get_doc_vectors(entity_grid_permutations)

        # Append numpy feature vectors to master list of document vectors
        all_test_vectors.append(doc_vectors)

        # Add topic object to an ordered list that matches the test vectors
        topic_objects_of_test_vectors.append(topic)

    # Return list of all feature vectors and sentence indices
    return all_test_vectors, test_vectors_sentence_indices, topic_objects_of_test_vectors


def build_entity_model(output_folder, num_permutations):
    """
    This function trains the entity model with gold standard summaries
    and a variable amount of permutations
    """

    # Training for Information Ordering
    # Read in gold summary document data
    # Return a list of document objects

    gold_summ_path = "/dropbox/18-19/573/Data/models/training/2009"
    gold_summ_docs = get_gold_standard_docs(gold_summ_path)

    # Generates entity grids and feature representations
    # for all document permutations
    # Returns the list of feature representations in
    # 2D numpy arrays for each document

    all_train_vectors = get_training_vectors(gold_summ_docs, num_permutations)

    # Builds the model to learn the ranking function
    # given all the training vectors and output folder
    # to store the model files
    build_svm_model(all_train_vectors, output_folder)


def order_info_entity(topics_with_summaries, num_permutations, output_folder):
    """
    Entity-based ordering approach
    This function takes in a list of Topic objects with ranked summaries
    generated by select_content(topics) in text_summarizer.py main.

    It uses a coherence model trained on gold summary data to find
    the best ordering of the ranked summaries.

    It returns a list of Topics with the most optimal order based
    on the ranking of the model
    """

    # Build the entity model using SVM rank
    build_entity_model(output_folder, num_permutations)

    # Get test vectors for model
    all_test_vectors, test_vectors_sentence_indices, topic_objects_of_test_vectors = get_testing_vectors(topics_with_summaries, num_permutations)

    # Iterate through every topic object and each test vector 
    # and get the most optimal ordering for each

    for i in range(len(topic_objects_of_test_vectors)):
        topic = topic_objects_of_test_vectors[i]

        test_vector = all_test_vectors[i]

        # Gets the best ranking index in the original test vector
        # for each summary given the test vectors and output folder
        best_index = get_svm_best_index(test_vector, output_folder)

        # Get the sentence indices of the best order
        sentence_indices_of_best_order = test_vectors_sentence_indices[i][best_index]

        # New Sentence objects list for each Topic object summary
        new_summary = []

        # Get current list of sentence objects summary
        current_summary = topic.summary

        # Rearrange ordering of summary based on the best order
        for index in sentence_indices_of_best_order:
            new_summary.append(current_summary[index])

        topic.summary = new_summary

    # Rename list with Topic objects and summaries
    topics_with_summaries = topic_objects_of_test_vectors

    # Return list of Topics with ordered sentences in the Topic summary
    return topics_with_summaries


def order_info_chron(topics_with_summaries):
    """
    Chronological ordering approach
    This function takes in a list of Topic objects with ranked summaries
    generated by select_content(topics) in text_summarizer.py main.

    It orders the Sentence objects chronologically (earliest date first)
    for each Topic summary and returns a list of Topics with chronologically
    ordered summaries
    """

    # Sort original dates in order and then by sentence position
    for topic in topics_with_summaries:
        # Get the list of ranked Sentence objects per Topic object
        sentences = topic.summary

        # Create list of tuples of Sentence objects with their date and position
        # [(sentence_object, sentence_date, sentence_position)]
        sentence_tuples = [(sentence_object, sentence_object.parent_doc.date, sentence_object.index) for sentence_object in sentences]
        # Sort Sentence objects first by date and then by position in the doc
        sentence_tuples.sort(key=itemgetter(1, 2))

        # Get only the Sentence object from each tuple
        sorted_sent = [sen_tup[0] for sen_tup in sentence_tuples]

        # Add chronologically sorted Sentence objects list for each Topic object summary
        topic.summary = sorted_sent

    # Return list of Topics with chronologically ordered sentences in the Topic summary
    return topics_with_summaries
