#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content Selector for multi-document text summarization that ranks and chooses the sentences to use in the summary."""

__author__ = "Shannon Ladymon, Haley Lepp, Amina Venton"
__email__ = "sladymon@uw.edu, hlepp@uw.ediu, aventon@uw.edu"

from data_input import Topic, Document, Sentence, Token
import math
import numpy as np
from operator import itemgetter


def _cosine_similarity(sentence_1, sentence_2):
    """
    Calculates and returns cosine similarity using raw count tf-idf of two sentence objects
    """
    numerator = 0
    denominator_1 = 0
    denominator_2 = 0
    for i in sentence_1.tf_idf.keys():
        numerator += sentence_1.tf_idf.get(i) * sentence_2.tf_idf.get(i, 0.0)
        denominator_1 += sentence_1.tf_idf.get(i) ** 2
    for i in sentence_2.tf_idf.values():
        denominator_2 += i * i
    denominator = math.sqrt(denominator_1 * denominator_2)
    if denominator != 0:
        return numerator / denominator
    else:
        return denominator  


def _build_sim_matrix(sent_list, intersent_threshold, intersent_formula, mle_lambda, k, topic):
    """
    Builds and returns a 2D numpy matrix of inter-sentential similarity.
    """
    num_sent = len(sent_list)

    # [num_sent] x [num_sent] matrix, defaulting to 0 for each similarity
    sim_matrix = np.zeros((num_sent, num_sent))

    # Fill the 2D numpy matrix with inter-sentential cosine similarity
    for i in range(num_sent):

        sim_vals = []

        # only iterate over top half triangle of matrix
        # since bottom half is identical
        for j in range(i, num_sent):

            if intersent_formula == "norm":
                # Get similarity via normalized generative probability
                sim = _calc_norm_gen_prob(sent_list[i], sent_list[j], mle_lambda, topic)

                # Keep track of the similarity values to rank & filter later
                sim_vals.append((i, j, sim))
            else:
                # Default is to use cosine similarity

                # Get the similarity for different sentences
                if i != j:
                    sim = _cosine_similarity(sent_list[i], sent_list[j])

		    # Only include similarities above the threshold
                    if sim >= intersent_threshold:
                        sim_matrix[i][j] = sim
                        sim_matrix[j][i] = sim

                else:
                    # If the same sentence, cosine sim is 1.0
                    sim_matrix[i][i] = 1.0

        # For norm similarity, add only the top k similarities for sentence i 
        if intersent_formula == "norm":
            sorted_sim_vals = sorted(sim_vals, key=itemgetter(2), reverse=True)[:k]
            for i, j, sim in sorted_sim_vals:
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

    # Normalize by dividing by sum of each row
    row_sums = sim_matrix.sum(axis=1, keepdims=True)

    sim_matrix = sim_matrix / row_sums

    return sim_matrix


def _build_bias_vec(sent_list, topic_sent, include_narrative, bias_formula, mle_lambda, topic):
    """
    Builds and returns a 1D numpy vector of the similarity between each sentence
    and the topic title. Additionally adds the similarity of the narrative if include_narrative.
    """

    # Grab the narrative, which is sometimes located in topic.category
    narrative = topic.narrative
    if narrative is None:
        narrative = topic.category

    # Holds the similarity for each sentence with the topic descritpion
    num_sent = len(sent_list)

    #1D matrix to hold the similiarity between each sentence and the topic
    bias_vec = np.zeros(num_sent) 

    # Get similarity for each sentence and the topic
    for i in range(num_sent):

        # Use the specified bias formula to calculate bias weight
        if bias_formula == "rel":

            # Get the relevance of the topic to this sentence
            bias_vec[i] = _calc_relevance(sent_list[i], topic_sent, topic)

            # Add the relevance of the narrative if needed
            if include_narrative:
                bias_vec[i] +=  _calc_relevance(sent_list[i], narrative, topic)

        elif bias_formula == "gen":

            # Get the generative probability of the topic for this sentence
            bias_vec[i] = _calc_gen_prob(topic_sent, sent_list[i], mle_lambda, topic)

            # Add the generative probability of the narrative if needed
            if include_narrative:
                bias_vec[i] +=  _calc_gen_prob(narrative, sent_list[i], mle_lambda, topic)
        else:

            # Default is cosine similarity
            bias_vec[i] = _cosine_similarity(sent_list[i], topic_sent)

            # Add the similarity of the narrative if needed
            if include_narrative:
                bias_vec[i] +=  _cosine_similarity(sent_list[i], narrative)

    # Normalize by dividing by the sum
    bias_sum = np.sum(bias_vec)

    # If the bias sum is 0, change to 1 to avoid division by 0
    if bias_sum == 0:
        bias_sum = 1

    bias_vec = bias_vec / bias_sum

    return bias_vec


def _calc_relevance(sent, topic_sent, topic):
    "Calculates relevance for two sentences."

    rel_sum = 0

    # Calculate the relevance based on the tf raw count for each word in 
    for i in topic_sent.raw_counts.keys():
        rel_sum += math.log(sent.raw_counts.get(i, 0.0) + 1) * math.log(topic_sent.raw_counts.get(i) + 1) * topic.idf.get(i) 

    return rel_sum


def _calc_smoothed_mle(word, sent, mle_lambda, topic):
    """
    Calculates the smoothed MLE (Maximum Likelihood Estimate) for a word
    in a sentence, smoothing with the tf values from the entire topic cluster.
    Returns the smoothed MLE value.
    """
    return ((1 - mle_lambda)* sent.tf_norm_values.get(word, 0)) + (mle_lambda * topic.tf_norm_values[word])


def _calc_gen_prob(sent_1, sent_2, mle_lambda, topic):
    """
    Calculates and returns the generative probability of sent_1 given sent_2.
    """
    gen_prod = 1
    for word in sent_1.raw_counts:
        gen_prod *= ( _calc_smoothed_mle(word, sent_2, mle_lambda, topic) ** sent_1.raw_counts[word] )

    return gen_prod

def _calc_norm_gen_prob(sent_1, sent_2, mle_lambda, topic):
    """
    Calculates and returns the length-normalized generative probability of sent_1 given sent_2.
    """
    sent_1_len = sum([count for count in sent_1.raw_counts.values()])

    return _calc_gen_prob(sent_1, sent_2, mle_lambda, topic) ** (1.0 / sent_1_len)


def _build_markov_matrix(sim_matrix, bias_vec, d):
    """
    Builds and returns the markov matrix (matrix to multiply by in the power method)
    using the Biased LexRank formula with cosine similarity.
    """
    markov_matrix = (d * (bias_vec)) + ((1-d) * (sim_matrix))
    return markov_matrix


def _power_method(markov_matrix, epsilon):
    """
    Uses the power method to find the LexRank values of each sentence
    and returns a 1D vector of the final LexRank values after convergence.
    """

    num_sent = len(markov_matrix)

    # Transpose the markov_matrix for power method calculations
    transition_matrix = markov_matrix.T 

    # Create an original vector of sentence probabilies
    # using uniform distribution
    prob_vec = np.ones(num_sent)/num_sent

    # Iterate through power method until convergence
    matrix_diff = 1.0
    while matrix_diff > epsilon:

        # (M^T)(v) for power method
        updated_prob_vec = np.dot(transition_matrix, prob_vec)
  
        # Get amount that the updated_prob_vec has changed
        # from the previous iteration, to see if converged
        matrix_diff = np.linalg.norm(np.subtract(updated_prob_vec, prob_vec))
        prob_vec = updated_prob_vec

    # Return the prob_vec, which is the LexRank scores
    return prob_vec


def _select_sentences(sorted_sentences, summary_threshold):
    """ 
    Takes a list of sentences sorted by LexRank value (descending)
    and selects the sentences to add to the summary greedily based on LexRank value
    while excluding sentences with cosine similarity >= summary_threshold (default = 0.5)
    to any sentence already in the summary.
    Returns a list of selected sentences.
    """

    max_summary_size = 100

    # array of added sentence objects to compare for cosine similarity
    added_sents = []

    # Track the number of tokens so far in the summary
    summary_size = 0

    for sent_index in range(len(sorted_sentences)):

        # add first ranked sentence
        if sent_index == 0:
            added_sents.append(sorted_sentences[sent_index])
               
            # update summary size
            summary_size += sorted_sentences[sent_index].sent_len

        # look at all other sentences            
        else:

            # check if summary hasn't gone over the limit
            if summary_size + sorted_sentences[sent_index].sent_len <= max_summary_size:
                
                # get cos similarity of current sentence with the sentences already added
                cos_sim = [_cosine_similarity(sorted_sentences[sent_index], added_sent) for added_sent in added_sents]          
                            
                # check if any cos sim is at or above the summary_threshold
                similar = any(cos_similarity >= summary_threshold for cos_similarity in cos_sim)

                # if sentence is not similar to any of the already added sentences, add to list
                if not similar:
                    added_sents.append(sorted_sentences[sent_index])

                    # update summary size
                    summary_size += sorted_sentences[sent_index].sent_len

    # Return the list of chosen sentences
    return added_sents


def select_content(topics_list, d = 0.7, intersent_threshold = 0.0, summary_threshold = 0.5, epsilon = 0.1, mle_lambda = 0.6, k = 20, min_sent_len = 5, include_narrative = False, bias_formula = "cos", intersent_formula = "cos"):
    """
    For each topic, creates summaries of <= 100 words (full sentences only) 
    using a Biased LexRank similarity graph algorithm
    with tf-idf cosine similarity and a bias for query topic.

    Args:
        topic_list: a list of Topic objects (which include Documents and Sentences)
        d: damping factor, amount to prioritize topic bias in Markov Matrix
        intersent_threshold: minimum amount of similarity required to include in Similarity Matrix
        summary_threshold: maximum amount of similarity between sentences in summary
        epsilon: minimum amount of difference between probabilities between rounds of power method
        mle_lambda: amount to prioritize topic MLE over sentence MLE 
        k: maximum number of intersentential similarity nodes to connect when doing normalized generation probability
        min_sent_len: minimum number of words in a sentence to be used in the summary
        include_narrative: True if the narrative (in addition to title) should be in the bias
        bias_formula: which formula to use for sentence-topic similarity weighting - cos (cosine similarity), rel (relevance), or gen (generation probability)
        intersent_formula: which formula to use for inter-sentential similarity weighting - cos (cosine similarity) or norm (normalized generation probability)

    Returns:
        topic_list: the modified topic_list from the input, with a list of selected sentences
        in the topic.summary fields of each topic.

    """
    # A dictionary of topics with a list of the (sentence, date) 
    # which are chosen for the summary (<= 100 words)
    topic_summaries = {}
    
    # For each topic, choose the top LexRanked sentences
    # up to 100 words
    for topic in topics_list:

        topic_id = topic.topic_id
        topic_title = topic.title
        topic_docs_list = topic.document_list

        # Get a list of all the sentence objects in this topic
        # Don't include sentences that are less than 5 words
        total_sentences = [sent for doc in topic.document_list for sent in doc.sentence_list if sent.sent_len >= min_sent_len]

        # Build the inter-sentential cosine similarity matrix
        sim_matrix = _build_sim_matrix(total_sentences, intersent_threshold, intersent_formula, mle_lambda, k, topic)

        # Build the topic-sentence bias vec
        bias_vec = _build_bias_vec(total_sentences, topic_title, include_narrative, bias_formula, mle_lambda, topic)

        # Build a Markov Matrix using the inter-sentential and bias similarities
        markov_matrix = _build_markov_matrix(sim_matrix, bias_vec, d)

        # Get the Biased LexRank for the sentences using the power method
        lex_rank_vec = _power_method(markov_matrix, epsilon)

        # Add the lex_rank to each sentence.score
        for i in range(len(total_sentences)):
            total_sentences[i].score = lex_rank_vec[i]

        # Sort the sentences by score
        sorted_sentences = sorted(total_sentences, reverse=True)

        # Select which sentences to use
        # and add the list to this topic's summary variable
        topic.summary = _select_sentences(sorted_sentences, summary_threshold)

    # Return the list of topics now that the summary has been added to each
    return topics_list
