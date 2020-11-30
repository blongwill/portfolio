#!opt/python-3.6/bin/python3
# -*- coding: utf-8 -*-

"""Unit tests for content_selection.py"""

__author__ = "Shannon Ladymon"
__email__ = "sladymon@uw.edu"

import unittest
import math
import numpy as np
import sys
sys.path.append("../src")
import content_selection
from data_input import Topic, Document, Sentence, Token
from data_input import build_pseudo_topic

class TestContentSelection(unittest.TestCase):

	def setUp(self):

		# Set up values for parameters
		self.d = 0.7
		self.threshold = 0.15
		self.epsilon = 0.1

		# Get a topic from the pseudo_topic.txt file
		self.topic = build_pseudo_topic('pseudo_topic_content_selection.txt')
		self.sent_list = [sent for doc in self.topic.document_list for sent in doc.sentence_list]


	def test_cosine_similarity(self):
		
		# Test that a sentence with itself has sim=1
		sim_same_sentence = content_selection._cosine_similarity(self.sent_list[0], self.sent_list[0])
		self.assertEqual(sim_same_sentence, 1)


		# Test that sentences with more words in common have greater similarity
		# Using the test: s0 ("This is sentence 1.") and s1 ("This is sentence 2.")
		# have greater similarity than s0 and s2 ("Number 3 is completely different!")
		sim_s0_s1 = content_selection._cosine_similarity(self.sent_list[0], self.sent_list[1])
		sim_s0_s2 = content_selection._cosine_similarity(self.sent_list[0], self.sent_list[2])
		self.assertTrue(sim_s0_s1 > sim_s0_s2)


	def test_build_sim_matrix(self):

		# Test that when no sentences, returns an empty np matrix (size==0)
		sim_matrix_none = content_selection._build_sim_matrix([], self.threshold)
		self.assertEqual(sim_matrix_none.size, 0)

		# Test that When 1 sentence, returns an np matrix with a value of 1
		sim_matrix_one = content_selection._build_sim_matrix([self.sent_list[0]], self.threshold)
		self.assertEqual(sim_matrix_one[0][0], 1.0)

		# Test that a matrix with two different sentences 
		# is identical to its transpose (top triangle should be the same as bottom)
		sim_matrix_two = content_selection._build_sim_matrix([self.sent_list[0], self.sent_list[1]], self.threshold)
		transpose_sim_matrix_two = sim_matrix_two.T
		self.assertTrue(np.array_equal(sim_matrix_two, transpose_sim_matrix_two))

		# Test that a matrix with two different sentences
		# has rows that add up to one 
		row_sums = np.sum(sim_matrix_two, axis=1)
		one_vec = np.array([1.0, 1.0])
		self.assertTrue(np.array_equal(row_sums, one_vec))


	def test_build_bias_vec(self):

		# Test that when no sentences, returns an empty vector (size==0)
		bias_vec_none = content_selection._build_bias_vec([], self.topic.title)
		self.assertEqual(bias_vec_none.size, 0)

		# Test that all bias vectors sum to 1

		# One identical sentence
		bias_vec_identical = content_selection._build_bias_vec([self.topic.title], self.topic.title)
		self.assertEqual(np.sum(bias_vec_identical), 1.0)

		# One different sentence
		bias_vec_one = content_selection._build_bias_vec([self.sent_list[0]], self.topic.title)
		self.assertEqual(np.sum(bias_vec_one), 1.0)

		# Two different sentences
		bias_vec_two = content_selection._build_bias_vec([self.sent_list[0], self.sent_list[1]], self.topic.title)
		self.assertEqual(np.sum(bias_vec_two), 1.0)

	
	def test_build_markov_matrix(self):

		# Test that when empty sim_matrix and bias_vec, returns an empty matrix
		sim_matrix_none = np.zeros((0,0))
		bias_vec_none = np.zeros(0)
		markov_matrix_none = content_selection._build_markov_matrix(sim_matrix_none, bias_vec_none, self.d)
		self.assertEqual(markov_matrix_none.size, 0)

		# Test that when sim_matrix has one element of 1 and bias_vec has 0
		# the resulting value is the same as 1 - d
		expected_m_m_one = 1 - self.d
		sim_matrix_one = np.array([1.0])	
		bias_vec_one_empty = np.zeros(1)
		markov_matrix_one = content_selection._build_markov_matrix(sim_matrix_one, bias_vec_one_empty, self.d)
		self.assertAlmostEqual(markov_matrix_one[0], expected_m_m_one)

		# Test that when sim_matrix has one element of 1 and bias_vec has 1
		# the resulting value of 1
		bias_vec_one = np.array([1.0])
		markov_matrix_one_one = content_selection._build_markov_matrix(sim_matrix_one, bias_vec_one, self.d)
		self.assertAlmostEqual(markov_matrix_one_one[0], 1.0)


	def test_power_method(self):

		# Test that when an empty matrix, returns an empty vec
		m_matrix_none = np.zeros((0,0))
		prob_vec_none = content_selection._power_method(m_matrix_none, self.epsilon)
		self.assertEqual(prob_vec_none.size, 0)

		# Test that when a matrix of a single value
		# returns a vec that sums to 1
		m_matrix_one = np.array([1.0])
		prob_vec_one = content_selection._power_method(m_matrix_one, self.epsilon)
		self.assertEqual(np.sum(prob_vec_one), 1.0)

		# Test that when a matrix of two values
		# returns a vec that sums to 1
		m_matrix_two = np.array([[0.5, 0.5], [0.3, 0.7]])
		prob_vec_two = content_selection._power_method(m_matrix_two, self.epsilon)
		self.assertEqual(np.sum(prob_vec_two), 1.0)


	def test_select_sentences(self):
		pass
		# TODO: test after changes are made to this function

		# check that length (number of words) is correct
		# check that score of sentences is ranked correctly
		# check that no sentence is above .5 cosine sim (excluding self)


	# NOTE: no test for select_content, since that needs to be an integration test
	# once we have a stable version and gold standard output

if __name__ == '__main__':
	unittest.main()
